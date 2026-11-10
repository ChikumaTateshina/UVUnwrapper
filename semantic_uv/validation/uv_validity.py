"""出力UVの Apply 必須条件 (設計仕様 8章)。

有限座標、非退化、向きの整合、自己交差なし、全対象横断の正面積重複なし、
許可領域内、(新規・再配置の島について) 余白。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..analysis.topology import Topology, uv_island_faces
from ..core.errors import ErrorCode, Target, ValidationReport
from ..core.models import ObjectData
from ..layout.domain import island_tile, margin_ok
from ..unwrap.distortion import signed_areas
from .injectivity import (boundary_distance_below, boundary_segments, island_self_overlap,
                          overlapping_island_pairs)
from .overlaps import find_overlaps


@dataclass
class OutIsland:
    object_key: str
    faces: List[int]
    tri_ids: np.ndarray     # topo.tris の index
    movable: bool           # 新規生成・再配置 (余白検査の対象)
    touches_writable: bool


def output_islands(obj: ObjectData, topo: Topology, uv: np.ndarray,
                   writable: np.ndarray, movable_faces: Optional[Set[int]] = None) -> List[OutIsland]:
    faces = list(range(obj.n_faces))
    comps = uv_island_faces(topo, uv, faces)
    lf = obj.loop_face()
    wf = set(int(f) for f in np.unique(lf[writable])) if writable.any() else set()
    out = []
    for comp in comps:
        tids = np.asarray([t for f in comp for t in topo.face_tris(f)], dtype=np.int64)
        touches = any(f in wf for f in comp)
        movable = touches if movable_faces is None else any(f in movable_faces for f in comp)
        out.append(OutIsland(obj.key, comp, tids, movable, touches))
    return out


def validate_uv(objects: Sequence[ObjectData], topos: Dict[str, Topology],
                uvs: Dict[str, np.ndarray], writables: Dict[str, np.ndarray],
                tiles: Sequence[int], gap: float, edge_margin: float,
                movable_faces: Optional[Dict[str, Set[int]]] = None,
                stage: str = "validate", check_gap: bool = True,
                generated_faces: Optional[Dict[str, Set[int]]] = None) -> ValidationReport:
    rep = ValidationReport()
    allowed = set(tiles)
    island_list: List[OutIsland] = []
    isl_tris: Dict[int, np.ndarray] = {}
    mixed: Dict[int, bool] = {}
    for obj in objects:
        topo = topos[obj.key]
        uv = uvs[obj.key]
        if not np.all(np.isfinite(uv)):
            bad = np.nonzero(~np.all(np.isfinite(uv), axis=1))[0]
            rep.error(ErrorCode.VALIDATION_FAILED, "非有限のUV座標があります", stage=stage,
                      targets=[Target(obj.key, face=int(obj.loop_face()[l]), corner=int(l)) for l in bad[:10]])
            continue
        mf = movable_faces.get(obj.key) if movable_faces is not None else None
        islands = output_islands(obj, topo, uv, writables[obj.key], mf)
        gen = generated_faces.get(obj.key, set()) if generated_faces else set()
        for isl in islands:
            ii = len(island_list)
            island_list.append(isl)
            if len(isl.tri_ids) == 0:
                continue
            tris = uv[topo.tris[isl.tri_ids]]
            sa = signed_areas(tris)
            tot = float(np.abs(sa).sum())
            # 退化 (スケール相対)
            world = topo.world_tri_areas()[isl.tri_ids]
            wsum = float(world.sum()) or 1.0
            # 3D で面積0の三角形は向き・退化の判定対象外
            solid = (world > 1e-9 * wsum) & (world > topo.solid_threshold())
            degenerate = (np.abs(sa) <= 1e-12 * max(tot, 1e-30)) & solid
            if (tot <= 0 and solid.any()) or degenerate.any():
                fl = [int(topo.tri_face[isl.tri_ids[i]]) for i in np.nonzero(degenerate)[0][:10]] or isl.faces[:5]
                rep.error(ErrorCode.VALIDATION_FAILED, "退化したUV三角形があります", stage=stage,
                          targets=[Target(obj.key, face=f) for f in fl],
                          advice="未選択部分の場合は該当面を選択に追加してください" if not isl.touches_writable else "")
            pos, neg = (sa > 0) & solid, (sa < 0) & solid
            major = -1.0 if neg.sum() > pos.sum() else 1.0
            flipped_zero = bool(np.any(~solid & (sa * major < 0)))
            if pos.any() and neg.any():
                fl = sorted(set(int(topo.tri_face[isl.tri_ids[i]]) for i in np.nonzero(neg if neg.sum() < pos.sum() else pos)[0][:10]))
                rep.error(ErrorCode.VALIDATION_FAILED, "島内で向きが不整合です (反転面)", stage=stage,
                          targets=[Target(obj.key, face=f) for f in fl])
            if any(f in gen for f in isl.faces) and neg.any() and not pos.any():
                rep.error(ErrorCode.VALIDATION_FAILED, "生成した島が鏡映されています", stage=stage,
                          targets=[Target(obj.key, face=isl.faces[0])])
            t = island_tile(tris.reshape(-1, 2))
            if t is None or t not in allowed:
                rep.error(ErrorCode.VALIDATION_FAILED,
                          "島が許可タイル外にあるか、タイル境界をまたいでいます" + (f" (タイル {t})" if t else ""),
                          stage=stage, targets=[Target(obj.key, face=isl.faces[0])],
                          advice="許可タイルの設定を確認してください")
            elif isl.movable and edge_margin > 0 and not margin_ok(tris.reshape(-1, 2), t, edge_margin):
                rep.error(ErrorCode.VALIDATION_FAILED, "タイル端の余白が不足しています", stage=stage,
                          targets=[Target(obj.key, face=isl.faces[0])])
            isl_tris[ii] = tris
            # 面積0の面が裏返っている島は境界単純性の前提が崩れるため厳密判定に回す
            mixed[ii] = bool(pos.any() and neg.any()) or flipped_zero
    if not island_list:
        return rep
    # 島内の自己重複 (向きが揃っていれば境界の単純性で判定)
    for ii, isl in enumerate(island_list):
        t = isl_tris.get(ii)
        if t is None or len(t) < 2:
            continue
        bad = bool(find_overlaps(t, max_pairs=1)) if mixed[ii] else island_self_overlap(t)
        if bad:
            rep.error(ErrorCode.VALIDATION_FAILED,
                      "島内に正の面積を持つUV重複 (自己交差) があります"
                      + ("" if isl.touches_writable else " (固定された未選択部分の既存重複)"),
                      stage=stage, targets=[Target(isl.object_key, face=isl.faces[0])],
                      advice="" if isl.touches_writable else "該当面を選択に追加して再展開してください")
    idx = [ii for ii in range(len(island_list)) if isl_tris.get(ii) is not None and len(isl_tris[ii])]
    pairs = overlapping_island_pairs([isl_tris[ii] for ii in idx], max_pairs=50)
    if pairs:
        targets = []
        fixed_only = True
        for a, b in pairs[:10]:
            ia, ib = island_list[idx[a]], island_list[idx[b]]
            targets += [Target(ia.object_key, face=ia.faces[0]), Target(ib.object_key, face=ib.faces[0])]
        for a, b in pairs:
            if island_list[idx[a]].touches_writable or island_list[idx[b]].touches_writable:
                fixed_only = False
        rep.error(ErrorCode.VALIDATION_FAILED,
                  "正の面積を持つUV重複があります" + (" (固定された未選択部分の既存重複)" if fixed_only else ""),
                  stage=stage, targets=targets,
                  advice="該当面を選択に追加して再展開してください" if fixed_only else "")
    if check_gap and gap > 0 and not pairs:
        segs = {ii: boundary_segments(isl_tris[ii]) for ii in idx}
        lo = {ii: isl_tris[ii].reshape(-1, 2).min(axis=0) for ii in idx}
        hi = {ii: isl_tris[ii].reshape(-1, 2).max(axis=0) for ii in idx}
        reported = 0
        for ii in idx:
            isl = island_list[ii]
            if not isl.movable:
                continue
            for jj in idx:
                if jj == ii or (island_list[jj].movable and jj < ii):
                    continue
                if np.any(lo[jj] > hi[ii] + gap) or np.any(hi[jj] < lo[ii] - gap):
                    continue
                if boundary_distance_below(segs[ii], segs[jj], gap):
                    other = island_list[jj]
                    rep.error(ErrorCode.VALIDATION_FAILED, "島間の余白が不足しています", stage=stage,
                              targets=[Target(isl.object_key, face=isl.faces[0]),
                                       Target(other.object_key, face=other.faces[0])])
                    reported += 1
                    break
            if reported >= 10:
                break
    return rep
