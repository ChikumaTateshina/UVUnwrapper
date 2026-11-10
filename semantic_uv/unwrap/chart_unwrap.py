"""Chart 単位の展開と独立検証。

1つの Chart (面集合) を UV 頂点へ溶接し、バックエンドで展開し、
反転・自己重複・非有限を検査する。境界固定 (pinned) を与えると
固定値は最終的に元の数値で代入し直してから検証する。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..analysis.topology import Topology, weld_groups
from ..core.settings import UnwrapBackend
from ..validation.injectivity import island_self_overlap
from ..validation.overlaps import find_overlaps

# Chart 全体の面積に対しこの比率以下の三角形は「3D で面積0」とみなす
SOLID_REL = 1e-9
from . import backend as be
from .distortion import tri_metrics


@dataclass
class ChartResult:
    faces: List[int]
    loop_uv: Dict[int, Tuple[float, float]]   # loop -> uv
    valid: bool
    angle_distortion: float = 0.0             # 面積加重平均
    max_angle_distortion: float = 0.0
    area_distortion: float = 0.0
    world_area: float = 0.0
    uv_area: float = 0.0
    method: str = ""
    reason: str = ""
    loops: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


def chart_loops(topo: Topology, faces: Sequence[int]) -> np.ndarray:
    obj = topo.obj
    return np.asarray([l for f in faces for l in obj.face_loops(f)], dtype=np.int64)


def evaluate_chart(topo: Topology, faces: Sequence[int], loop_uv: np.ndarray,
                   require_positive: bool = True) -> Tuple[bool, str, float, float, float, float, float]:
    """loop_uv: (L,2) 全loop配列 (chart の loop のみ参照)。
    返り値: valid, reason, mean_angle, max_angle, area_dist, world_area, uv_area"""
    obj = topo.obj
    tids = [t for f in faces for t in topo.face_tris(f)]
    if not tids:
        return False, "三角形がありません", 1, 1, 1, 0, 0
    tris = topo.tris[tids]
    uvt = loop_uv[tris]
    if not np.all(np.isfinite(uvt)):
        return False, "非有限のUV座標", 1, 1, 1, 0, 0
    w = obj.world_co()[obj.loop_vert[tris]]
    m = tri_metrics(w, uvt)
    wa = float(m.world_area.sum())
    signed = m.uv_signed_area
    ua = float(np.abs(signed).sum())
    if wa <= topo.solid_threshold():
        # 3D で面積0の面だけの Chart: テクスチャに現れないため有限なら有効
        return True, "", 0.0, 0.0, 0.0, wa, ua
    if ua <= 0:
        return False, "退化したChart", 1, 1, 1, wa, ua
    k = ua / wa
    # 退化 (スケール相対)。3D で面積0の三角形は向き・退化の判定対象外。
    solid = (m.world_area > SOLID_REL * wa) & (m.world_area > topo.solid_threshold())
    deg = np.abs(signed) <= 1e-12 * ua + 0.0
    pos = (signed > 0) & solid
    neg = (signed < 0) & solid
    if require_positive:
        if np.any(solid & ~pos):
            return False, "反転または退化した三角形", 1, 1, 1, wa, ua
    else:
        if pos.any() and neg.any():
            return False, "向きが不整合 (折り返し)", 1, 1, 1, wa, ua
    if np.any(deg & solid):
        return False, "退化したUV三角形", 1, 1, 1, wa, ua
    major = 1.0 if not neg.any() else -1.0
    if np.any(~solid & (signed * major < 0)):
        # 面積0の面が裏返っている場合は境界単純性の前提が崩れるため厳密判定
        if find_overlaps(uvt[solid | (signed * major < 0)], max_pairs=1):
            return False, "Chart内の自己重複", 1, 1, 1, wa, ua
    elif island_self_overlap(uvt):   # 向きが揃っている前提の境界単純性判定
        return False, "Chart内の自己重複", 1, 1, 1, wa, ua
    ad = m.angle_distortion
    weights = m.world_area / wa
    mean_a = float(np.sum(ad * weights))
    max_a = float(ad.max())
    area_d = float(np.sum(m.area_distortion(k) * weights))
    return True, "", mean_a, max_a, area_d, wa, ua


def unwrap_charts(topo: Topology, charts: List[List[int]], cut_edges: Set[int],
                  backend: UnwrapBackend,
                  pinned_loops: Optional[Dict[int, Tuple[float, float]]] = None,
                  weld_uv: Optional[np.ndarray] = None,
                  initial_uv: Optional[np.ndarray] = None) -> List[ChartResult]:
    """複数 Chart をまとめて展開する (外部バックエンドは1回の呼び出しで処理)。
    pinned_loops: 固定する loop -> uv。固定を含む Chart は内蔵ソルバーで解く。
    weld_uv: 与えた場合、入力UVで連続なコーナーだけを溶接 (既存の分断を保持)。"""
    obj = topo.obj
    pinned_loops = pinned_loops or {}
    jobs: List[Tuple[be.UnwrapJob, np.ndarray, np.ndarray]] = []
    for faces in charts:
        group, ng = weld_groups(topo, faces, cut_edges, uv=weld_uv)
        job, rep = be.build_job(topo, faces, group, ng)
        loops = chart_loops(topo, faces)
        for l in loops:
            if int(l) in pinned_loops:
                job.pinned[int(group[l])] = pinned_loops[int(l)]
        if initial_uv is not None:
            init = np.zeros((ng, 2))
            for l in loops:
                init[group[l]] = initial_uv[l]
            job.initial = init
        jobs.append((job, group, loops))

    ext_results: List[Optional[np.ndarray]] = [None] * len(jobs)
    if backend != UnwrapBackend.BUILTIN_LSCM and be.external_available():
        free_idx = [i for i, (job, _, _) in enumerate(jobs) if not job.pinned]
        if free_idx:
            method = "ANGLE_BASED" if backend == UnwrapBackend.BLENDER_ANGLE_BASED else "CONFORMAL"
            res = be.unwrap_external([jobs[i][0] for i in free_idx], method)
            if res is not None:
                for i, r in zip(free_idx, res):
                    if r is not None and len(jobs[i][0].tris):
                        t = r[jobs[i][0].tris]
                        sa = ((t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1])
                              - (t[:, 1, 1] - t[:, 0, 1]) * (t[:, 2, 0] - t[:, 0, 0]))
                        if sa.sum() < 0:   # 鏡映で返った場合は向きを揃える
                            r = r.copy()
                            r[:, 0] = -r[:, 0]
                    ext_results[i] = r

    out: List[ChartResult] = []
    scratch = np.full((obj.n_loops, 2), np.nan)
    for ci, (job, group, loops) in enumerate(jobs):
        faces = charts[ci]
        attempts: List[Tuple[str, Optional[np.ndarray]]] = []
        if ext_results[ci] is not None:
            attempts.append((backend.value, ext_results[ci]))
        attempts.append(("LSCM", None))
        if job.pinned and len(job.pinned) >= 3:
            attempts.append(("TUTTE", None))
        best: Optional[ChartResult] = None
        last_reason = ""
        for name, uvv in attempts:
            if uvv is None:
                uvv = be.lscm(job) if name == "LSCM" else be.tutte(job)
            if uvv is None:
                last_reason = f"{name}: 解が得られません"
                continue
            uvv = np.asarray(uvv, dtype=np.float64)
            for g, (u, v) in job.pinned.items():
                uvv[g] = (u, v)
            scratch[loops] = uvv[group[loops]]
            valid, reason, ma, mx, ad, wa, ua = evaluate_chart(topo, faces, scratch,
                                                               require_positive=True)
            if not valid and job.pinned:
                # 固定境界を持つ入力は元の向きに従う (鏡映の既存UV)
                valid, reason, ma, mx, ad, wa, ua = evaluate_chart(topo, faces, scratch,
                                                                   require_positive=False)
            res = ChartResult(faces=list(faces),
                              loop_uv={int(l): (float(scratch[l, 0]), float(scratch[l, 1])) for l in loops},
                              valid=valid, angle_distortion=ma, max_angle_distortion=mx,
                              area_distortion=ad, world_area=wa, uv_area=ua, method=name,
                              reason=reason, loops=loops.copy())
            scratch[loops] = np.nan
            if valid:
                if best is None or (ma + ad) < (best.angle_distortion + best.area_distortion):
                    best = res
                if name != "LSCM" or not job.pinned:
                    break
            else:
                last_reason = f"{name}: {reason}"
        if best is None:
            best = ChartResult(faces=list(faces), loop_uv={}, valid=False, method="",
                               reason=last_reason, loops=loops.copy())
        out.append(best)
    return out
