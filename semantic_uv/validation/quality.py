"""比較指標 (設計仕様 8章)。

グループ分散・方向一致率は人間の可読性の代理指標であり、
可読性そのものを断定しない。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..analysis.shape_features import edge_lengths_world
from ..analysis.topology import Topology, uv_edge_continuous, uv_island_faces
from ..core.models import ObjectData, QualityMetrics
from ..core.settings import SoftWeights
from ..layout.orientation import angle_difference, up_gradient
from ..unwrap.distortion import tri_metrics


def compute_quality(objects: Sequence[ObjectData], topos: Dict[str, Topology],
                    uvs: Dict[str, np.ndarray], n_tiles: int, weights: SoftWeights,
                    face_filter: Optional[Dict[str, Set[int]]] = None,
                    target_k: Optional[float] = None,
                    groups: Optional[Dict[Tuple[str, int], int]] = None,
                    pairs: Optional[List[Tuple[Tuple[str, int], Tuple[str, int]]]] = None
                    ) -> QualityMetrics:
    """face_filter: 歪みを測る面 (None なら全面)。groups: (object, 代表面) -> group。
    pairs: 対応島 (対称・類似) の代表面ペア。"""
    q = QualityMetrics()
    ang_sum = area_sum = w_sum = 0.0
    max_ang = max_area = 0.0
    uv_area_total = 0.0
    world_total = 0.0
    seam_len = 0.0
    island_info: List[Tuple[str, int, float, float, np.ndarray]] = []  # obj, rep face, k, wa, centroid
    rep_angle: Dict[Tuple[str, int], float] = {}
    for obj in objects:
        topo = topos[obj.key]
        uv = uvs.get(obj.key)
        if uv is None or not np.all(np.isfinite(uv)):
            continue
        ff = face_filter.get(obj.key) if face_filter else None
        lengths = edge_lengths_world(topo)
        for e in range(obj.n_edges):
            fs = topo.edge_faces[e]
            if len(fs) == 2 and (ff is None or fs[0] in ff or fs[1] in ff):
                if not uv_edge_continuous(topo, uv, e):
                    seam_len += float(lengths[e])
        comps = uv_island_faces(topo, uv, range(obj.n_faces))
        w_all = obj.world_co()
        for comp in comps:
            if ff is not None and not any(f in ff for f in comp):
                tids = [t for f in comp for t in topo.face_tris(f)]
                if tids:
                    uv_area_total += float(np.abs(_sa(uv[topo.tris[tids]])).sum())
                continue
            tids = np.asarray([t for f in comp for t in topo.face_tris(f)], dtype=np.int64)
            if len(tids) == 0:
                continue
            tris = topo.tris[tids]
            m = tri_metrics(w_all[obj.loop_vert[tris]], uv[tris])
            wa = float(m.world_area.sum())
            ua = float(np.abs(m.uv_signed_area).sum())
            uv_area_total += ua
            if wa <= 0:
                continue
            world_total += wa
            k = ua / wa
            ad = m.angle_distortion
            ar = m.area_distortion(k)
            ang_sum += float((ad * m.world_area).sum())
            area_sum += float((ar * m.world_area).sum())
            w_sum += wa
            max_ang = max(max_ang, float(ad.max()))
            max_area = max(max_area, float(ar.max()))
            cen = uv[tris].reshape(-1, 2).mean(axis=0)
            island_info.append((obj.key, comp[0], k, wa, cen))
            loops = tris.reshape(-1)
            g = up_gradient(uv[loops], w_all[obj.loop_vert[loops]])
            if g is not None:
                rep_angle[(obj.key, comp[0])] = math.atan2(g[1], g[0])
    q.island_count = len(island_info)
    if w_sum > 0:
        q.mean_angle_distortion = ang_sum / w_sum
        q.mean_area_distortion = area_sum / w_sum
    q.max_angle_distortion = max_ang
    q.max_area_distortion = max_area
    q.fill_ratio = uv_area_total / max(n_tiles, 1)
    q.seam_length = seam_len
    if island_info:
        if target_k is None:
            target_k = sum(i[2] * i[3] for i in island_info) / max(sum(i[3] for i in island_info), 1e-30)
        q.density_deviation_max = max(abs(math.sqrt(i[2] / target_k) - 1.0) for i in island_info) \
            if target_k > 0 else 0.0
    if groups and island_info:
        by_group: Dict[int, List[np.ndarray]] = {}
        for key, face, _, _, cen in island_info:
            g = groups.get((key, face))
            if g is not None:
                by_group.setdefault(g, []).append(cen)
        disp = []
        for cs in by_group.values():
            if len(cs) > 1:
                c = np.mean(cs, axis=0)
                disp.append(float(np.mean([np.linalg.norm(x - c) for x in cs])))
        q.group_dispersion = float(np.mean(disp)) / math.sqrt(max(n_tiles, 1)) if disp else 0.0
    if pairs:
        agree = total = 0
        for a, b in pairs:
            if a in rep_angle and b in rep_angle:
                total += 1
                if angle_difference(rep_angle[a], rep_angle[b]) <= math.radians(10):
                    agree += 1
        q.orientation_agreement = agree / total if total else 1.0
    sqrt_area = math.sqrt(world_total) if world_total > 0 else 1.0
    q.total_cost = (weights.distortion * (q.mean_angle_distortion + 0.5 * q.mean_area_distortion)
                    + weights.seam_length * q.seam_length / sqrt_area * 0.1
                    + weights.island_count * q.island_count * 0.01
                    + weights.packing * (1.0 - min(q.fill_ratio, 1.0))
                    + weights.readability * (q.group_dispersion + (1.0 - q.orientation_agreement)))
    return q


def _sa(t: np.ndarray) -> np.ndarray:
    return 0.5 * ((t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1])
                  - (t[:, 1, 1] - t[:, 0, 1]) * (t[:, 2, 0] - t[:, 0, 0]))
