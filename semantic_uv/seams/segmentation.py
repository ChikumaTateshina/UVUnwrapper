"""Chart 分割。

スコアの低い (切りたくない) エッジから順に面を統合する Kruskal 型の領域成長。
- FORBID エッジは最初に必ず統合 (切断禁止)。
- REQUIRE / 保護シーム / 呼び出し側指定の切断エッジは統合しない。
- 法線コーンの半角が上限を超える統合はしない。
- マテリアル境界は有力な候補として後回しにし、断片化を避ける場合のみ統合 (R-003)。
同順位はエッジ番号でタイブレークし、結果は決定的。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

from ..analysis.topology import Topology
from ..core.models import EdgeFeatures
from .constraints import ResolvedConstraints

TINY_REGION_FRACTION = 0.02


class _Regions:
    def __init__(self, faces: Sequence[int], normals: np.ndarray, areas: np.ndarray):
        self.parent = {int(f): int(f) for f in faces}
        self.nsum = {int(f): normals[f] * max(areas[f], 1e-30) for f in faces}
        self.cone = {int(f): 0.0 for f in faces}
        self.area = {int(f): float(areas[f]) for f in faces}

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def avg(self, r: int) -> np.ndarray:
        n = self.nsum[r]
        ln = np.linalg.norm(n)
        return n / ln if ln > 0 else n

    def merged_cone(self, ra: int, rb: int) -> float:
        na, nb = self.avg(ra), self.avg(rb)
        s = self.nsum[ra] + self.nsum[rb]
        ln = np.linalg.norm(s)
        if ln == 0:
            return math.pi
        nm = s / ln
        ca = math.acos(float(np.clip(np.dot(nm, na), -1, 1))) + self.cone[ra]
        cb = math.acos(float(np.clip(np.dot(nm, nb), -1, 1))) + self.cone[rb]
        return max(ca, cb)

    def union(self, ra: int, rb: int, cone: Optional[float] = None) -> int:
        if cone is None:
            cone = self.merged_cone(ra, rb)
        keep, drop = (ra, rb) if ra < rb else (rb, ra)
        self.parent[drop] = keep
        self.nsum[keep] = self.nsum[keep] + self.nsum[drop]
        self.cone[keep] = cone
        self.area[keep] += self.area[drop]
        return keep


def segment_charts(topo: Topology, feats: EdgeFeatures, rc: ResolvedConstraints,
                   faces: Sequence[int], angle_limit_deg: float,
                   extra_cut: Optional[Set[int]] = None) -> List[List[int]]:
    faces = sorted(int(f) for f in faces)
    fset = set(faces)
    extra_cut = extra_cut or set()
    normals = topo.face_normals_world()
    areas = topo.world_face_areas()
    reg = _Regions(faces, normals, areas)
    limit = math.radians(angle_limit_deg)

    interior: List[int] = []
    for f in faces:
        for e in topo.face_edges(f):
            fs = topo.edge_faces[e]
            if len(fs) == 2 and fs[0] in fset and fs[1] in fset and fs[0] != fs[1]:
                interior.append(e)
    interior = sorted(set(interior))

    # 1) FORBID は無条件で統合
    for e in interior:
        if e in rc.must_join:
            a, b = topo.edge_faces[e]
            ra, rb = reg.find(a), reg.find(b)
            if ra != rb:
                reg.union(ra, rb)

    blocked = rc.must_cut | extra_cut
    candidates = [e for e in interior if e not in blocked and e not in rc.must_join]
    plain = [e for e in candidates if not feats.material_boundary[e]]
    material = [e for e in candidates if feats.material_boundary[e]]

    def key(e):
        return (float(feats.score[e]), e)

    # 2) 通常エッジ
    for e in sorted(plain, key=key):
        a, b = topo.edge_faces[e]
        ra, rb = reg.find(a), reg.find(b)
        if ra == rb:
            continue
        cone = reg.merged_cone(ra, rb)
        if cone <= limit:
            reg.union(ra, rb, cone)

    # 3) マテリアル境界は小断片の解消時のみ統合
    total = float(sum(areas[f] for f in faces)) or 1.0
    for e in sorted(material, key=key):
        a, b = topo.edge_faces[e]
        ra, rb = reg.find(a), reg.find(b)
        if ra == rb:
            continue
        if min(reg.area[ra], reg.area[rb]) > TINY_REGION_FRACTION * total:
            continue
        cone = reg.merged_cone(ra, rb)
        if cone <= limit:
            reg.union(ra, rb, cone)

    groups: Dict[int, List[int]] = {}
    for f in faces:
        groups.setdefault(reg.find(f), []).append(f)
    return [sorted(g) for _, g in sorted(groups.items(), key=lambda kv: min(kv[1]))]


def chart_cut_edges(topo: Topology, charts: List[List[int]]) -> Set[int]:
    """Chart 間のエッジ (多様体エッジのうち両側が異なる Chart)。"""
    owner: Dict[int, int] = {}
    for ci, ch in enumerate(charts):
        for f in ch:
            owner[f] = ci
    cuts: Set[int] = set()
    for ci, ch in enumerate(charts):
        for f in ch:
            for e in topo.face_edges(f):
                fs = topo.edge_faces[e]
                if len(fs) == 2:
                    a, b = fs
                    if a in owner and b in owner and owner[a] != owner[b]:
                        cuts.add(e)
    return cuts
