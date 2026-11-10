"""選択境界と、境界固定モードで固定するコーナーの抽出。"""
from __future__ import annotations

from typing import Sequence, Set, Tuple

import numpy as np

from ..analysis.topology import Topology, uv_edge_continuous


def selection_boundary(topo: Topology, selected: np.ndarray) -> Tuple[Set[int], np.ndarray]:
    """S/U をまたぐエッジと、S 側のメッシュ外周 (片面エッジ) を境界とする。
    返り値: (境界エッジ集合, S側境界コーナー bool[L])。
    境界コーナー = 境界エッジの端点にある S 面のコーナー。"""
    obj = topo.obj
    edges: Set[int] = set()
    for e, fs in enumerate(topo.edge_faces):
        if not fs:
            continue
        s = [bool(selected[f]) for f in fs]
        if any(s) and (not all(s) or len(fs) == 1):
            edges.add(e)
    bverts = np.zeros(len(obj.co), dtype=bool)
    for e in edges:
        bverts[obj.edges[e]] = True
    lf = obj.loop_face()
    loops = selected[lf] & bverts[obj.loop_vert]
    return edges, loops


def keep_boundary_fixed_loops(topo: Topology, faces: Sequence[int], uv: np.ndarray,
                              must_cut: Set[int]) -> np.ndarray:
    """KEEP_BOUNDARY で固定するコーナー (bool[L])。

    再計算の単位 (入力UVで連続な S の部分島 faces) の境界にある全コーナーを固定する。
    境界 = 選択境界、メッシュ外周、入力UVの分断、REQUIRE/保護シーム。
    もともと分断されている境界は両側それぞれの元座標を維持する。"""
    obj = topo.obj
    fset = set(int(f) for f in faces)
    fixed = np.zeros(obj.n_loops, dtype=bool)
    for f in fset:
        for l in obj.face_loops(f):
            e = int(obj.loop_edge[l])
            fs = topo.edge_faces[e]
            interior = (len(fs) == 2 and fs[0] in fset and fs[1] in fset
                        and e not in must_cut and uv_edge_continuous(topo, uv, e))
            if not interior:
                fixed[l] = True
                fixed[int(topo.loop_next[l])] = True
    # 同じUV頂点 (溶接されるコーナー) に固定を伝播
    vert_fixed = {}
    for f in fset:
        for l in obj.face_loops(f):
            if fixed[l]:
                key = (int(obj.loop_vert[l]), float(uv[l, 0]), float(uv[l, 1]))
                vert_fixed[key] = True
    for f in fset:
        for l in obj.face_loops(f):
            key = (int(obj.loop_vert[l]), float(uv[l, 0]), float(uv[l, 1]))
            if key in vert_fixed:
                fixed[l] = True
    return fixed
