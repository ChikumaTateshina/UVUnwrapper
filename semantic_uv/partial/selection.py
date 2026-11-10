"""部分再展開のスコープ構築 (設計仕様 6.1 / 実装指示書 6.1)。

起動時の面選択から S (選択面) と U (残り、隠れ面を含む) を作る。
選択を勝手に拡張しない。UVコーナーは実際のUV連続性で扱い、
同じ頂点だからという理由で溶接しない。
"""
from __future__ import annotations

from typing import Dict, List, Set

import numpy as np

from ..analysis.topology import Topology, connected_components
from ..core.models import InputSnapshot, PartialScope
from .boundaries import selection_boundary


def build_partial_scope(snapshot: InputSnapshot, topos: Dict[str, Topology],
                        boundary_mode: str) -> PartialScope:
    sel: Dict[str, np.ndarray] = {}
    uns: Dict[str, np.ndarray] = {}
    writable: Dict[str, np.ndarray] = {}
    bedges: Dict[str, Set[int]] = {}
    bloops: Dict[str, np.ndarray] = {}
    pins: Dict[str, np.ndarray] = {}
    comps: Dict[str, List[List[int]]] = {}
    for obj in snapshot.iter_objects():
        topo = topos[obj.key]
        mask = obj.face_select & ~obj.face_hide
        S = np.nonzero(mask)[0].astype(np.int64)
        U = np.nonzero(~mask)[0].astype(np.int64)
        sel[obj.key] = S
        uns[obj.key] = U
        lf = obj.loop_face()
        w = mask[lf]
        writable[obj.key] = w
        be, bl = selection_boundary(topo, mask)
        bedges[obj.key] = be
        bloops[obj.key] = bl
        pins[obj.key] = obj.input_pin() & w
        comps[obj.key] = connected_components(topo, S.tolist()) if len(S) else []
    return PartialScope(selected_faces=sel, unselected_faces=uns, writable_loops=writable,
                        boundary_edges=bedges, boundary_loops=bloops, pinned_loops=pins,
                        components=comps, boundary_mode=boundary_mode)
