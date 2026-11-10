"""接続・三角形化・非多様体／退化検出・UV連続性。Blender非依存。"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..core.models import ObjectData


class Topology:
    """ObjectData から導出される不変のトポロジー情報。"""

    def __init__(self, obj: ObjectData):
        self.obj = obj
        L, E = obj.n_loops, obj.n_edges
        start = obj.face_loop_start.astype(np.int64)
        total = obj.face_loop_total.astype(np.int64)
        self.loop_face = obj.loop_face()
        # next loop in face
        nxt = np.arange(L, dtype=np.int64) + 1
        ends = start + total
        nxt[ends - 1] = start
        self.loop_next = nxt
        prv = np.arange(L, dtype=np.int64) - 1
        prv[start] = ends - 1
        self.loop_prev = prv
        # edge -> [(face, loop)]  loop は edge の始点側コーナー
        self.edge_loops: List[List[int]] = [[] for _ in range(E)]
        for l in range(L):
            self.edge_loops[int(obj.loop_edge[l])].append(l)
        self.edge_faces: List[Tuple[int, ...]] = [
            tuple(int(self.loop_face[l]) for l in ls) for ls in self.edge_loops
        ]
        if obj.tri_loops is not None and len(obj.tri_loops):
            self.tris = np.asarray(obj.tri_loops, dtype=np.int64).reshape(-1, 3)
        else:
            self.tris = triangulate(obj, self)      # (T,3) loop index
        self.tri_face = self.loop_face[self.tris[:, 0]] if len(self.tris) else np.zeros(0, dtype=np.int64)
        self._face_tris: Optional[List[List[int]]] = None

    # -----------------------------------------------------------------
    def face_tris(self, f: int) -> List[int]:
        if self._face_tris is None:
            ft: List[List[int]] = [[] for _ in range(self.obj.n_faces)]
            for t, face in enumerate(self.tri_face):
                ft[int(face)].append(t)
            self._face_tris = ft
        return self._face_tris[f]

    def face_edges(self, f: int) -> List[int]:
        return [int(self.obj.loop_edge[l]) for l in self.obj.face_loops(f)]

    def neighbors(self, f: int, blocked: Optional[Set[int]] = None) -> Iterable[Tuple[int, int]]:
        """(edge, neighbor face)。多様体エッジ (2面) のみ。"""
        for e in self.face_edges(f):
            if blocked is not None and e in blocked:
                continue
            fs = self.edge_faces[e]
            if len(fs) == 2:
                other = fs[1] if fs[0] == f else fs[0]
                if other != f:
                    yield e, other

    def is_boundary_edge(self, e: int) -> bool:
        return len(self.edge_faces[e]) == 1

    def world_tri_areas(self) -> np.ndarray:
        cached = getattr(self, "_tri_areas", None)
        if cached is not None:
            return cached
        if len(self.tris) == 0:
            return np.zeros(0)
        w = self.obj.world_co()
        v = self.obj.loop_vert[self.tris]
        a, b, c = w[v[:, 0]], w[v[:, 1]], w[v[:, 2]]
        self._tri_areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        return self._tri_areas

    def solid_threshold(self) -> float:
        """Object 全体の面積に対し、これ以下の三角形は「3D で面積0」とみなす。"""
        return 1e-12 * float(self.world_tri_areas().sum())

    def world_face_areas(self) -> np.ndarray:
        areas = np.zeros(self.obj.n_faces)
        if len(self.tris):
            np.add.at(areas, self.tri_face, self.world_tri_areas())
        return areas

    def face_normals_world(self) -> np.ndarray:
        """ワールド空間法線。負スケールでも見た目の表側を向くよう補正。"""
        w = self.obj.world_co()
        normals = np.zeros((self.obj.n_faces, 3))
        if len(self.tris):
            v = self.obj.loop_vert[self.tris]
            a, b, c = w[v[:, 0]], w[v[:, 1]], w[v[:, 2]]
            np.add.at(normals, self.tri_face, np.cross(b - a, c - a))
        if self.obj.mirror_transform:
            normals = -normals
        n = np.linalg.norm(normals, axis=1)
        n[n == 0] = 1.0
        return normals / n[:, None]


# ---------------------------------------------------------------------
def triangulate(obj: ObjectData, topo: "Topology") -> np.ndarray:
    """面を loop index の三角形に分割する。元の面構成は変更しない。
    四角形は短い対角線、n角形は投影平面上の耳切り。"""
    tris: List[Tuple[int, int, int]] = []
    co = obj.co
    for f in range(obj.n_faces):
        loops = list(obj.face_loops(f))
        n = len(loops)
        if n < 3:
            continue
        if n == 3:
            tris.append((loops[0], loops[1], loops[2]))
            continue
        pts = co[obj.loop_vert[loops]]
        if n == 4:
            d02 = np.linalg.norm(pts[0] - pts[2])
            d13 = np.linalg.norm(pts[1] - pts[3])
            if d02 <= d13:
                tris.append((loops[0], loops[1], loops[2]))
                tris.append((loops[0], loops[2], loops[3]))
            else:
                tris.append((loops[0], loops[1], loops[3]))
                tris.append((loops[1], loops[2], loops[3]))
            continue
        tris.extend(_ear_clip(loops, pts))
    if not tris:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(tris, dtype=np.int64)


def _ear_clip(loops: List[int], pts: np.ndarray) -> List[Tuple[int, int, int]]:
    # Newell 法線で投影平面を決める
    normal = np.zeros(3)
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        normal += np.array([(a[1] - b[1]) * (a[2] + b[2]),
                            (a[2] - b[2]) * (a[0] + b[0]),
                            (a[0] - b[0]) * (a[1] + b[1])])
    axis = int(np.argmax(np.abs(normal)))
    keep = [i for i in range(3) if i != axis]
    p2 = pts[:, keep]
    if normal[axis] < 0:
        p2 = p2[:, ::-1]
    idx = list(range(n))
    out: List[Tuple[int, int, int]] = []
    guard = 0
    while len(idx) > 3 and guard < n * n:
        guard += 1
        clipped = False
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = p2[i0], p2[i1], p2[i2]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross <= 0:
                continue
            inside = False
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                if _point_in_tri(p2[j], a, b, c):
                    inside = True
                    break
            if inside:
                continue
            out.append((loops[i0], loops[i1], loops[i2]))
            idx.pop(k)
            clipped = True
            break
        if not clipped:
            break
    if len(idx) > 3:  # 自己交差など。扇形で代替 (退化検出で報告される)
        for k in range(1, len(idx) - 1):
            out.append((loops[idx[0]], loops[idx[k]], loops[idx[k + 1]]))
    elif len(idx) == 3:
        out.append((loops[idx[0]], loops[idx[1]], loops[idx[2]]))
    return out


def _point_in_tri(p, a, b, c) -> bool:
    def s(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1, d2, d3 = s(p, a, b), s(p, b, c), s(p, c, a)
    return d1 >= 0 and d2 >= 0 and d3 >= 0


# ---------------------------------------------------------------------
def connected_components(topo: Topology, faces: Sequence[int],
                         blocked_edges: Optional[Set[int]] = None) -> List[List[int]]:
    """面集合の連結成分。反復DFS、面番号の昇順で安定。"""
    allowed = set(int(f) for f in faces)
    seen: Set[int] = set()
    comps: List[List[int]] = []
    for f0 in sorted(allowed):
        if f0 in seen:
            continue
        stack = [f0]
        seen.add(f0)
        comp = []
        while stack:
            f = stack.pop()
            comp.append(f)
            for _, g in topo.neighbors(f, blocked_edges):
                if g in allowed and g not in seen:
                    seen.add(g)
                    stack.append(g)
        comps.append(sorted(comp))
    return comps


def nonmanifold_edges(topo: Topology) -> List[int]:
    """3面以上で共有されるエッジ、または同方向に辿られる (向き不整合) エッジ。"""
    bad: List[int] = []
    obj = topo.obj
    for e, ls in enumerate(topo.edge_loops):
        if len(ls) > 2:
            bad.append(e)
        elif len(ls) == 2:
            a, b = ls
            if obj.loop_vert[a] == obj.loop_vert[b]:
                bad.append(e)
    return bad


def degenerate_faces(topo: Topology, rel_eps: float = 1e-12) -> List[int]:
    """ワールド面積がスケール相対で0とみなせる面、重複頂点を含む面。"""
    obj = topo.obj
    w = obj.world_co()
    if len(w) == 0:
        return []
    diag = float(np.linalg.norm(w.max(axis=0) - w.min(axis=0)))
    eps = (diag * diag) * rel_eps
    areas = topo.world_face_areas()
    bad = []
    for f in range(obj.n_faces):
        verts = obj.loop_vert[list(obj.face_loops(f))]
        if len(set(verts.tolist())) != len(verts) or areas[f] <= eps or not np.isfinite(areas[f]):
            bad.append(f)
    return bad


def uv_island_faces(topo: Topology, uv: np.ndarray, faces: Sequence[int]) -> List[List[int]]:
    """実際のUV連続性 (両端コーナーのUVが完全一致) によるアイランド。
    同じ3D頂点であることを理由にコーナーを溶接しない。"""
    allowed = set(int(f) for f in faces)
    blocked = set()
    for e in range(topo.obj.n_edges):
        if not uv_edge_continuous(topo, uv, e):
            blocked.add(e)
    return connected_components(topo, sorted(allowed), blocked)


def uv_edge_continuous(topo: Topology, uv: np.ndarray, e: int) -> bool:
    ls = topo.edge_loops[e]
    if len(ls) != 2:
        return False
    a, b = ls
    a2, b2 = topo.loop_next[a], topo.loop_next[b]
    obj = topo.obj
    # 逆向きに辿る: a の始点 = b2 の頂点
    if obj.loop_vert[a] == obj.loop_vert[b2] and obj.loop_vert[a2] == obj.loop_vert[b]:
        return bool(np.array_equal(uv[a], uv[b2]) and np.array_equal(uv[a2], uv[b]))
    if obj.loop_vert[a] == obj.loop_vert[b] and obj.loop_vert[a2] == obj.loop_vert[b2]:
        return bool(np.array_equal(uv[a], uv[b]) and np.array_equal(uv[a2], uv[b2]))
    return False


def weld_groups(topo: Topology, faces: Sequence[int], cut_edges: Set[int],
                uv: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int]:
    """チャート内で同じ頂点を共有するコーナーを、切断されないエッジ経由で
    連結したものを1つのUV頂点とする。uv を与えた場合はUVが連続な場合のみ連結。
    返り値: loop -> group id (チャート外は -1), group 数。"""
    obj = topo.obj
    allowed = set(int(f) for f in faces)
    parent: Dict[int, int] = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    loops = []
    for f in sorted(allowed):
        for l in obj.face_loops(f):
            parent[l] = l
            loops.append(l)
    for f in sorted(allowed):
        for l in obj.face_loops(f):
            e = int(obj.loop_edge[l])
            if e in cut_edges:
                continue
            ls = topo.edge_loops[e]
            if len(ls) != 2:
                continue
            a, b = ls
            if topo.loop_face[a] not in allowed or topo.loop_face[b] not in allowed:
                continue
            if uv is not None and not uv_edge_continuous(topo, uv, e):
                continue
            a2, b2 = int(topo.loop_next[a]), int(topo.loop_next[b])
            if obj.loop_vert[a] == obj.loop_vert[b2]:
                union(a, b2)
                union(a2, b)
            else:
                union(a, b)
                union(a2, b2)
    group = -np.ones(obj.n_loops, dtype=np.int64)
    ids: Dict[int, int] = {}
    for l in loops:
        r = find(l)
        if r not in ids:
            ids[r] = len(ids)
        group[l] = ids[r]
    return group, len(ids)
