"""UV 三角形の正面積重複・距離判定と空間インデックス。

接する隣接三角形 (共有辺・共有頂点) は重複面積 0 であり、重複と混同しない。
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

OVERLAP_EPS_ABS = 1e-12
OVERLAP_EPS_REL = 1e-6


class SpatialHash:
    def __init__(self, cell: float):
        self.cell = max(float(cell), 1e-6)
        self.cells: Dict[Tuple[int, int], List[int]] = {}

    def _range(self, lo, hi):
        c = self.cell
        return (int(math.floor(lo[0] / c)), int(math.floor(lo[1] / c)),
                int(math.floor(hi[0] / c)), int(math.floor(hi[1] / c)))

    def insert(self, idx: int, lo, hi) -> None:
        x0, y0, x1, y1 = self._range(lo, hi)
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                self.cells.setdefault((x, y), []).append(idx)

    def query(self, lo, hi) -> Set[int]:
        x0, y0, x1, y1 = self._range(lo, hi)
        out: Set[int] = set()
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                out.update(self.cells.get((x, y), ()))
        return out


def choose_cell(tris: np.ndarray, fallback: float = 0.05) -> float:
    if len(tris) == 0:
        return fallback
    ext = tris.max(axis=1) - tris.min(axis=1)
    m = float(np.median(np.max(ext, axis=1)))
    return max(m * 2.0, 1e-4)


def build_hash(tris: np.ndarray, inflate: float = 0.0, cell: Optional[float] = None) -> SpatialHash:
    h = SpatialHash(cell or choose_cell(tris))
    lo = tris.min(axis=1) - inflate
    hi = tris.max(axis=1) + inflate
    for i in range(len(tris)):
        h.insert(i, lo[i], hi[i])
    return h


# ------------------------------------------------------------ 幾何 ---
def _area(poly: List[Tuple[float, float]]) -> float:
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def _ccw(t: np.ndarray) -> List[Tuple[float, float]]:
    pts = [(float(p[0]), float(p[1])) for p in t]
    return pts if _area(pts) >= 0 else pts[::-1]


def intersection_area(t1: np.ndarray, t2: np.ndarray) -> float:
    """2つの三角形の交差面積 (Sutherland–Hodgman)。"""
    subject = _ccw(t1)
    clip = _ccw(t2)
    out = subject
    for i in range(3):
        if not out:
            return 0.0
        ax, ay = clip[i]
        bx, by = clip[(i + 1) % 3]
        inp = out
        out = []

        def side(p):
            return (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax)
        n = len(inp)
        for j in range(n):
            p, q = inp[j], inp[(j + 1) % n]
            sp, sq = side(p), side(q)
            if sp >= 0:
                out.append(p)
                if sq < 0:
                    t = sp / (sp - sq)
                    out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
            elif sq >= 0:
                t = sp / (sp - sq)
                out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return abs(_area(out)) if len(out) >= 3 else 0.0


def _seg_point_dist(p, a, b) -> float:
    abx, aby = b[0] - a[0], b[1] - a[1]
    apx, apy = p[0] - a[0], p[1] - a[1]
    d = abx * abx + aby * aby
    t = 0.0 if d == 0 else max(0.0, min(1.0, (apx * abx + apy * aby) / d))
    dx, dy = apx - t * abx, apy - t * aby
    return math.hypot(dx, dy)


def _segments_intersect(a, b, c, d) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    o1, o2 = orient(a, b, c), orient(a, b, d)
    o3, o4 = orient(c, d, a), orient(c, d, b)
    if ((o1 > 0) != (o2 > 0)) and ((o3 > 0) != (o4 > 0)) and o1 != 0 and o2 != 0 and o3 != 0 and o4 != 0:
        return True
    return False


def _point_in(p, t) -> bool:
    a, b, c = t

    def s(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1, d2, d3 = s(p, a, b), s(p, b, c), s(p, c, a)
    neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (neg and pos)


def tri_distance(t1: np.ndarray, t2: np.ndarray) -> float:
    """2D 三角形間の最短距離 (交差・接触は0)。"""
    p = [(float(x[0]), float(x[1])) for x in t1]
    q = [(float(x[0]), float(x[1])) for x in t2]
    for i in range(3):
        for j in range(3):
            if _segments_intersect(p[i], p[(i + 1) % 3], q[j], q[(j + 1) % 3]):
                return 0.0
    if _point_in(p[0], q) or _point_in(q[0], p):
        return 0.0
    best = float("inf")
    for i in range(3):
        for j in range(3):
            best = min(best, _seg_point_dist(p[i], q[j], q[(j + 1) % 3]),
                       _seg_point_dist(q[j], p[i], p[(i + 1) % 3]))
    return best


def _shares_edge_opposite(t1: np.ndarray, t2: np.ndarray) -> bool:
    """厳密に同じ座標の2頂点を共有し、残りの頂点が辺の反対側にある (正面積重複なし)。"""
    shared = []
    for i in range(3):
        for j in range(3):
            if t1[i][0] == t2[j][0] and t1[i][1] == t2[j][1]:
                shared.append((i, j))
    if len(shared) != 2:
        return False
    (i1, j1), (i2, j2) = shared
    a, b = t1[i1], t1[i2]
    o1 = t1[3 - i1 - i2]
    o2 = t2[3 - j1 - j2]

    def side(p):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    s1, s2 = side(o1), side(o2)
    return (s1 > 0 and s2 < 0) or (s1 < 0 and s2 > 0)


def overlap_eps(a1: float, a2: float) -> float:
    return max(OVERLAP_EPS_ABS, OVERLAP_EPS_REL * min(a1, a2))


def find_overlaps(tris: np.ndarray, max_pairs: int = 1000) -> List[Tuple[int, int, float]]:
    """正面積で重複する三角形ペア (i<j, 面積)。"""
    if len(tris) < 2:
        return []
    areas = np.abs(0.5 * ((tris[:, 1, 0] - tris[:, 0, 0]) * (tris[:, 2, 1] - tris[:, 0, 1])
                          - (tris[:, 1, 1] - tris[:, 0, 1]) * (tris[:, 2, 0] - tris[:, 0, 0])))
    lo = tris.min(axis=1)
    hi = tris.max(axis=1)
    h = build_hash(tris)
    out: List[Tuple[int, int, float]] = []
    for i in range(len(tris)):
        for j in h.query(lo[i], hi[i]):
            if j <= i:
                continue
            if lo[j][0] >= hi[i][0] or hi[j][0] <= lo[i][0] or lo[j][1] >= hi[i][1] or hi[j][1] <= lo[i][1]:
                continue
            if _shares_edge_opposite(tris[i], tris[j]):
                continue
            a = intersection_area(tris[i], tris[j])
            if a > overlap_eps(areas[i], areas[j]):
                out.append((i, j, a))
                if len(out) >= max_pairs:
                    return out
    return out


def min_distance_violations(tris_a: np.ndarray, tris_b: np.ndarray, gap: float,
                            hash_b: Optional[SpatialHash] = None,
                            tol: float = 1e-9, first_only: bool = True) -> List[Tuple[int, int, float]]:
    """tris_a の各三角形と tris_b の三角形の距離が gap 未満の組。"""
    if len(tris_a) == 0 or len(tris_b) == 0:
        return []
    if hash_b is None:
        hash_b = build_hash(tris_b)
    lo_b = tris_b.min(axis=1)
    hi_b = tris_b.max(axis=1)
    out = []
    for i in range(len(tris_a)):
        lo = tris_a[i].min(axis=0) - gap
        hi = tris_a[i].max(axis=0) + gap
        for j in sorted(hash_b.query(lo, hi)):
            if lo_b[j][0] > hi[0] or hi_b[j][0] < lo[0] or lo_b[j][1] > hi[1] or hi_b[j][1] < lo[1]:
                continue
            d = tri_distance(tris_a[i], tris_b[j])
            if d < gap - tol:
                out.append((i, j, d))
                if first_only:
                    return out
    return out


def has_overlap_between(tris_a: np.ndarray, tris_b: np.ndarray,
                        hash_b: Optional[SpatialHash] = None) -> bool:
    if len(tris_a) == 0 or len(tris_b) == 0:
        return False
    if hash_b is None:
        hash_b = build_hash(tris_b)
    areas_b = np.abs(0.5 * ((tris_b[:, 1, 0] - tris_b[:, 0, 0]) * (tris_b[:, 2, 1] - tris_b[:, 0, 1])
                            - (tris_b[:, 1, 1] - tris_b[:, 0, 1]) * (tris_b[:, 2, 0] - tris_b[:, 0, 0])))
    for i in range(len(tris_a)):
        t = tris_a[i]
        a_i = abs(0.5 * ((t[1, 0] - t[0, 0]) * (t[2, 1] - t[0, 1]) - (t[1, 1] - t[0, 1]) * (t[2, 0] - t[0, 0])))
        for j in hash_b.query(t.min(axis=0), t.max(axis=0)):
            if _shares_edge_opposite(t, tris_b[j]):
                continue
            if intersection_area(t, tris_b[j]) > overlap_eps(a_i, areas_b[j]):
                return True
    return False
