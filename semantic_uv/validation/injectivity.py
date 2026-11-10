"""島単位の高速な重複判定。

向きの揃った (全三角形が同符号の) 島は局所的に単射であり、さらに境界が
単純 (境界線分同士に交差・接触がない) なら写像次数は高々1、すなわち島内に
正面積の重複はない。島同士は、境界が交差すれば重複、交差も接触もなければ
一方が他方に含まれる場合のみ重複 (内部点の包含判定で確定)。
境界の接触・共線重なりなど判定が曖昧な場合は、該当範囲の三角形を
厳密な交差面積で判定する (overlaps.find_overlaps / has_overlap_between)。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .overlaps import SpatialHash, find_overlaps, has_overlap_between

PROPER, TOUCH, NONE = 2, 1, 0


def boundary_segments(tris: np.ndarray) -> np.ndarray:
    """三角形辺のうち、座標が完全一致する相手を持たないもの (B,2,2)。"""
    if len(tris) == 0:
        return np.zeros((0, 2, 2))
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])   # (3T,2,2)
    a, b = e[:, 0], e[:, 1]
    swap = (a[:, 0] > b[:, 0]) | ((a[:, 0] == b[:, 0]) & (a[:, 1] > b[:, 1]))
    lo = np.where(swap[:, None], b, a)
    hi = np.where(swap[:, None], a, b)
    key = np.ascontiguousarray(np.c_[lo, hi])
    view = key.view(np.dtype((np.void, key.dtype.itemsize * 4))).ravel()
    _, inv, counts = np.unique(view, return_inverse=True, return_counts=True)
    single = counts[inv] == 1
    return e[single]


def _orient(p, q, r) -> float:
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def _on_segment(p, q, r) -> bool:
    return min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= r[1] <= max(p[1], q[1])


def _contact(a, b, c, d, shared_ok: bool) -> int:
    """線分 ab と cd の関係。PROPER = 内部で交差、TOUCH = 接触・共線重なり。
    shared_ok: 端点を厳密に共有するだけの接触 (隣接境界辺) を無視する。"""
    sa = (a[0] == c[0] and a[1] == c[1]) or (a[0] == d[0] and a[1] == d[1])
    sb = (b[0] == c[0] and b[1] == c[1]) or (b[0] == d[0] and b[1] == d[1])
    o1, o2 = _orient(a, b, c), _orient(a, b, d)
    o3, o4 = _orient(c, d, a), _orient(c, d, b)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return PROPER
    touch = ((o1 == 0 and _on_segment(a, b, c)) or (o2 == 0 and _on_segment(a, b, d))
             or (o3 == 0 and _on_segment(c, d, a)) or (o4 == 0 and _on_segment(c, d, b)))
    if not touch:
        return NONE
    if shared_ok and (sa or sb):
        # 端点共有のみ (共線で重ならない) なら隣接辺
        collinear = o1 == 0 and o2 == 0
        if not collinear:
            return NONE
        # 共線で端点共有: 逆方向に伸びていれば接触のみ
        if sa and sb:
            return TOUCH   # 同一辺が2回 (縮退)
        shared = a if sa else b
        other1 = b if sa else a
        other2 = d if ((c[0] == shared[0] and c[1] == shared[1])) else c
        v1 = (other1[0] - shared[0], other1[1] - shared[1])
        v2 = (other2[0] - shared[0], other2[1] - shared[1])
        return TOUCH if v1[0] * v2[0] + v1[1] * v2[1] > 0 else NONE
    return TOUCH


def _segment_hash(segs: np.ndarray, cell: float) -> SpatialHash:
    h = SpatialHash(cell)
    lo = segs.min(axis=1)
    hi = segs.max(axis=1)
    for i in range(len(segs)):
        h.insert(i, lo[i], hi[i])
    return h


def _cell_for(segs: np.ndarray) -> float:
    if len(segs) == 0:
        return 0.05
    ln = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)
    return max(float(np.median(ln)) * 2.0, 1e-5)


def self_contacts(segs: np.ndarray) -> int:
    """同一島の境界線分同士の最悪の関係。"""
    if len(segs) < 3:
        return NONE
    h = _segment_hash(segs, _cell_for(segs))
    lo = segs.min(axis=1)
    hi = segs.max(axis=1)
    worst = NONE
    for i in range(len(segs)):
        a, b = segs[i]
        for j in h.query(lo[i], hi[i]):
            if j <= i:
                continue
            if lo[j][0] > hi[i][0] or hi[j][0] < lo[i][0] or lo[j][1] > hi[i][1] or hi[j][1] < lo[i][1]:
                continue
            c = _contact(a, b, segs[j][0], segs[j][1], shared_ok=True)
            if c == PROPER:
                return PROPER
            worst = max(worst, c)
    return worst


def cross_contacts(sa: np.ndarray, sb: np.ndarray, hash_b: Optional[SpatialHash] = None) -> int:
    if len(sa) == 0 or len(sb) == 0:
        return NONE
    hb = hash_b or _segment_hash(sb, _cell_for(np.concatenate([sa, sb])))
    lo_b = sb.min(axis=1)
    hi_b = sb.max(axis=1)
    worst = NONE
    for i in range(len(sa)):
        a, b = sa[i]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        for j in hb.query(lo, hi):
            if lo_b[j][0] > hi[0] or hi_b[j][0] < lo[0] or lo_b[j][1] > hi[1] or hi_b[j][1] < lo[1]:
                continue
            c = _contact(a, b, sb[j][0], sb[j][1], shared_ok=False)
            if c == PROPER:
                return PROPER
            worst = max(worst, c)
    return worst


def point_in_region(p, segs: np.ndarray) -> bool:
    """偶奇規則 (穴を含む境界線分集合)。"""
    x, y = float(p[0]), float(p[1])
    a, b = segs[:, 0], segs[:, 1]
    cond = (a[:, 1] > y) != (b[:, 1] > y)
    if not cond.any():
        return False
    a, b = a[cond], b[cond]
    xi = a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])
    return bool(np.count_nonzero(xi > x) % 2 == 1)


def _interior_point(tris: np.ndarray) -> np.ndarray:
    sa = np.abs(0.5 * ((tris[:, 1, 0] - tris[:, 0, 0]) * (tris[:, 2, 1] - tris[:, 0, 1])
                       - (tris[:, 1, 1] - tris[:, 0, 1]) * (tris[:, 2, 0] - tris[:, 0, 0])))
    return tris[int(np.argmax(sa))].mean(axis=0)


# ------------------------------------------------------------------
def island_self_overlap(tris: np.ndarray) -> bool:
    """向きが揃っていることを前提に、島内の正面積重複の有無。"""
    if len(tris) < 2:
        return False
    c = self_contacts(boundary_segments(tris))
    if c == PROPER:
        return True
    if c == TOUCH:
        return bool(find_overlaps(tris, max_pairs=1))
    return False


def overlapping_island_pairs(islands: Sequence[np.ndarray], max_pairs: int = 50
                             ) -> List[Tuple[int, int]]:
    """島 (三角形配列) 同士で正面積重複がある組 (i<j)。各島は単射であること。"""
    n = len(islands)
    segs = [boundary_segments(t) for t in islands]
    lo = np.array([t.reshape(-1, 2).min(axis=0) if len(t) else (np.inf, np.inf) for t in islands])
    hi = np.array([t.reshape(-1, 2).max(axis=0) if len(t) else (-np.inf, -np.inf) for t in islands])
    order = np.argsort(lo[:, 0], kind="stable") if n else []
    out: List[Tuple[int, int]] = []
    active: List[int] = []
    hashes: Dict[int, SpatialHash] = {}
    for idx in order:
        i = int(idx)
        active = [j for j in active if hi[j][0] > lo[i][0]]
        for j in active:
            if hi[j][1] <= lo[i][1] or lo[j][1] >= hi[i][1]:
                continue
            a, b = (j, i) if j < i else (i, j)
            if b not in hashes:
                hashes[b] = _segment_hash(segs[b], _cell_for(segs[b]))
            c = cross_contacts(segs[a], segs[b], hashes[b])
            if c == PROPER:
                hit = True
            elif c == TOUCH:
                hit = has_overlap_between(islands[a], islands[b])
            else:
                hit = point_in_region(_interior_point(islands[a]), segs[b]) or \
                    point_in_region(_interior_point(islands[b]), segs[a])
            if hit:
                out.append((a, b))
                if len(out) >= max_pairs:
                    return sorted(out)
        active.append(i)
    return sorted(out)


def boundary_distance_below(sa: np.ndarray, sb: np.ndarray, gap: float, tol: float = 1e-9) -> bool:
    """重複のない2島の最短距離 (= 境界線分間の最短距離) が gap 未満か。"""
    if len(sa) == 0 or len(sb) == 0:
        return False
    hb = SpatialHash(max(gap * 2, _cell_for(sb)))
    lo_b = sb.min(axis=1)
    hi_b = sb.max(axis=1)
    for j in range(len(sb)):
        hb.insert(j, lo_b[j], hi_b[j])
    for i in range(len(sa)):
        a, b = sa[i]
        lo = np.minimum(a, b) - gap
        hi = np.maximum(a, b) + gap
        for j in hb.query(lo, hi):
            if lo_b[j][0] > hi[0] or hi_b[j][0] < lo[0] or lo_b[j][1] > hi[1] or hi_b[j][1] < lo[1]:
                continue
            if _seg_seg_dist(a, b, sb[j][0], sb[j][1]) < gap - tol:
                return True
    return False


def _seg_seg_dist(a, b, c, d) -> float:
    if _contact(a, b, c, d, shared_ok=False) != NONE:
        return 0.0

    def pd(p, s, e):
        ex, ey = e[0] - s[0], e[1] - s[1]
        dd = ex * ex + ey * ey
        t = 0.0 if dd == 0 else max(0.0, min(1.0, ((p[0] - s[0]) * ex + (p[1] - s[1]) * ey) / dd))
        return math.hypot(p[0] - s[0] - t * ex, p[1] - s[1] - t * ey)
    return min(pd(a, c, d), pd(b, c, d), pd(c, a, b), pd(d, a, b))
