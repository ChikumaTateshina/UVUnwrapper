"""固定障害物の管理と保守的ラスタ化。

障害物 (固定U・先に配置した島) を UV 三角形として保持し、
空間ハッシュで候補を絞ってから厳密な三角形距離で確定する。
空き探索用に、障害物と可動島をそれぞれ余白の半分だけ膨張させた
保守的な占有グリッドを作る (二重加算しない)。
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from ..core.models import ObstacleSet
from ..validation.overlaps import SpatialHash, build_hash, min_distance_violations, has_overlap_between
from .domain import tile_origin


class ObstacleIndex:
    def __init__(self, obstacles: ObstacleSet, gap: float):
        self.gap = gap
        self.tris = np.asarray(obstacles.tris, dtype=np.float64).reshape(-1, 3, 2)
        self.owners = list(obstacles.owners)
        self._hash: Optional[SpatialHash] = None

    @property
    def hash(self) -> SpatialHash:
        if self._hash is None:
            cell = max(self.gap * 4, 1.0 / 64)
            self._hash = build_hash(self.tris, cell=cell) if len(self.tris) else SpatialHash(cell)
        return self._hash

    def add(self, tris: np.ndarray, owner: Tuple[str, int]) -> None:
        tris = np.asarray(tris, dtype=np.float64).reshape(-1, 3, 2)
        base = len(self.tris)
        self.tris = np.concatenate([self.tris, tris]) if len(self.tris) else tris.copy()
        self.owners.extend([owner] * len(tris))
        h = self.hash
        lo = tris.min(axis=1)
        hi = tris.max(axis=1)
        for i in range(len(tris)):
            h.insert(base + i, lo[i], hi[i])

    def collides(self, tris: np.ndarray, gap: float) -> bool:
        """正面積重複、または余白不足があれば True。"""
        if len(self.tris) == 0:
            return False
        if gap > 0:
            return bool(min_distance_violations(tris, self.tris, gap, self.hash, first_only=True))
        return has_overlap_between(tris, self.tris, self.hash)

    def tris_in_tile(self, tile: int) -> np.ndarray:
        if len(self.tris) == 0:
            return self.tris
        ou, ov = tile_origin(tile)
        c = self.tris.mean(axis=1)
        mask = (c[:, 0] >= ou - 1e-9) & (c[:, 0] <= ou + 1 + 1e-9) & \
               (c[:, 1] >= ov - 1e-9) & (c[:, 1] <= ov + 1 + 1e-9)
        return self.tris[mask]


def _point_tri_dist(px: np.ndarray, py: np.ndarray, t: np.ndarray) -> np.ndarray:
    """多数の点と1三角形の距離 (内部は0)。"""
    a, b, c = t
    def edge_d(p0, p1):
        ex, ey = p1[0] - p0[0], p1[1] - p0[1]
        d = ex * ex + ey * ey
        if d == 0:
            return np.hypot(px - p0[0], py - p0[1])
        tt = np.clip(((px - p0[0]) * ex + (py - p0[1]) * ey) / d, 0, 1)
        return np.hypot(px - (p0[0] + tt * ex), py - (p0[1] + tt * ey))
    dist = np.minimum(np.minimum(edge_d(a, b), edge_d(b, c)), edge_d(c, a))
    def s(p1, p2):
        return (px - p2[0]) * (p1[1] - p2[1]) - (p1[0] - p2[0]) * (py - p2[1])
    d1, d2, d3 = s(a, b), s(b, c), s(c, a)
    neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    inside = ~(neg & pos)
    return np.where(inside, 0.0, dist)


def rasterize(tris: np.ndarray, origin: Tuple[float, float], cell: float, n: int,
              inflate: float) -> np.ndarray:
    """保守的ラスタ化: セル内のいずれかの点が三角形から inflate 未満なら占有。
    返り値: (n, n) bool, [ix, iy]。"""
    grid = np.zeros((n, n), dtype=bool)
    half_diag = cell * math.sqrt(2) / 2
    r = inflate + half_diag
    ox, oy = origin
    for t in tris:
        lo = t.min(axis=0) - r
        hi = t.max(axis=0) + r
        x0 = max(int(math.floor((lo[0] - ox) / cell)), 0)
        y0 = max(int(math.floor((lo[1] - oy) / cell)), 0)
        x1 = min(int(math.floor((hi[0] - ox) / cell)), n - 1)
        y1 = min(int(math.floor((hi[1] - oy) / cell)), n - 1)
        if x1 < x0 or y1 < y0:
            continue
        xs = ox + (np.arange(x0, x1 + 1) + 0.5) * cell
        ys = oy + (np.arange(y0, y1 + 1) + 0.5) * cell
        PX, PY = np.meshgrid(xs, ys, indexing="ij")
        d = _point_tri_dist(PX, PY, t)
        grid[x0:x1 + 1, y0:y1 + 1] |= d < r
    return grid
