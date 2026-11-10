"""配置領域 (許可タイル・余白・密度基準)。"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from ..core.models import LayoutDomain, ObstacleSet
from ..core.settings import Settings, offset_to_udim, udim_to_offset


def make_domain(settings: Settings, obstacles: Optional[ObstacleSet] = None,
                density_k: Optional[float] = None, density_source: str = "auto") -> LayoutDomain:
    gap = settings.gap_uv
    return LayoutDomain(tiles=settings.effective_tiles(), resolution=settings.resolution,
                        gap_uv=gap, edge_margin_uv=gap,
                        obstacles=obstacles or ObstacleSet(),
                        density_k=density_k, density_source=density_source)


def tile_origin(tile: int) -> Tuple[float, float]:
    u, v = udim_to_offset(tile)
    return float(u), float(v)


def tile_of_point(u: float, v: float) -> Optional[int]:
    if not (math.isfinite(u) and math.isfinite(v)) or u < 0 or v < 0 or u >= 10:
        return None
    return offset_to_udim(int(math.floor(u)), int(math.floor(v)))


def island_tile(uv: np.ndarray) -> Optional[int]:
    """島の全点が同一タイル内 (境界上の点は両側を許容) ならそのタイル。"""
    if len(uv) == 0 or not np.all(np.isfinite(uv)):
        return None
    lo = uv.min(axis=0)
    hi = uv.max(axis=0)
    tu, tv = math.floor(lo[0]), math.floor(lo[1])
    # 上端がちょうど整数上にある場合を同タイルとみなす
    if hi[0] > tu + 1 or hi[1] > tv + 1:
        return None
    if tu < 0 or tv < 0 or tu > 9:
        return None
    return offset_to_udim(int(tu), int(tv))


def margin_ok(uv: np.ndarray, tile: int, margin: float, tol: float = 1e-9) -> bool:
    ou, ov = tile_origin(tile)
    lo = uv.min(axis=0) - (ou, ov)
    hi = uv.max(axis=0) - (ou, ov)
    return bool(lo[0] >= margin - tol and lo[1] >= margin - tol
                and hi[0] <= 1 - margin + tol and hi[1] <= 1 - margin + tol)
