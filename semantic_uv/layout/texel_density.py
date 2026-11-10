"""テクセル密度 (設計仕様 7章)。

k = A_uv / A_world を全対象で統一する。長さ倍率は sqrt(k_target / k_current)。
解像度 R の正方形タイルでの密度は R * sqrt(k) px/ワールド単位。
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple


def k_of(uv_area: float, world_area: float) -> Optional[float]:
    if world_area <= 0 or uv_area <= 0 or not math.isfinite(uv_area / world_area):
        return None
    return uv_area / world_area


def length_scale(k_target: float, k_current: float) -> float:
    """面積比をそのまま長さ倍率にしない。"""
    return math.sqrt(k_target / k_current)


def px_per_unit(k: float, resolution: int) -> float:
    return resolution * math.sqrt(k)


def k_from_px(px: float, resolution: int) -> float:
    return (px / float(resolution)) ** 2


def representative_k(pairs: Iterable[Tuple[float, float]]) -> Optional[float]:
    """(uv_area, world_area) の面積加重代表値 = ΣA_uv / ΣA_world。"""
    su = sw = 0.0
    for ua, wa in pairs:
        if ua > 0 and wa > 0 and math.isfinite(ua) and math.isfinite(wa):
            su += ua
            sw += wa
    return su / sw if sw > 0 and su > 0 else None


def deviation(k: float, k_target: float) -> float:
    """長さ基準の密度偏差 |sqrt(k/k_target) - 1|。"""
    if k_target <= 0 or k <= 0:
        return float("inf")
    return abs(math.sqrt(k / k_target) - 1.0)
