"""島の方向決定。

主軸 (最小面積外接矩形) を U/V 軸に揃え、さらにワールド上方向
(Z、無ければ Y) が +V になる 90度候補を選ぶ。主軸の信頼度が低い場合は
元の方向を保つ (90度単位のみで上方向を合わせる)。
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..analysis.shape_features import min_area_rect_angle, rotation

CONFIDENCE_MIN = 0.08


def up_gradient(uv: np.ndarray, world: np.ndarray) -> Optional[np.ndarray]:
    """UV 平面上でワールド高さが増える方向。最小二乗の線形近似。"""
    if len(uv) < 3:
        return None
    A = np.c_[uv, np.ones(len(uv))]
    for axis in (2, 1):
        h = world[:, axis]
        if np.ptp(h) <= 1e-9 * (np.ptp(world, axis=0).max() + 1e-30):
            continue
        coef, *_ = np.linalg.lstsq(A, h, rcond=None)
        g = coef[:2]
        if np.linalg.norm(g) > 0:
            return g / np.linalg.norm(g)
    return None


def choose_rotation(uv: np.ndarray, world: np.ndarray, allow_free: bool = True) -> float:
    """島を回転させる角度 (ラジアン)。"""
    base = 0.0
    if allow_free:
        ang, conf = min_area_rect_angle(uv)
        if conf >= CONFIDENCE_MIN:
            base = ang
    g = up_gradient(uv, world)
    if g is None:
        return base
    best, best_dot = base, -2.0
    for k in range(4):
        th = base + k * math.pi / 2
        gr = rotation(th) @ g
        if gr[1] > best_dot + 1e-9:
            best, best_dot = th, gr[1]
    return best


def normalize_angle(theta: float) -> float:
    t = math.fmod(theta, 2 * math.pi)
    return t + 2 * math.pi if t < 0 else t


def angle_difference(a: float, b: float) -> float:
    d = abs(normalize_angle(a) - normalize_angle(b))
    return min(d, 2 * math.pi - d)
