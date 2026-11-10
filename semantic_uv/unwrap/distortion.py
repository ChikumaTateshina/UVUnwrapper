"""ワールド空間三角形と UV 三角形の対応から歪みを測る。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TriMetrics:
    sigma_max: np.ndarray
    sigma_min: np.ndarray
    uv_signed_area: np.ndarray
    world_area: np.ndarray

    @property
    def angle_distortion(self) -> np.ndarray:
        """0 = 等角。1 - σmin/σmax。"""
        s = np.where(self.sigma_max > 0, self.sigma_max, 1.0)
        return np.where(self.sigma_max > 0, 1.0 - self.sigma_min / s, 1.0)

    def area_distortion(self, k: float) -> np.ndarray:
        """k = 島全体の A_uv/A_world に対する各三角形の比の偏差 (0 = 均一)。"""
        w = np.where(self.world_area > 0, self.world_area, 1.0)
        r = np.abs(self.uv_signed_area) / w / (k if k > 0 else 1.0)
        r = np.where(r > 0, r, 1e-12)
        return np.maximum(r, 1.0 / r) - 1.0


def tri_metrics(world: np.ndarray, uv: np.ndarray) -> TriMetrics:
    """world: (T,3,3), uv: (T,3,2)"""
    if len(world) == 0:
        z = np.zeros(0)
        return TriMetrics(z, z, z, z)
    e1 = world[:, 1] - world[:, 0]
    e2 = world[:, 2] - world[:, 0]
    l1 = np.linalg.norm(e1, axis=1)
    n = np.cross(e1, e2)
    ln = np.linalg.norm(n, axis=1)
    l1s = np.where(l1 > 0, l1, 1.0)
    x = e1 / l1s[:, None]
    y = np.cross(n / np.where(ln > 0, ln, 1.0)[:, None], x)
    # 3D 局所座標
    p1 = np.stack([l1, np.zeros_like(l1)], axis=1)
    p2 = np.stack([np.einsum("ij,ij->i", e2, x), np.einsum("ij,ij->i", e2, y)], axis=1)
    q1 = uv[:, 1] - uv[:, 0]
    q2 = uv[:, 2] - uv[:, 0]
    # J * [p1 p2] = [q1 q2]
    Pm = np.stack([p1, p2], axis=2)        # (T,2,2) 列ベクトル
    Qm = np.stack([q1, q2], axis=2)
    det = Pm[:, 0, 0] * Pm[:, 1, 1] - Pm[:, 0, 1] * Pm[:, 1, 0]
    ok = np.abs(det) > 0
    inv = np.zeros_like(Pm)
    ds = np.where(ok, det, 1.0)
    inv[:, 0, 0] = Pm[:, 1, 1] / ds
    inv[:, 0, 1] = -Pm[:, 0, 1] / ds
    inv[:, 1, 0] = -Pm[:, 1, 0] / ds
    inv[:, 1, 1] = Pm[:, 0, 0] / ds
    J = Qm @ inv
    sv = np.linalg.svd(J, compute_uv=False)
    smax = np.where(ok, sv[:, 0], 0.0)
    smin = np.where(ok, sv[:, 1], 0.0)
    signed = 0.5 * (q1[:, 0] * q2[:, 1] - q1[:, 1] * q2[:, 0])
    return TriMetrics(smax, smin, signed, 0.5 * ln)


def signed_areas(uv_tris: np.ndarray) -> np.ndarray:
    """uv_tris: (T,3,2)"""
    if len(uv_tris) == 0:
        return np.zeros(0)
    a, b, c = uv_tris[:, 0], uv_tris[:, 1], uv_tris[:, 2]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
