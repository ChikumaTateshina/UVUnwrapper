"""二面角・凹凸・頂点曲率 (角度欠損)。ワールド空間で計算する。"""
from __future__ import annotations

import math

import numpy as np

from .topology import Topology


def edge_dihedral(topo: Topology):
    """返り値: (dihedral[E] ラジアン 0=平坦, concavity[E] +凸/-凹)。
    境界・非多様体エッジは nan。"""
    obj = topo.obj
    E = obj.n_edges
    normals = topo.face_normals_world()
    w = obj.world_co()
    dihedral = np.full(E, np.nan)
    concavity = np.full(E, np.nan)
    centers = np.zeros((obj.n_faces, 3))
    for f in range(obj.n_faces):
        centers[f] = w[obj.loop_vert[list(obj.face_loops(f))]].mean(axis=0)
    for e in range(E):
        fs = topo.edge_faces[e]
        if len(fs) != 2:
            continue
        n1, n2 = normals[fs[0]], normals[fs[1]]
        d = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
        dihedral[e] = math.acos(d)
        # 凸: 相手の面中心が自分の法線の裏側
        side = float(np.dot(centers[fs[1]] - centers[fs[0]], n1))
        concavity[e] = 1.0 if side <= 0 else -1.0
    return dihedral, concavity


def vertex_angle_defect(topo: Topology) -> np.ndarray:
    """ガウス曲率の離散近似 (2π - 角度和)。境界頂点は π - 角度和。"""
    obj = topo.obj
    w = obj.world_co()
    V = len(w)
    angle_sum = np.zeros(V)
    for tri in topo.tris:
        vs = obj.loop_vert[tri]
        p = w[vs]
        for i in range(3):
            a = p[(i + 1) % 3] - p[i]
            b = p[(i + 2) % 3] - p[i]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na == 0 or nb == 0:
                continue
            angle_sum[vs[i]] += math.acos(float(np.clip(np.dot(a, b) / (na * nb), -1, 1)))
    boundary = np.zeros(V, dtype=bool)
    for e, fs in enumerate(topo.edge_faces):
        if len(fs) == 1:
            boundary[obj.edges[e]] = True
    full = np.where(boundary, math.pi, 2 * math.pi)
    return full - angle_sum
