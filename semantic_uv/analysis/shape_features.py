"""面集合 (Chart/島) の形状特徴量と、2D島の主軸・外接矩形。"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

from .topology import Topology


def faces_world_area(topo: Topology, faces: Sequence[int]) -> float:
    areas = topo.world_face_areas()
    return float(areas[list(faces)].sum()) if len(faces) else 0.0


def boundary_edges_of(topo: Topology, faces: Sequence[int]) -> List[int]:
    fset = set(int(f) for f in faces)
    out = []
    seen = set()
    for f in fset:
        for e in topo.face_edges(f):
            if e in seen:
                continue
            seen.add(e)
            inside = sum(1 for g in topo.edge_faces[e] if g in fset)
            if inside == 1 or len(topo.edge_faces[e]) == 1:
                out.append(e)
    return sorted(out)


def edge_lengths_world(topo: Topology) -> np.ndarray:
    w = topo.obj.world_co()
    e = topo.obj.edges
    if len(e) == 0:
        return np.zeros(0)
    return np.linalg.norm(w[e[:, 0]] - w[e[:, 1]], axis=1)


def shape_descriptor(topo: Topology, faces: Sequence[int]) -> np.ndarray:
    """回転不変な記述子: [面積, 周長, 主軸長3つ(降順), 面数, 平均曲がり]。
    類似形状の候補絞り込みに用いる。"""
    obj = topo.obj
    faces = list(faces)
    area = faces_world_area(topo, faces)
    lengths = edge_lengths_world(topo)
    perim = float(sum(lengths[e] for e in boundary_edges_of(topo, faces)))
    verts = sorted(set(int(v) for f in faces for v in obj.loop_vert[list(obj.face_loops(f))]))
    pts = obj.world_co()[verts]
    if len(pts) >= 2:
        c = pts - pts.mean(axis=0)
        ev = np.sort(np.linalg.eigvalsh(c.T @ c / len(pts)))[::-1]
        ext = np.sqrt(np.maximum(ev, 0))
    else:
        ext = np.zeros(3)
    normals = topo.face_normals_world()[faces]
    mean_n = normals.mean(axis=0) if len(faces) else np.zeros(3)
    bend = 1.0 - float(np.linalg.norm(mean_n))
    return np.array([area, perim, ext[0], ext[1], ext[2], float(len(faces)), bend])


def principal_axes_world(topo: Topology, faces: Sequence[int]):
    obj = topo.obj
    verts = sorted(set(int(v) for f in faces for v in obj.loop_vert[list(obj.face_loops(f))]))
    pts = obj.world_co()[verts]
    c = pts - pts.mean(axis=0)
    vals, vecs = np.linalg.eigh(c.T @ c / max(len(pts), 1))
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


# ---------------------------------------------------------------- 2D ---
def convex_hull_2d(pts: np.ndarray) -> np.ndarray:
    """Andrew の単調連鎖。"""
    p = np.unique(np.round(pts, 12), axis=0)
    if len(p) <= 2:
        return p
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower: List = []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    upper: List = []
    for q in p[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.array(lower[:-1] + upper[:-1])


def min_area_rect_angle(pts: np.ndarray) -> Tuple[float, float]:
    """最小面積外接矩形の回転角と、その主軸信頼度 (0..1)。
    角度は矩形の長辺が U 軸に揃う回転量 (ラジアン)。"""
    hull = convex_hull_2d(pts)
    if len(hull) < 3:
        return 0.0, 0.0
    best = (float("inf"), 0.0, 1.0, 1.0)
    for i in range(len(hull)):
        d = hull[(i + 1) % len(hull)] - hull[i]
        n = math.hypot(d[0], d[1])
        if n == 0:
            continue
        ang = -math.atan2(d[1], d[0])
        r = rotation(ang)
        q = hull @ r.T
        w, h = np.ptp(q[:, 0]), np.ptp(q[:, 1])
        area = w * h
        if area < best[0] - 1e-15:
            best = (area, ang, w, h)
    _, ang, w, h = best
    if h > w:
        ang += math.pi / 2
        w, h = h, w
    confidence = 0.0 if w == 0 else float(1.0 - h / w)
    return ang, confidence


def rotation(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])
