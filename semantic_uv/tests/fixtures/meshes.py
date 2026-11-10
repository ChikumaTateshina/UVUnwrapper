"""Blender なしで使う ObjectData フィクスチャ。"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from semantic_uv.core.models import InputSnapshot, MaterialInfo, ObjectData


def build_object(key: str, verts: Sequence[Sequence[float]], faces: Sequence[Sequence[int]],
                 matrix: Optional[np.ndarray] = None, uv: Optional[Dict[str, np.ndarray]] = None,
                 face_material: Optional[Sequence[int]] = None,
                 materials: Optional[List[MaterialInfo]] = None,
                 seams: Optional[Sequence[Tuple[int, int]]] = None,
                 mesh_key: Optional[str] = None) -> ObjectData:
    co = np.asarray(verts, dtype=np.float64)
    edge_index: Dict[Tuple[int, int], int] = {}
    edges: List[Tuple[int, int]] = []
    loop_vert, loop_edge, start, total = [], [], [], []
    for f in faces:
        start.append(len(loop_vert))
        total.append(len(f))
        for i, v in enumerate(f):
            w = f[(i + 1) % len(f)]
            k = (min(v, w), max(v, w))
            if k not in edge_index:
                edge_index[k] = len(edges)
                edges.append(k)
            loop_vert.append(v)
            loop_edge.append(edge_index[k])
    E = len(edges)
    seam_arr = np.zeros(E, dtype=bool)
    for a, b in seams or []:
        seam_arr[edge_index[(min(a, b), max(a, b))]] = True
    L = len(loop_vert)
    F = len(faces)
    obj = ObjectData(
        key=key, mesh_key=mesh_key or (key + "_mesh"),
        matrix_world=np.eye(4) if matrix is None else np.asarray(matrix, dtype=np.float64),
        co=co, edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2),
        face_loop_start=np.asarray(start, dtype=np.int64),
        face_loop_total=np.asarray(total, dtype=np.int64),
        loop_vert=np.asarray(loop_vert, dtype=np.int64),
        loop_edge=np.asarray(loop_edge, dtype=np.int64),
        face_material=np.asarray(face_material if face_material is not None else [0] * F, dtype=np.int64),
        face_select=np.zeros(F, dtype=bool), face_hide=np.zeros(F, dtype=bool),
        seams=seam_arr, edge_constraint=np.zeros(E, dtype=np.int64),
        materials=materials or [MaterialInfo("Mat")])
    for name, arr in (uv or {}).items():
        obj.uv_layer_names.append(name)
        obj.uv[name] = np.asarray(arr, dtype=np.float64)
        obj.uv_pin[name] = np.zeros(L, dtype=bool)
        obj.uv_select[name] = np.zeros(L, dtype=bool)
        if obj.active_uv is None:
            obj.active_uv = name
            obj.render_uv = name
            obj.input_uv_name = name
    obj._edge_index = edge_index
    return obj


def edge_id(obj: ObjectData, a: int, b: int) -> int:
    return obj._edge_index[(min(a, b), max(a, b))]


def grid(n: int = 4, size: float = 1.0):
    verts = [(x * size / n, y * size / n, 0.0) for y in range(n + 1) for x in range(n + 1)]
    faces = []
    for y in range(n):
        for x in range(n):
            a = y * (n + 1) + x
            faces.append([a, a + 1, a + n + 2, a + n + 1])
    return verts, faces


def grid_uv(verts, faces, scale: float = 0.8, offset=(0.1, 0.1)):
    loops = [v for f in faces for v in f]
    return np.asarray([(verts[v][0] * scale + offset[0], verts[v][1] * scale + offset[1]) for v in loops])


def cube(size: float = 1.0):
    s = size / 2
    verts = [(-s, -s, -s), (s, -s, -s), (s, s, -s), (-s, s, -s),
             (-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s)]
    faces = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    return verts, faces


def cylinder(segments: int = 12, radius: float = 0.5, height: float = 2.0, caps: bool = True):
    verts = []
    for z in (0.0, height):
        for i in range(segments):
            t = 2 * math.pi * i / segments
            verts.append((radius * math.cos(t), radius * math.sin(t), z))
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([i, j, segments + j, segments + i])
    if caps:
        faces.append(list(range(segments))[::-1])
        faces.append(list(range(segments, 2 * segments)))
    return verts, faces


def uv_sphere(seg: int = 12, rings: int = 6, r: float = 1.0):
    verts = [(0, 0, r)]
    for i in range(1, rings):
        phi = math.pi * i / rings
        for j in range(seg):
            th = 2 * math.pi * j / seg
            verts.append((r * math.sin(phi) * math.cos(th), r * math.sin(phi) * math.sin(th), r * math.cos(phi)))
    verts.append((0, 0, -r))
    faces = []
    for j in range(seg):
        faces.append([0, 1 + j, 1 + (j + 1) % seg])
    for i in range(rings - 2):
        for j in range(seg):
            a = 1 + i * seg + j
            b = 1 + i * seg + (j + 1) % seg
            faces.append([a, a + seg, b + seg, b])
    last = len(verts) - 1
    base = 1 + (rings - 2) * seg
    for j in range(seg):
        faces.append([base + (j + 1) % seg, base + j, last])
    return verts, faces


def translate(verts, d):
    return [(v[0] + d[0], v[1] + d[1], v[2] + d[2]) for v in verts]


def scale_matrix(sx, sy, sz, t=(0, 0, 0)):
    m = np.eye(4)
    m[0, 0], m[1, 1], m[2, 2] = sx, sy, sz
    m[:3, 3] = t
    return m


def snapshot_of(*objs: ObjectData) -> InputSnapshot:
    return InputSnapshot(objects={o.key: o for o in objs}, order=[o.key for o in objs])
