"""確定前バックアップ。全UV座標・属性、レイヤー名と順序、アクティブ／レンダー設定、
シーム、参照 (Object→Mesh) を保持する。"""
from __future__ import annotations

import hashlib
from typing import Dict

import numpy as np

from ..blender import uv_adapter
from .serialization import pack_array, unpack_array


def topology_fingerprint(mesh) -> str:
    h = hashlib.sha256()
    n = len(mesh.loops)
    lv = np.zeros(n, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", lv)
    h.update(lv.tobytes())
    f = len(mesh.polygons)
    ls = np.zeros(f, dtype=np.int32)
    mesh.polygons.foreach_get("loop_start", ls)
    h.update(ls.tobytes())
    h.update(f"{len(mesh.vertices)}/{len(mesh.edges)}".encode())
    return h.hexdigest()


def seams_array(mesh) -> np.ndarray:
    s = np.zeros(len(mesh.edges), dtype=bool)
    mesh.edges.foreach_get("use_seam", s)
    return s


def backup_mesh(obj) -> Dict:
    me = obj.data
    layers = []
    for layer in me.uv_layers:
        uv, pin, sel = uv_adapter.read_layer(me, layer.name)
        layers.append({"name": layer.name, "uv": pack_array(uv), "pin": pack_array(pin),
                       "select": pack_array(sel),
                       "fingerprint": uv_adapter.layer_fingerprint(me, layer.name)})
    active, render = uv_adapter.active_render_names(me)
    return {"object": obj.name, "mesh": me.name, "layers": layers, "active": active,
            "render": render, "seams": pack_array(seams_array(me)),
            "topology": topology_fingerprint(me)}


def verify_backup(obj, backup: Dict) -> bool:
    """バックアップから元状態を再構成できるか (配列が読める・形状が一致)。"""
    me = obj.data
    for l in backup["layers"]:
        uv = unpack_array(l["uv"])
        if uv.shape != (len(me.loops), 2):
            return False
    return unpack_array(backup["seams"]).shape == (len(me.edges),)
