"""UV レイヤーの作成・書込・読取。トランザクションからのみ呼ぶ。"""
from __future__ import annotations

import hashlib
from typing import Iterable, Optional

import numpy as np

from . import compatibility as compat


def unique_layer_name(meshes: Iterable, prefix: str) -> str:
    names = set()
    for me in meshes:
        names.update(l.name for l in me.uv_layers)
        names.update(a.name for a in me.attributes)
    i = 1
    while True:
        cand = f"{prefix}{i:03d}"
        if cand not in names:
            return cand
        i += 1


def read_layer(mesh, name: str):
    layer = mesh.uv_layers[name]
    n = len(mesh.loops)
    uv = np.zeros(n * 2, dtype=np.float32)
    layer.data.foreach_get("uv", uv)
    return (uv.reshape(n, 2), compat.read_uv_bool(mesh, layer, "pin"),
            compat.read_uv_bool(mesh, layer, "vertex_selection"))


def layer_fingerprint(mesh, name: str) -> str:
    uv, pin, sel = read_layer(mesh, name)
    h = hashlib.sha256()
    h.update(uv.tobytes())
    h.update(pin.tobytes())
    h.update(sel.tobytes())
    return h.hexdigest()


def create_layer(mesh, name: str):
    if len(mesh.uv_layers) >= compat.max_uv_layers():
        raise RuntimeError("UV レイヤー数が上限に達しています")
    active = mesh.uv_layers.active.name if mesh.uv_layers.active else None
    render = next((l.name for l in mesh.uv_layers if l.active_render), None)
    layer = mesh.uv_layers.new(name=name, do_init=False)
    if layer is None:
        raise RuntimeError("UV レイヤーを作成できません")
    # 既存のアクティブ／レンダー設定は切り替えない
    if active is not None and active in mesh.uv_layers:
        mesh.uv_layers.active = mesh.uv_layers[active]
    if render is not None and render in mesh.uv_layers:
        mesh.uv_layers[render].active_render = True
    return layer


def write_layer(mesh, name: str, uv: np.ndarray, pin: np.ndarray, select: np.ndarray) -> None:
    layer = mesh.uv_layers[name]
    if uv.shape != (len(mesh.loops), 2):
        raise RuntimeError("コーナー数が一致しません")
    layer.data.foreach_set("uv", uv.astype(np.float32).reshape(-1))
    compat.write_uv_bool(mesh, layer, "pin", pin)
    compat.write_uv_bool(mesh, layer, "vertex_selection", select)
    mesh.update()


def remove_layer(mesh, name: str) -> None:
    layer = mesh.uv_layers.get(name)
    if layer is not None:
        mesh.uv_layers.remove(layer)
    for prefix in (".pn.", ".vs.", ".es."):
        a = mesh.attributes.get(prefix + name)
        if a is not None:
            mesh.attributes.remove(a)


def active_render_names(mesh):
    active = mesh.uv_layers.active.name if mesh.uv_layers.active else None
    render = next((l.name for l in mesh.uv_layers if l.active_render), None)
    return active, render


def set_active_render(mesh, active: Optional[str], render: Optional[str]) -> None:
    if active is not None and active in mesh.uv_layers:
        mesh.uv_layers.active = mesh.uv_layers[active]
    if render is not None and render in mesh.uv_layers:
        mesh.uv_layers[render].active_render = True
