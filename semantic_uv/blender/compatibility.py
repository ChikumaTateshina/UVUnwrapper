"""Blender API 差異の吸収と、標準展開 (Angle Based / Conformal) の作業コピー実行。

初期対象は Blender 4.5 LTS。4.2〜4.5 と 5.x の差異はここに集約する。
"""
from __future__ import annotations

from typing import List, Optional

import bpy
import numpy as np

from ..unwrap import backend as be

WORK_NAME = ".SUO_work"


def version() -> tuple:
    return tuple(bpy.app.version)


# ------------------------------------------------------- UV 属性
def uv_bool_layer(mesh, uv_layer, kind: str):
    """kind: 'pin' / 'vertex_selection'。存在しない場合 None。"""
    coll = getattr(uv_layer, kind, None)
    if coll is not None:
        return coll
    prefix = {"pin": ".pn.", "vertex_selection": ".vs."}[kind]
    attr = mesh.attributes.get(prefix + uv_layer.name)
    return attr.data if attr is not None else None


def read_uv_bool(mesh, uv_layer, kind: str) -> np.ndarray:
    n = len(mesh.loops)
    out = np.zeros(n, dtype=bool)
    coll = uv_bool_layer(mesh, uv_layer, kind)
    if coll is not None and len(coll) == n:
        tmp = np.zeros(n, dtype=bool)
        coll.foreach_get("value", tmp)
        out[:] = tmp
    return out


def write_uv_bool(mesh, uv_layer, kind: str, values: np.ndarray) -> None:
    if not values.any():
        coll = uv_bool_layer(mesh, uv_layer, kind)
        if coll is None or len(coll) != len(values):
            return  # 未作成の属性は False と等価
    coll = uv_bool_layer(mesh, uv_layer, kind)
    if coll is None or len(coll) != len(values):
        prefix = {"pin": ".pn.", "vertex_selection": ".vs."}[kind]
        attr = mesh.attributes.new(prefix + uv_layer.name, "BOOLEAN", "CORNER")
        coll = attr.data
    coll.foreach_set("value", values.astype(bool))


def max_uv_layers() -> int:
    return 8


def is_editable(id_block) -> bool:
    if id_block is None:
        return False
    if getattr(id_block, "library", None) is not None:
        ov = getattr(id_block, "override_library", None)
        return ov is not None and not getattr(ov, "is_system_override", False)
    ed = getattr(id_block, "is_editable", None)
    return True if ed is None else bool(ed)


# ------------------------------------------------- 標準展開バックエンド
def _blender_unwrap(jobs: List[be.UnwrapJob], method: str) -> Optional[List[Optional[np.ndarray]]]:
    """全 Job を1つの作業用 Mesh に分離島として作り、bpy.ops.uv.unwrap を1回実行する。
    作業用データ以外には作用しない。失敗時は None (内蔵ソルバーへ)。"""
    context = bpy.context
    if not jobs:
        return []
    verts: List = []
    faces: List = []
    offsets = []
    for job in jobs:
        off = len(verts)
        offsets.append(off)
        verts.extend(job.positions.tolist())
        faces.extend([[off + i for i in poly] for poly in job.polys])
    mesh = bpy.data.meshes.new(WORK_NAME)
    obj = None
    coll = None
    vl = context.view_layer
    prev_active = vl.objects.active
    prev_sel = [o for o in context.selected_objects]
    prev_mode = context.mode
    try:
        mesh.from_pydata(verts, [], faces)
        mesh.uv_layers.new(name="UV")
        obj = bpy.data.objects.new(WORK_NAME, mesh)
        coll = bpy.data.collections.new(WORK_NAME)
        context.scene.collection.children.link(coll)
        coll.objects.link(obj)
        if prev_mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        for o in prev_sel:
            o.select_set(False)
        obj.select_set(True)
        vl.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        kwargs = {"method": method, "margin": 0.0}
        res = bpy.ops.uv.unwrap(**kwargs)
        bpy.ops.object.mode_set(mode="OBJECT")
        if "FINISHED" not in res:
            return None
        n = len(mesh.loops)
        uv = np.zeros(n * 2, dtype=np.float32)
        mesh.uv_layers[0].data.foreach_get("uv", uv)
        uv = uv.reshape(n, 2).astype(np.float64)
        lv = np.zeros(n, dtype=np.int64)
        mesh.loops.foreach_get("vertex_index", lv)
        out: List[Optional[np.ndarray]] = []
        for ji, job in enumerate(jobs):
            r = np.full((len(job.positions), 2), np.nan)
            out.append(r)
        bounds = offsets + [len(verts)]
        for l in range(n):
            v = int(lv[l])
            ji = int(np.searchsorted(bounds, v, side="right") - 1)
            out[ji][v - offsets[ji]] = uv[l]
        return [r if np.all(np.isfinite(r)) else None for r in out]
    except Exception:
        return None
    finally:
        try:
            if context.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            pass
        if obj is not None:
            bpy.data.objects.remove(obj)
        if coll is not None:
            bpy.data.collections.remove(coll)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
        for o in prev_sel:
            try:
                o.select_set(True)
            except (RuntimeError, ReferenceError):
                pass
        vl.objects.active = prev_active
        if prev_mode == "EDIT_MESH" and prev_active is not None:
            try:
                bpy.ops.object.mode_set(mode="EDIT")
            except Exception:
                pass


def register():
    be.register_external_backend(_blender_unwrap)


def unregister():
    be.register_external_backend(None)
