"""Blender Mesh → 純データ (ObjectData / InputSnapshot)。

解析層に長寿命の Blender 参照を渡さない。Edit Mode の Object は
update_from_editmode() で編集データを Mesh に反映してから読む
(データ自体は変更しない)。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import bpy
import numpy as np

from ..core.models import EdgeConstraint, InputSnapshot, MaterialInfo, ObjectData
from ..core.settings import Settings
from . import compatibility as compat

CONSTRAINT_ATTR = "suo_edge_constraint"


def _foreach(coll, attr: str, n: int, dtype, width: int = 1) -> np.ndarray:
    arr = np.zeros(n * width, dtype=dtype)
    if n:
        coll.foreach_get(attr, arr)
    return arr.reshape(n, width) if width > 1 else arr


def _base_color(mat) -> Optional[Tuple[float, float, float]]:
    """静的に取得できる Base Color のみ。リンクされた入力や複雑なノードは不明 (None)。"""
    if mat is None:
        return None
    if not mat.use_nodes or mat.node_tree is None:
        return tuple(mat.diffuse_color[:3])
    out = next((n for n in mat.node_tree.nodes
                if n.type == "OUTPUT_MATERIAL" and getattr(n, "is_active_output", True)), None)
    if out is None or not out.inputs["Surface"].is_linked:
        return None
    node = out.inputs["Surface"].links[0].from_node
    if node.type != "BSDF_PRINCIPLED":
        return None
    inp = node.inputs.get("Base Color")
    if inp is None or inp.is_linked:
        return None
    return tuple(inp.default_value[:3])


def mesh_users(mesh) -> Tuple[str, ...]:
    return tuple(sorted(o.name for o in bpy.data.objects if o.data == mesh))


def read_object(obj, input_uv_name: Optional[str]) -> ObjectData:
    if obj.mode == "EDIT":
        obj.update_from_editmode()
    me = obj.data
    V, E, F, L = len(me.vertices), len(me.edges), len(me.polygons), len(me.loops)
    co = _foreach(me.vertices, "co", V, np.float32, 3).astype(np.float64)
    edges = _foreach(me.edges, "vertices", E, np.int32, 2).astype(np.int64)
    start = _foreach(me.polygons, "loop_start", F, np.int32).astype(np.int64)
    total = _foreach(me.polygons, "loop_total", F, np.int32).astype(np.int64)
    lv = _foreach(me.loops, "vertex_index", L, np.int32).astype(np.int64)
    le = _foreach(me.loops, "edge_index", L, np.int32).astype(np.int64)
    mat = _foreach(me.polygons, "material_index", F, np.int32).astype(np.int64)
    fsel = _foreach(me.polygons, "select", F, bool)
    fhide = _foreach(me.polygons, "hide", F, bool)
    seams = _foreach(me.edges, "use_seam", E, bool)
    cons = np.zeros(E, dtype=np.int64)
    attr = me.attributes.get(CONSTRAINT_ATTR)
    if attr is not None and attr.domain == "EDGE" and attr.data_type == "INT":
        tmp = np.zeros(E, dtype=np.int32)
        attr.data.foreach_get("value", tmp)
        cons = np.clip(tmp.astype(np.int64), 0, int(EdgeConstraint.FORBID))
    me.calc_loop_triangles()
    tri_loops = _foreach(me.loop_triangles, "loops", len(me.loop_triangles), np.int32, 3).astype(np.int64)
    materials = [MaterialInfo(s.material.name if s.material else f"<slot {i}>", _base_color(s.material))
                 for i, s in enumerate(obj.material_slots)]
    od = ObjectData(
        key=obj.name, mesh_key=me.name, matrix_world=np.array(obj.matrix_world, dtype=np.float64),
        co=co, edges=edges, face_loop_start=start, face_loop_total=total, loop_vert=lv,
        loop_edge=le, face_material=mat, face_select=fsel, face_hide=fhide, seams=seams,
        edge_constraint=cons, materials=materials, tri_loops=tri_loops,
        editable=compat.is_editable(obj) and compat.is_editable(me))
    for layer in me.uv_layers:
        uv = _foreach(layer.data, "uv", L, np.float32, 2).astype(np.float64)
        od.uv_layer_names.append(layer.name)
        od.uv[layer.name] = uv
        od.uv_pin[layer.name] = compat.read_uv_bool(me, layer, "pin")
        od.uv_select[layer.name] = compat.read_uv_bool(me, layer, "vertex_selection")
        if layer.active_render:
            od.render_uv = layer.name
    od.active_uv = me.uv_layers.active.name if me.uv_layers.active else None
    if input_uv_name and input_uv_name in od.uv:
        od.input_uv_name = input_uv_name
    else:
        od.input_uv_name = od.active_uv
    users = mesh_users(me)
    od.mesh_users = len(users)
    od.mesh_user_keys = users
    return od


def target_objects(context, extra_names=()) -> Tuple[List, List[Tuple[str, str]]]:
    objs, excluded = [], []
    seen = set()
    cands = list(context.selected_objects)
    for name in extra_names:
        ob = bpy.data.objects.get(name)
        if ob is not None and ob not in cands:
            cands.append(ob)
    for ob in sorted(cands, key=lambda o: o.name):
        if ob.name in seen:
            continue
        seen.add(ob.name)
        if ob.type != "MESH":
            excluded.append((ob.name, f"Mesh ではありません ({ob.type})"))
            continue
        objs.append(ob)
    return objs, excluded


def capture_input(context, settings: Settings, object_names: Optional[List[str]] = None) -> InputSnapshot:
    """選択 Mesh Object (または指定名) のスナップショット。"""
    if object_names is not None:
        obs = [bpy.data.objects[n] for n in object_names if n in bpy.data.objects]
        excluded = [(n, "存在しません") for n in object_names if n not in bpy.data.objects]
    else:
        obs, excluded = target_objects(context)
    objects: Dict[str, ObjectData] = {}
    for ob in obs:
        objects[ob.name] = read_object(ob, settings.input_uv_for(ob.name))
    from .context import ContextState
    return InputSnapshot(objects=objects, order=[o.name for o in obs], excluded=excluded,
                         context_state=ContextState(context).as_dict())
