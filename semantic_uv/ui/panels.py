"""3D View / UV Editor のサイドバーパネル。"""
from __future__ import annotations

import bpy

from .operators import Session


def _draw(layout, context):
    p = context.scene.suo_settings
    col = layout.column(align=True)
    row = col.row(align=True)
    row.operator("suo.analyze")
    row.operator("suo.generate")
    row = col.row(align=True)
    row.prop(p, "show_preview", text="Preview", toggle=True)
    row.operator("suo.compare")
    row = col.row(align=True)
    row.operator("suo.apply")
    row.operator("suo.cancel")
    row.operator("suo.restore")
    layout.label(text=f"状態: {Session.state.value}")

    box = layout.box()
    box.label(text="対象と入力UV")
    meshes = [o for o in context.selected_objects if o.type == "MESH"]
    for ob in sorted(meshes, key=lambda o: o.name):
        r = box.row()
        r.label(text=ob.name, icon="MESH_DATA")
        r.prop_search(ob, "suo_input_uv", ob.data, "uv_layers", text="")
    others = [o.name for o in context.selected_objects if o.type != "MESH"]
    if others:
        box.label(text="対象外: " + ", ".join(others), icon="INFO")

    box = layout.box()
    box.label(text="処理")
    box.prop(p, "scope")
    box.prop(p, "uv_mode")
    if p.scope == "SELECTED_FACES":
        box.prop(p, "boundary_mode")
        if p.boundary_mode == "RELOCATE_SELECTED":
            box.label(text="未選択側固定、選択側は境界分離", icon="INFO")
    box.prop(p, "seam_mode")
    box.prop(p, "backend")
    box.prop(p, "preset")

    box = layout.box()
    box.label(text="配置")
    box.prop(p, "density_mode")
    if p.density_mode == "SPECIFIED":
        box.prop(p, "density_px")
    box.prop(p, "resolution")
    box.prop(p, "gap_px")
    box.prop(p, "tile_mode")
    if p.tile_mode == "UDIM":
        box.prop(p, "allowed_tiles")

    box = layout.box()
    box.label(text="探索予算")
    box.prop(p, "angle_limit")
    box.prop(p, "distortion_limit")
    box.prop(p, "max_candidates")
    box.prop(p, "max_iterations")
    box.prop(p, "placement_candidates")
    box.prop(p, "seed")
    box.prop(p, "time_limit")
    box.prop(p, "allow_single_user")

    box = layout.box()
    box.label(text="シームと制約")
    if context.mode == "EDIT_MESH":
        r = box.row(align=True)
        for c in ("REQUIRE", "FORBID", "PREFER", "AVOID", "AUTO"):
            r.operator("suo.set_edge_constraint", text=c).constraint = c
    box.operator("suo.apply_seams")
    box.operator("suo.recover_backup")

    if Session.messages:
        box = layout.box()
        for line in Session.messages:
            for chunk in _wrap(line, 60):
                box.label(text=chunk)


def _wrap(text, n):
    return [text[i:i + n] for i in range(0, len(text), n)] or [""]


class SUO_PT_view3d(bpy.types.Panel):
    bl_label = "Semantic UV Organizer"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Semantic UV"

    def draw(self, context):
        _draw(self.layout, context)


class SUO_PT_uv(bpy.types.Panel):
    bl_label = "Semantic UV Organizer"
    bl_space_type = "IMAGE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Semantic UV"

    def draw(self, context):
        _draw(self.layout, context)


classes = (SUO_PT_view3d, SUO_PT_uv)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
