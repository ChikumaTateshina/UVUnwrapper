"""モード・アクティブObject・Object選択の保存と復元。

面選択・UV選択はメッシュデータ側にあり、本アドオンは変更しない
(update_from_editmode とモード切替はデータを変えない)。
"""
from __future__ import annotations

from contextlib import contextmanager

import bpy


class ContextState:
    def __init__(self, context):
        vl = context.view_layer
        self.mode = context.mode
        self.active = vl.objects.active.name if vl.objects.active else None
        self.selected = [o.name for o in context.selected_objects]
        self.in_edit = [o.name for o in getattr(context, "objects_in_mode", []) or []] \
            if context.mode == "EDIT_MESH" else []
        self.edit = context.mode == "EDIT_MESH"

    def as_dict(self) -> dict:
        return {"mode": self.mode, "active": self.active, "selected": list(self.selected),
                "in_edit": list(self.in_edit)}

    def restore(self, context) -> None:
        vl = context.view_layer
        if context.mode != "OBJECT":
            _mode_set("OBJECT")
        for o in vl.objects:
            try:
                o.select_set(o.name in self.selected or o.name in self.in_edit)
            except RuntimeError:
                pass
        if self.active and self.active in bpy.data.objects:
            vl.objects.active = bpy.data.objects[self.active]
        if self.edit and vl.objects.active is not None:
            for name in self.in_edit:
                ob = bpy.data.objects.get(name)
                if ob is not None:
                    ob.select_set(True)
            _mode_set("EDIT")
            for o in vl.objects:
                try:
                    o.select_set(o.name in self.selected)
                except RuntimeError:
                    pass


def _mode_set(mode: str) -> None:
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode=mode)


@contextmanager
def object_mode(context):
    """Blender データへの書込はオブジェクトモードで行う (Edit Mode では
    終了時に編集データで上書きされるため)。処理後に元のモードへ戻す。"""
    state = ContextState(context)
    try:
        if context.mode != "OBJECT":
            for name in state.in_edit:
                ob = bpy.data.objects.get(name)
                if ob is not None:
                    ob.update_from_editmode()
            _mode_set("OBJECT")
        yield state
    finally:
        state.restore(context)
