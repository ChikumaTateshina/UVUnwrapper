"""UV Editor 上のプレビュー描画。作業コピー (ResultPlan) のみを描き、元データに触れない。"""
from __future__ import annotations

import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from .operators import Session

_handle = None
_cache = {"plan": None, "edges": None}

COLOR_MOVED = (1.0, 0.55, 0.1, 0.9)
COLOR_FIXED = (0.6, 0.6, 0.6, 0.5)


def _edge_lines():
    plan = Session.plan
    if _cache["plan"] is plan:
        return _cache["edges"]
    moved, fixed = [], []
    snap = Session.snapshot
    for key in plan.order:
        obj = snap.objects[key]
        op = plan.objects[key]
        for f in range(obj.n_faces):
            loops = list(obj.face_loops(f))
            pts = op.uv[loops]
            target = moved if op.writable[loops[0]] else fixed
            for i in range(len(loops)):
                target.append(pts[i])
                target.append(pts[(i + 1) % len(loops)])
    edges = (np.asarray(moved).reshape(-1, 2) if moved else np.zeros((0, 2)),
             np.asarray(fixed).reshape(-1, 2) if fixed else np.zeros((0, 2)))
    _cache["plan"], _cache["edges"] = plan, edges
    return edges


def _draw():
    ctx = bpy.context
    if Session.plan is None or not ctx.scene.suo_settings.show_preview:
        return
    region = ctx.region
    v2d = region.view2d
    x0, y0 = v2d.view_to_region(0.0, 0.0, clip=False)
    x1, y1 = v2d.view_to_region(1.0, 1.0, clip=False)
    sx, sy = x1 - x0, y1 - y0
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")
    for pts, color in zip(_edge_lines(), (COLOR_MOVED, COLOR_FIXED)):
        if len(pts) == 0:
            continue
        coords = np.c_[pts[:, 0] * sx + x0, pts[:, 1] * sy + y0].astype(np.float32)
        batch = batch_for_shader(shader, "LINES", {"pos": coords})
        shader.uniform_float("color", color)
        batch.draw(shader)
    gpu.state.blend_set("NONE")


def register():
    global _handle
    if not bpy.app.background:
        _handle = bpy.types.SpaceImageEditor.draw_handler_add(_draw, (), "WINDOW", "POST_PIXEL")


def unregister():
    global _handle
    if _handle is not None:
        bpy.types.SpaceImageEditor.draw_handler_remove(_handle, "WINDOW")
        _handle = None
