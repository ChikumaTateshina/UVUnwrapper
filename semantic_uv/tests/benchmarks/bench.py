"""ベンチマーク (実装指示書 10章)。

    blender --background --factory-startup --python semantic_uv/tests/benchmarks/bench.py -- out.json [levels]

CPU・OS・Blender ビルド・入力ハッシュ・設定・候補予算・段階別時間を記録する。
製品閾値はこの実測をもとに別途確定する (未測定の性能は保証しない)。
"""
import json
import os
import platform
import sys
import time

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

import semantic_uv  # noqa: E402
from semantic_uv.blender.mesh_adapter import capture_input  # noqa: E402
from semantic_uv.core import pipeline as P  # noqa: E402
from semantic_uv.ui import properties  # noqa: E402


def make_case(level, n_objects):
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    obs = []
    for i in range(n_objects):
        bpy.ops.mesh.primitive_monkey_add(location=(i * 3, 0, 0))
        ob = bpy.context.active_object
        ob.name = f"M{i:03d}"
        if level:
            mod = ob.modifiers.new("s", "SUBSURF")
            mod.levels = level
            bpy.ops.object.modifier_apply(modifier=mod.name)
        obs.append(ob)
    bpy.ops.object.select_all(action="DESELECT")
    for ob in obs:
        ob.select_set(True)
    bpy.context.view_layer.objects.active = obs[0]
    return obs


def run_case(level, n_objects, backend):
    obs = make_case(level, n_objects)
    tris = sum(len(o.data.polygons) for o in obs) * 2
    bpy.context.scene.suo_settings.backend = backend
    s = properties.to_settings(bpy.context)
    t = {}
    t0 = time.perf_counter()
    snap = capture_input(bpy.context, s)
    t["capture"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    rep = P.validate_input(snap, s)
    t["validate_input"] = time.perf_counter() - t0
    scope = P.build_scope(snap, s)
    dom = P.build_domain(snap, scope, s)
    stages = []
    t0 = time.perf_counter()
    it = P.compute_plan_iter(snap, scope, dom, s)
    last = t0
    while True:
        try:
            frac, msg = next(it)
            now = time.perf_counter()
            stages.append((msg, now - last))
            last = now
        except StopIteration as stop:
            plan = stop.value
            break
    t["compute"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    vr = P.validate_plan(snap, plan, s)
    t["validate_plan"] = time.perf_counter() - t0
    q = plan.quality
    return {"level": level, "objects": n_objects, "approx_tris": tris, "backend": backend,
            "input_hash": snap.fingerprint()[:16], "valid": vr.ok, "times_s": t,
            "stages": stages[:12], "islands": q.island_count,
            "mean_angle": q.mean_angle_distortion, "mean_area": q.mean_area_distortion,
            "fill": q.fill_ratio, "budget": s.to_dict()["search"]}


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out_path = argv[0] if argv else None
    cases = [(0, 1), (2, 1), (0, 10)]
    if len(argv) > 1:
        cases = [tuple(int(x) for x in c.split("x")) for c in argv[1].split(",")]
    semantic_uv.register()
    results = []
    for level, n in cases:
        for backend in ("BLENDER_ANGLE_BASED",):
            r = run_case(level, n, backend)
            print(f"level={level} objects={n} tris~{r['approx_tris']} valid={r['valid']} "
                  f"compute={r['times_s']['compute']:.2f}s validate={r['times_s']['validate_plan']:.2f}s "
                  f"islands={r['islands']} fill={r['fill']:.3f}")
            results.append(r)
    env = {"blender": bpy.app.version_string, "build_hash": bpy.app.build_hash.decode() if isinstance(bpy.app.build_hash, bytes) else str(bpy.app.build_hash),
           "os": platform.platform(), "cpu": platform.processor(), "python": platform.python_version()}
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({"env": env, "results": results}, fh, ensure_ascii=False, indent=1)
    semantic_uv.unregister()


main()
