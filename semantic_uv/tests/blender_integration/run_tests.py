"""Blender 実機統合テスト。

    blender --background --factory-startup --python semantic_uv/tests/blender_integration/run_tests.py -- [out.json]

配列比較で合否を判定し、スクリーンショットには依存しない。
"""
import json
import os
import sys
import tempfile
import traceback

import bpy
import bmesh
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import semantic_uv  # noqa: E402
from semantic_uv.blender import uv_adapter  # noqa: E402
from semantic_uv.blender.mesh_adapter import capture_input  # noqa: E402
from semantic_uv.core import pipeline as P  # noqa: E402
from semantic_uv.core.errors import ErrorCode, SUOError  # noqa: E402
from semantic_uv.core.models import TxState  # noqa: E402
from semantic_uv.persistence import restore as R  # noqa: E402
from semantic_uv.persistence import serialization as ser  # noqa: E402
from semantic_uv.persistence import transaction as T  # noqa: E402
from semantic_uv.ui.operators import Session  # noqa: E402
from semantic_uv.validation.overlaps import find_overlaps  # noqa: E402

RESULTS = []
TMP = tempfile.mkdtemp(prefix="suo_test_")


# ------------------------------------------------------------- helpers
def reset():
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    for m in list(bpy.data.meshes):
        bpy.data.meshes.remove(m)
    for t in list(bpy.data.texts):
        bpy.data.texts.remove(t)
    for m in list(bpy.data.materials):
        bpy.data.materials.remove(m)
    Session.clear()
    T.FAULT_HOOK = None
    p = bpy.context.scene.suo_settings
    for prop in p.bl_rna.properties:
        if prop.identifier != "rna_type" and not prop.is_readonly:
            p.property_unset(prop.identifier)


def add(kind, name, loc=(0, 0, 0), uv=True, **kw):
    getattr(bpy.ops.mesh, f"primitive_{kind}_add")(location=loc, **kw)
    ob = bpy.context.active_object
    ob.name = name
    ob.data.name = name + "_mesh"
    if not uv:
        while ob.data.uv_layers:
            ob.data.uv_layers.remove(ob.data.uv_layers[0])
    return ob


def select(*obs):
    bpy.ops.object.select_all(action="DESELECT")
    for o in obs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = obs[0]


def uv_state(ob):
    me = ob.data
    out = {}
    for l in me.uv_layers:
        uv, pin, sel = uv_adapter.read_layer(me, l.name)
        out[l.name] = (uv.copy(), pin.copy(), sel.copy())
    act, ren = uv_adapter.active_render_names(me)
    s = np.zeros(len(me.edges), dtype=bool)
    me.edges.foreach_get("use_seam", s)
    return {"layers": out, "order": [l.name for l in me.uv_layers], "active": act, "render": ren,
            "seams": s, "mesh": me.name}


def same_state(a, b):
    if a["order"] != b["order"] or a["active"] != b["active"] or a["render"] != b["render"]:
        return False
    if a["mesh"] != b["mesh"] or not np.array_equal(a["seams"], b["seams"]):
        return False
    for k in a["layers"]:
        for x, y in zip(a["layers"][k], b["layers"][k]):
            if not np.array_equal(x, y):
                return False
    return True


def op(name):
    """スクリプトからの呼出しでは ERROR レポートが RuntimeError になるため CANCELLED として扱う。"""
    try:
        return getattr(bpy.ops.suo, name)()
    except RuntimeError:
        return {"CANCELLED"}


def generate():
    r = op("generate")
    if "FINISHED" not in r:
        raise AssertionError("generate failed: " + " | ".join(Session.messages[:3]))
    return Session.plan


def apply():
    r = op("apply")
    if "FINISHED" not in r:
        raise AssertionError("apply failed: " + " | ".join(Session.messages[:3]))


def result_layer(ob):
    names = [l.name for l in ob.data.uv_layers if l.name.startswith("SUO_Result_")]
    return names[-1] if names else None


def all_result_tris(obs):
    tris = []
    for ob in obs:
        me = ob.data
        name = result_layer(ob)
        uv, _, _ = uv_adapter.read_layer(me, name)
        me.calc_loop_triangles()
        idx = np.zeros(len(me.loop_triangles) * 3, dtype=np.int32)
        me.loop_triangles.foreach_get("loops", idx)
        tris.append(uv[idx.reshape(-1, 3)].astype(np.float64))
    return np.concatenate(tris)


def test(fn):
    RESULTS.append(fn)
    return fn


# ---------------------------------------------------------------- tests
@test
def T101_T102_multi_object_generate_apply():
    a = add("cube", "A")
    b = add("cylinder", "B", (3, 0, 0), vertices=16)
    c = add("monkey", "C", (0, 4, 0))
    select(a, b, c)
    before = {o.name: uv_state(o) for o in (a, b, c)}
    plan = generate()
    assert set(plan.write_targets()) == {"A", "B", "C"}
    # プレビューでは元データは不変
    for o in (a, b, c):
        assert same_state(before[o.name], uv_state(o)), "プレビューで元データが変化"
    apply()
    for o in (a, b, c):
        st = uv_state(o)
        name = result_layer(o)
        assert name == "SUO_Result_001", name
        for k in before[o.name]["layers"]:
            for x, y in zip(before[o.name]["layers"][k], st["layers"][k]):
                assert np.array_equal(x, y), "既存UVが変化"
        assert st["active"] == before[o.name]["active"] and st["render"] == before[o.name]["render"]
        assert len(o.data.vertices) and o.name in bpy.data.objects   # 結合なし
    assert not find_overlaps(all_result_tris((a, b, c)), max_pairs=1), "Object間重複"


@test
def BACKEND_blender_angle_based_used():
    a = add("uv_sphere", "S", segments=16, ring_count=8)
    select(a)
    plan = generate()
    assert plan.quality.mean_angle_distortion < 0.3
    assert bpy.data.objects.get(".SUO_work") is None and bpy.data.meshes.get(".SUO_work") is None, "作業資源が残存"


@test
def T104_density_with_unapplied_scale():
    a = add("cube", "A")
    b = add("cube", "B", (4, 0, 0))
    b.scale = (2.0, 1.0, 0.5)
    select(a, b)
    plan = generate()
    assert plan.quality.density_deviation_max <= 0.05, plan.quality.density_deviation_max
    assert tuple(b.scale) == (2.0, 1.0, 0.5)   # 未適用変換を変更しない


@test
def T110_udim():
    obs = [add("monkey", f"M{i}", (i * 3, 0, 0)) for i in range(3)]
    select(*obs)
    p = bpy.context.scene.suo_settings
    p.tile_mode = "UDIM"
    p.allowed_tiles = "1001,1002,1011"
    generate()
    apply()
    from semantic_uv.layout.domain import island_tile
    for ob in obs:
        uv, _, _ = uv_adapter.read_layer(ob.data, result_layer(ob))
        lo = np.floor(uv.min(axis=0))
        assert lo[0] >= 0 and lo[1] >= 0


@test
def T111_fault_on_second_object_restores_all():
    a = add("cube", "A")
    b = add("cube", "B", (3, 0, 0))
    select(a, b)
    before = {o.name: uv_state(o) for o in (a, b)}
    generate()

    def hook(stage, key):
        if key == "B" and stage == "before_write":
            raise RuntimeError("injected")
    T.FAULT_HOOK = hook
    r = op("apply")
    T.FAULT_HOOK = None
    assert "CANCELLED" in r
    for o in (a, b):
        assert same_state(before[o.name], uv_state(o)), f"{o.name} が復元されていない"
        assert result_layer(o) is None
    assert not ser.list_ids(), "未確定の記録が残存"


@test
def T112_deterministic_blender_backend():
    a = add("monkey", "M")
    select(a)
    p1 = generate()
    u1 = p1.objects["M"].uv.copy()
    Session.clear()
    p2 = generate()
    assert np.array_equal(u1, p2.objects["M"].uv)


@test
def T201_T216_partial_edit_mode_apply_restore():
    a = add("grid", "G", x_subdivisions=8, y_subdivisions=8, size=2)
    b = add("cube", "B", (4, 0, 0))
    select(a, b)
    bpy.ops.object.mode_set(mode="EDIT")
    for ob in (a, b):
        bm = bmesh.from_edit_mesh(ob.data)
        for f in bm.faces:
            f.select_set(False)
        if ob is a:
            for f in bm.faces:
                c = f.calc_center_median()
                if abs(c.x) < 0.4 and abs(c.y) < 0.4:
                    f.select_set(True)
        bmesh.update_edit_mesh(ob.data)
    p = bpy.context.scene.suo_settings
    p.scope = "SELECTED_FACES"
    p.boundary_mode = "RELOCATE_SELECTED"
    bpy.ops.object.mode_set(mode="OBJECT")
    # 既存UVを小さくして空きを作る (入力)
    for ob in (a, b):
        uv, pin, sel = uv_adapter.read_layer(ob.data, ob.data.uv_layers[0].name)
        uv = uv * 0.4 + (0.05 if ob is a else 0.55)
        ob.data.uv_layers[0].data.foreach_set("uv", uv.reshape(-1))
    before = {o.name: uv_state(o) for o in (a, b)}
    bpy.ops.object.mode_set(mode="EDIT")
    plan = generate()
    assert bpy.context.mode == "EDIT_MESH", "モードが復元されていない"
    assert plan.write_targets() == ["G"]
    apply()
    assert bpy.context.mode == "EDIT_MESH"
    bpy.ops.object.mode_set(mode="OBJECT")
    src, _, _ = before["G"]["layers"][before["G"]["order"][0]]
    out, _, _ = uv_adapter.read_layer(a.data, result_layer(a))
    fsel = np.zeros(len(a.data.polygons), dtype=bool)
    a.data.polygons.foreach_get("select", fsel)
    lsel = np.repeat(fsel, [p.loop_total for p in a.data.polygons])
    assert np.array_equal(out[~lsel], src[~lsel]), "未選択コーナーが変化"
    assert result_layer(b) is None, "選択ゼロObjectに書込"
    tris = all_result_tris((a,))
    assert not find_overlaps(tris, max_pairs=1)
    # Restore
    r = op("restore")
    assert "FINISHED" in r, Session.messages
    for o in (a, b):
        assert same_state(before[o.name], uv_state(o))


@test
def T216_cancel_partial_compute_keeps_input():
    a = add("grid", "G", x_subdivisions=6, y_subdivisions=6)
    select(a)
    bpy.context.scene.suo_settings.scope = "SELECTED_FACES"
    for i, poly in enumerate(a.data.polygons):
        poly.select = i in (14, 15)
    before = uv_state(a)
    s = __import__("semantic_uv.ui.properties", fromlist=["x"]).to_settings(bpy.context)
    snap = capture_input(bpy.context, s)
    scope = P.build_scope(snap, s)
    dom = P.build_domain(snap, scope, s)
    it = P.compute_plan_iter(snap, scope, dom, s)
    next(it)
    it.close()   # 取消
    assert same_state(before, uv_state(a))


@test
def T301_shared_mesh():
    a = add("cube", "A")
    b = bpy.data.objects.new("B", a.data)
    bpy.context.scene.collection.objects.link(b)
    b.location = (3, 0, 0)
    select(a)
    r = op("generate")
    assert "CANCELLED" in r and any("SHARED_MESH" in m for m in Session.messages), Session.messages
    bpy.context.scene.suo_settings.allow_single_user = True
    before_b = uv_state(b)
    generate()
    apply()
    assert a.data != b.data, "独立化されていない"
    assert same_state(before_b, uv_state(b)), "非対象インスタンスが変化"
    r = op("restore")
    assert "FINISHED" in r, Session.messages
    assert a.data == b.data, "Restore で共有参照へ戻っていない"


@test
def T302_stale_input_rejected():
    a = add("cube", "A")
    select(a)
    generate()
    a.data.vertices[0].co.x += 0.1
    r = op("apply")
    assert "CANCELLED" in r and any("STALE_INPUT" in m for m in Session.messages), Session.messages
    assert result_layer(a) is None


@test
def T303_restore_conflict_on_manual_edit():
    a = add("cube", "A")
    select(a)
    generate()
    apply()
    me = a.data
    me.uv_layers[result_layer(a)].data[0].uv = (0.5, 0.5)
    r = op("restore")
    assert "CANCELLED" in r and any("RESTORE_CONFLICT" in m for m in Session.messages), Session.messages
    assert result_layer(a) is not None, "手動編集を上書きした"
    assert tuple(me.uv_layers[result_layer(a)].data[0].uv) == (0.5, 0.5)
    r = op("recover_backup")
    assert "FINISHED" in r
    assert bpy.data.objects.get("A_SUO_recovered") is not None


@test
def T304_restore_after_save_reload():
    a = add("cube", "A")
    b = add("cylinder", "B", (3, 0, 0))
    select(a, b)
    before = {o.name: uv_state(o) for o in (a, b)}
    generate()
    apply()
    path = os.path.join(TMP, "t304.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.open_mainfile(filepath=path)
    Session.clear()
    r = op("restore")
    assert "FINISHED" in r, Session.messages
    for name in ("A", "B"):
        assert same_state(before[name], uv_state(bpy.data.objects[name])), name


@test
def T305_fault_at_every_stage():
    for stage in ("before_single_user", "before_layer", "before_write", "after_write", "verify"):
        reset()
        a = add("cube", "A")
        b = add("cube", "B", (3, 0, 0))
        select(a, b)
        before = {o.name: uv_state(o) for o in (a, b)}
        generate()

        def hook(st, key, stage=stage):
            if st == stage and key == "B":
                raise MemoryError("injected " + stage)
        T.FAULT_HOOK = hook
        r = op("apply")
        T.FAULT_HOOK = None
        assert "CANCELLED" in r, stage
        for o in (a, b):
            assert same_state(before[o.name], uv_state(o)), f"{stage}: {o.name}"
    # ロールバック自体の失敗は正直に報告する
    reset()
    a = add("cube", "A")
    select(a)
    generate()
    orig = uv_adapter.remove_layer
    T.FAULT_HOOK = lambda st, key: (_ for _ in ()).throw(RuntimeError("x")) if st == "after_write" else None
    uv_adapter.remove_layer = lambda *a_, **k_: (_ for _ in ()).throw(RuntimeError("rollback broken"))
    try:
        r = op("apply")
    finally:
        uv_adapter.remove_layer = orig
        T.FAULT_HOOK = None
    assert "CANCELLED" in r
    assert any("ROLLBACK_FAILED" in m for m in Session.messages), Session.messages


@test
def T306_invalid_inputs_stop():
    a = add("cube", "Z")
    a.scale = (1, 0, 1)
    select(a)
    r = op("generate")
    assert "CANCELLED" in r and any("ゼロスケール" in m for m in Session.messages), Session.messages
    reset()
    me = bpy.data.meshes.new("NM")
    me.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)], [],
                   [(0, 1, 2), (1, 0, 3), (0, 1, 4)])
    ob = bpy.data.objects.new("NM", me)
    bpy.context.scene.collection.objects.link(ob)
    select(ob)
    before = uv_state(ob)
    r = op("generate")
    assert "CANCELLED" in r and any("非多様体" in m for m in Session.messages), Session.messages
    assert same_state(before, uv_state(ob))


@test
def T307_no_network_or_ml_imports():
    import pkgutil
    bad = {"socket", "urllib", "requests", "http", "torch", "tensorflow", "sklearn", "onnxruntime"}
    pkg_dir = os.path.join(ROOT, "semantic_uv")
    found = []
    for dirpath, _, files in os.walk(pkg_dir):
        if "tests" in dirpath:
            continue
        for f in files:
            if f.endswith(".py"):
                text = open(os.path.join(dirpath, f), encoding="utf-8").read()
                for m in bad:
                    if f"import {m}" in text or f"from {m}" in text:
                        found.append((f, m))
    assert not found, found
    semantic_uv.unregister()
    semantic_uv.register()


@test
def SEAMS_apply_and_restore():
    a = add("cube", "A")
    select(a)
    before = uv_state(a)
    generate()
    r = op("apply_seams")
    assert "FINISHED" in r, Session.messages
    s = np.zeros(len(a.data.edges), dtype=bool)
    a.data.edges.foreach_get("use_seam", s)
    assert s.any()
    r = op("restore")
    assert "FINISHED" in r, Session.messages
    assert same_state(before, uv_state(a))


# ---------------------------------------------------------------- main
def main():
    semantic_uv.register()
    out = []
    for fn in RESULTS:
        reset()
        try:
            fn()
            out.append({"test": fn.__name__, "ok": True})
            print(f"PASS {fn.__name__}")
        except Exception as ex:
            out.append({"test": fn.__name__, "ok": False, "error": f"{type(ex).__name__}: {ex}",
                        "trace": traceback.format_exc()})
            print(f"FAIL {fn.__name__}: {type(ex).__name__}: {ex}")
            traceback.print_exc()
    semantic_uv.unregister()
    ok = sum(r["ok"] for r in out)
    print(f"\n{ok}/{len(out)} passed on Blender {bpy.app.version_string}")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if argv:
        with open(argv[0], "w", encoding="utf-8") as fh:
            json.dump({"blender": bpy.app.version_string, "results": out}, fh, ensure_ascii=False, indent=1)
    sys.exit(0 if ok == len(out) else 1)


main()
