"""部分再展開 (T-201〜T-215 の純データ版)。"""
import numpy as np
import pytest

from helpers import assert_unchanged, frozen_copy, run, settings
from fixtures.meshes import build_object, edge_id, grid, grid_uv, snapshot_of, translate
from semantic_uv.analysis.topology import Topology, uv_edge_continuous
from semantic_uv.core import pipeline as P
from semantic_uv.core.errors import ErrorCode, SUOError
from semantic_uv.core.models import EdgeConstraint
from semantic_uv.core.settings import BoundaryMode, Scope, TileMode, UVMode
from semantic_uv.validation.overlaps import find_overlaps

KEEP = dict(scope=Scope.SELECTED_FACES, boundary_mode=BoundaryMode.KEEP_BOUNDARY)
RELOC = dict(scope=Scope.SELECTED_FACES, boundary_mode=BoundaryMode.RELOCATE_SELECTED)


def grid_obj(key="G", n=6, scale=0.4, offset=(0.05, 0.05), verts_offset=(0, 0, 0)):
    v, f = grid(n)
    v = translate(v, verts_offset)
    uv = grid_uv(grid(n)[0], f, scale, offset)
    return build_object(key, v, f, uv={"UVMap": uv})


def center_faces(n=6):
    m = n // 2
    return [(m - 1) * n + (m - 1), (m - 1) * n + m, m * n + (m - 1), m * n + m]


def assert_unselected_exact(obj, plan):
    op = plan.objects[obj.key]
    keep = ~op.writable
    assert np.array_equal(op.uv[keep], obj.input_uv()[keep])
    assert np.array_equal(op.pin[keep], obj.input_pin()[keep])
    assert np.array_equal(op.select[keep], obj.input_select()[keep])


def perturb_vertex(obj, v, d=0.01):
    for l in np.nonzero(obj.loop_vert == v)[0]:
        obj.uv["UVMap"][l] += d


def test_T201_T202_T212_keep_boundary_center():
    o = grid_obj()
    perturb_vertex(o, 24)            # 中央頂点 (内部) をずらす
    o.input_select()[:] = True       # UV選択属性も保持されること
    o.face_select[center_faces()] = True
    frozen = frozen_copy(o)
    plan, vr = run(snapshot_of(o), settings(**KEEP))
    assert_unselected_exact(o, plan)
    assert_unchanged(o, frozen)
    op = plan.objects["G"]
    fixed = plan.partial_summary["fixed_loops"]["G"]
    assert np.array_equal(op.uv[fixed], o.input_uv()[fixed])   # 境界は完全一致
    center_loops = np.nonzero(o.loop_vert == 24)[0]
    assert np.allclose(op.uv[center_loops], (0.05 + 0.2, 0.05 + 0.2), atol=1e-6)
    assert vr.ok    # S/U の正規の接触を重複と誤検出しない


def test_T203_existing_uv_seam_boundary_preserved():
    v, f = grid(4)
    uv = grid_uv(v, f, 0.4, (0.05, 0.05))
    o = build_object("G", v, f, uv={"UVMap": uv})
    # x>=2 の面を右へずらして UV を分断 (共有3D頂点はそのまま)
    for fi in range(o.n_faces):
        if fi % 4 >= 2:
            for l in o.face_loops(fi):
                o.uv["UVMap"][l, 0] += 0.5
    sel = [5, 6, 9, 10]
    o.face_select[sel] = True
    perturb_vertex(o, 6, 0.0)
    plan, vr = run(snapshot_of(o), settings(**KEEP))
    op = plan.objects["G"]
    topo = Topology(o)
    seam_edge = edge_id(o, 7, 12)
    assert not uv_edge_continuous(topo, op.uv, seam_edge)
    fixed = plan.partial_summary["fixed_loops"]["G"]
    assert np.array_equal(op.uv[fixed], o.input_uv()[fixed])
    assert_unselected_exact(o, plan)


def test_T204_pins_holes_disjoint_selection():
    o = grid_obj(n=8)
    sel = [9, 10, 11, 17, 19, 25, 26, 27, 45, 46]   # 穴あきリング + 離れた面群
    o.face_select[sel] = True
    pin_loops = list(o.face_loops(10))[:1]
    o.uv_pin["UVMap"][pin_loops] = True
    perturb_vertex(o, int(o.loop_vert[pin_loops[0]]), 0.003)
    plan, vr = run(snapshot_of(o), settings(**KEEP))
    op = plan.objects["G"]
    assert np.array_equal(op.uv[pin_loops], o.input_uv()[pin_loops])
    assert op.pin[pin_loops].all()
    assert int(op.writable.sum()) == sum(len(list(o.face_loops(f))) for f in sel)  # 拡張なし
    assert_unselected_exact(o, plan)


def test_T205_unsolvable_with_fixed_boundary_fails_without_switch():
    o = grid_obj(n=4)
    o.face_select[[5, 6, 9, 10]] = True
    # 選択境界の2頂点のUVを入れ替えて境界を自己交差させる
    a, b = 6, 8
    la = np.nonzero(o.loop_vert == a)[0]
    lb = np.nonzero(o.loop_vert == b)[0]
    ua, ub = o.uv["UVMap"][la[0]].copy(), o.uv["UVMap"][lb[0]].copy()
    o.uv["UVMap"][la] = ub
    o.uv["UVMap"][lb] = ua
    frozen = frozen_copy(o)
    with pytest.raises(SUOError) as ei:
        run(snapshot_of(o), settings(**KEEP))
    assert ei.value.code in (ErrorCode.UNWRAP_FAILED, ErrorCode.VALIDATION_FAILED)
    assert_unchanged(o, frozen)


def test_T206_relocate_into_free_space():
    o = grid_obj()
    o.face_select[center_faces()] = True
    s = settings(**RELOC)
    plan, vr = run(snapshot_of(o), s)
    assert_unselected_exact(o, plan)
    op = plan.objects["G"]
    w = op.writable
    assert op.uv[w].min() > 0.45 or op.uv[w, 0].min() > 0.45 or op.uv[w, 1].min() > 0.45
    assert plan.partial_summary["density_achieved"]
    topo = Topology(o)
    bedge = edge_id(o, 16, 17)  # 選択境界のエッジ
    assert not uv_edge_continuous(topo, op.uv, bedge)


def test_T207_T208_multi_object_obstacles():
    a = grid_obj("A")
    b = grid_obj("B", offset=(0.5, 0.05), verts_offset=(3, 0, 0))
    a.face_select[center_faces()] = True
    b.face_select[[0, 1]] = True
    plan, vr = run(snapshot_of(a, b), settings(**RELOC))
    assert vr.ok
    tris = []
    for o in (a, b):
        topo = Topology(o)
        tris.append(plan.objects[o.key].uv[topo.tris])
    assert not find_overlaps(np.concatenate(tris), max_pairs=1)
    assert_unselected_exact(a, plan)
    assert_unselected_exact(b, plan)


def test_T209_no_space_fails_and_nothing_written():
    o = grid_obj(scale=0.98, offset=(0.01, 0.01))
    o.face_select[center_faces()] = True
    frozen = frozen_copy(o)
    with pytest.raises(SUOError) as ei:
        run(snapshot_of(o), settings(**RELOC))
    assert ei.value.code == ErrorCode.NO_FEASIBLE_LAYOUT
    assert_unchanged(o, frozen)


def test_T210_udim_relocate_to_allowed_tile():
    o = grid_obj(scale=0.98, offset=(0.01, 0.01))
    o.face_select[center_faces()] = True
    s = settings(tile_mode=TileMode.UDIM, allowed_tiles=(1001, 1002), **RELOC)
    plan, vr = run(snapshot_of(o), s)
    op = plan.objects["G"]
    assert op.uv[op.writable, 0].min() >= 1.0      # 1002 へ配置
    assert_unselected_exact(o, plan)


def test_T211_forbid_on_boundary_and_pinned_collision():
    o = grid_obj()
    o.face_select[center_faces()] = True
    o.edge_constraint[edge_id(o, 16, 17)] = EdgeConstraint.FORBID
    with pytest.raises(SUOError) as ei:
        run(snapshot_of(o), settings(**RELOC))
    assert ei.value.code == ErrorCode.CONSTRAINT_CONFLICT
    o2 = grid_obj()
    o2.face_select[center_faces()] = True
    l0 = list(o2.face_loops(center_faces()[0]))[0]
    o2.uv_pin["UVMap"][l0] = True
    with pytest.raises(SUOError) as ei:
        run(snapshot_of(o2), settings(**RELOC))
    assert ei.value.code == ErrorCode.NO_FEASIBLE_LAYOUT


def test_T213_no_uv_and_zero_selection():
    v, f = grid(2)
    o = build_object("NoUV", v, f)
    o.face_select[:] = True
    with pytest.raises(SUOError) as ei:
        run(snapshot_of(o), settings(**KEEP))
    assert ei.value.code == ErrorCode.INVALID_INPUT
    g = grid_obj()
    plan, _ = run(snapshot_of(g), settings(**KEEP))
    assert plan.no_op and not plan.write_targets()
    a = grid_obj("A")
    b = grid_obj("B", offset=(0.5, 0.05), verts_offset=(3, 0, 0))
    a.face_select[center_faces()] = True
    plan, _ = run(snapshot_of(a, b), settings(**RELOC))
    assert plan.write_targets() == ["A"]
    assert np.array_equal(plan.objects["B"].uv, b.input_uv())


def test_T214_invalid_fixed_uv_rejected():
    o = grid_obj()
    o.face_select[center_faces()] = True
    o.uv["UVMap"][0] = (np.nan, 0.0)
    with pytest.raises(SUOError) as ei:
        run(snapshot_of(o), settings(**KEEP))
    assert ei.value.code == ErrorCode.INVALID_INPUT
    o = grid_obj(offset=(1.05, 0.05))   # 許可外タイル 1002
    o.face_select[center_faces()] = True
    with pytest.raises(SUOError):
        run(snapshot_of(o), settings(**KEEP))
    # 固定Uの既存重複は Apply 不可 (修復しない)
    o = grid_obj()
    o.face_select[center_faces()] = True
    src = o.uv["UVMap"].copy()
    shift = src[list(o.face_loops(1))[0]] - src[list(o.face_loops(0))[0]]
    for l in o.face_loops(0):
        o.uv["UVMap"][l] = src[l] + shift * 0.5   # 面0を面1へ半分重ねる
    plan, vr = run(snapshot_of(o), settings(**KEEP), expect_valid=False)
    assert not vr.ok
    assert any("重複" in i.message for i in vr.errors), [i.describe() for i in vr.errors]


def test_T215_old_region_release_keeps_overlapping_obstacles():
    a = grid_obj("A")
    b = grid_obj("B", n=2, scale=0.1, offset=(0.2, 0.2), verts_offset=(5, 0, 0))  # Aの中央に重なる固定U
    a.face_select[center_faces()] = True
    # A の U と B は重ならないよう、B は A の選択領域上だけに置く
    plan, vr = run(snapshot_of(a, b), settings(**RELOC))
    assert vr.ok
    assert np.array_equal(plan.objects["B"].uv, b.input_uv())


def test_partial_improve_keeps_better_existing():
    o = grid_obj()
    o.face_select[center_faces()] = True
    plan, vr = run(snapshot_of(o), settings(uv_mode=UVMode.IMPROVE, **KEEP))
    assert plan.no_op
    assert np.array_equal(plan.objects["G"].uv, o.input_uv())
