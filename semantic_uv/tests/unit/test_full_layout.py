"""全体展開 (T-101〜T-110, T-112 の純データ版)。"""
import math

import numpy as np
import pytest

from helpers import assert_unchanged, frozen_copy, run, settings
from fixtures.meshes import (build_object, cube, cylinder, edge_id, grid, grid_uv, scale_matrix,
                             snapshot_of, translate, uv_sphere)
from semantic_uv.core.errors import ErrorCode, SUOError
from semantic_uv.core.models import EdgeConstraint
from semantic_uv.core.settings import SeamMode, TileMode, UVMode
from semantic_uv.analysis.topology import Topology, uv_edge_continuous
from semantic_uv.layout.domain import island_tile
from semantic_uv.validation.uv_validity import output_islands


def three_objects():
    v, f = cube()
    a = build_object("A_Cube", v, f)
    v, f = cylinder()
    b = build_object("B_Cyl", translate(v, (3, 0, 0)), f)
    v, f = uv_sphere()
    c = build_object("C_Sph", v, f, matrix=scale_matrix(1, 2, 1, (0, 4, 0)))
    return a, b, c


def test_T101_multiple_objects_common_space_no_overlap():
    objs = three_objects()
    plan, vr = run(snapshot_of(*objs), settings())
    assert set(plan.write_targets()) == {o.key for o in objs}
    assert vr.ok  # validate_plan は全対象横断の重複を検査する
    assert plan.quality.island_count >= 3


def test_T102_generate_keeps_existing_uv():
    v, f = grid(3)
    o = build_object("G", v, f, uv={"UVMap": grid_uv(v, f), "Second": grid_uv(v, f, 0.5)})
    frozen = frozen_copy(o)
    plan, _ = run(snapshot_of(o), settings())
    assert_unchanged(o, frozen)
    assert plan.objects["G"].writable.all()


def test_T103_improve_uses_existing_and_not_worse():
    v, f = grid(4)
    o = build_object("G", v, f, uv={"UVMap": grid_uv(v, f, 0.8, (0.1, 0.1))})
    plan, _ = run(snapshot_of(o), settings(uv_mode=UVMode.IMPROVE))
    assert plan.baseline_quality is not None
    assert plan.quality.total_cost <= plan.baseline_quality.total_cost + 1e-9
    # 正常な既存UVは維持される
    assert np.array_equal(plan.objects["G"].uv, o.input_uv())


def test_T103_improve_repairs_flipped_island():
    v, f = grid(4)
    uv = grid_uv(v, f, 0.8, (0.1, 0.1))
    o = build_object("G", v, f, uv={"UVMap": uv})
    # 中央頂点を大きく動かして反転を作る
    center = 12
    for l in np.nonzero(o.loop_vert == center)[0]:
        o.uv["UVMap"][l] = (0.95, 0.95)
    plan, _ = run(snapshot_of(o), settings(uv_mode=UVMode.IMPROVE))
    assert any("再展開" in s for s in plan.search_log)


def test_T104_density_uniform_with_nonuniform_scale():
    v, f = cube()
    a = build_object("A", v, f)
    b = build_object("B", v, f, matrix=scale_matrix(2, 1, 0.5, (4, 0, 0)))
    plan, _ = run(snapshot_of(a, b), settings())
    assert plan.quality.density_deviation_max <= 0.05


def test_T105_protect_require_forbid():
    v, f = grid(4)
    seam = [(2, 7), (7, 12)]
    o = build_object("G", v, f, seams=seam)
    req = edge_id(o, 12, 17)
    forbid = edge_id(o, 6, 7)
    o.edge_constraint[req] = EdgeConstraint.REQUIRE
    o.edge_constraint[forbid] = EdgeConstraint.FORBID
    frozen = frozen_copy(o)
    plan, _ = run(snapshot_of(o), settings(seam_mode=SeamMode.PROTECT))
    topo = Topology(o)
    uv = plan.objects["G"].uv
    for e in [edge_id(o, *s) for s in seam] + [req]:
        assert not uv_edge_continuous(topo, uv, e)
    assert uv_edge_continuous(topo, uv, forbid)
    assert_unchanged(o, frozen)


def test_conflicting_hard_constraints_stop():
    v, f = grid(2)
    o = build_object("G", v, f, seams=[(1, 4)])
    o.edge_constraint[edge_id(o, 1, 4)] = EdgeConstraint.FORBID
    with pytest.raises(SUOError) as ei:
        run(snapshot_of(o), settings(seam_mode=SeamMode.PROTECT))
    assert ei.value.code == ErrorCode.CONSTRAINT_CONFLICT


def test_T106_prefer_reports_changes():
    v, f = cylinder(caps=False)
    o = build_object("Cyl", v, f, seams=[(0, 12)])
    plan, _ = run(snapshot_of(o), settings(seam_mode=SeamMode.PREFER))
    assert any("PREFER" in n for n in plan.notes) or plan.objects["Cyl"].candidate_seams


def test_T107_ignore_independent_of_soft_seams():
    v, f = cylinder()
    a = build_object("Cyl", v, f, seams=[(0, 12), (1, 13)])
    b = build_object("Cyl", v, f, seams=[(5, 17)])
    pa, _ = run(snapshot_of(a), settings(seam_mode=SeamMode.IGNORE))
    pb, _ = run(snapshot_of(b), settings(seam_mode=SeamMode.IGNORE))
    assert np.array_equal(pa.objects["Cyl"].uv, pb.objects["Cyl"].uv)


def test_T108_symmetric_parts_not_stacked():
    v1, f1 = cube(0.5)
    v2 = translate(v1, (-1.5, 0, 0))
    v1 = translate(v1, (1.5, 0, 0))
    verts = v1 + v2
    faces = f1 + [[i + 8 for i in face] for face in f1]
    o = build_object("Sym", verts, faces)
    plan, vr = run(snapshot_of(o), settings())
    assert vr.ok
    partners = [i.symmetry_partner for i in plan.islands if i.symmetry_partner is not None]
    assert partners, "対称部品が検出されていない"
    assert plan.quality.orientation_agreement >= 0.5


def test_T109_single_tile_with_margin():
    objs = three_objects()
    s = settings()
    plan, _ = run(snapshot_of(*objs), s)
    g = s.gap_uv
    for key, op in plan.objects.items():
        assert op.uv.min() >= g - 1e-9 and op.uv.max() <= 1 - g + 1e-9


def test_T110_udim_allowed_tiles_only():
    objs = list(three_objects())
    v, f = uv_sphere(16, 8)
    for i in range(3):
        objs.append(build_object(f"S{i}", translate(v, (0, 0, 3 * i + 3)), f))
    s = settings(tile_mode=TileMode.UDIM, allowed_tiles=(1001, 1002, 1011))
    plan, vr = run(snapshot_of(*objs), s)
    used = set()
    for o in objs:
        topo = Topology(o)
        for isl in output_islands(o, topo, plan.objects[o.key].uv, plan.objects[o.key].writable):
            t = island_tile(plan.objects[o.key].uv[topo.tris[isl.tri_ids]].reshape(-1, 2))
            assert t in (1001, 1002, 1011)
            used.add(t)
    assert len(used) >= 1


def test_T112_deterministic():
    objs = three_objects()
    p1, _ = run(snapshot_of(*objs), settings())
    p2, _ = run(snapshot_of(*three_objects()), settings())
    for k in p1.order:
        assert np.array_equal(p1.objects[k].uv, p2.objects[k].uv)
    assert [i.tile for i in p1.islands] == [i.tile for i in p2.islands]
    assert [i.group for i in p1.islands] == [i.group for i in p2.islands]


def test_specified_density_too_large_fails():
    v, f = cube(4.0)
    o = build_object("Big", v, f)
    from semantic_uv.core.settings import DensityMode
    with pytest.raises(SUOError) as ei:
        run(snapshot_of(o), settings(density_mode=DensityMode.SPECIFIED, density_px_per_unit=4096))
    assert ei.value.code == ErrorCode.NO_FEASIBLE_LAYOUT
