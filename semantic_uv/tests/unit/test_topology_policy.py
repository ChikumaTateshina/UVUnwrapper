"""非多様体・向き不整合・面積0の面を含む実データ相当の入力 (Issue: 実モデルで展開できない)。"""
import numpy as np
import pytest

from helpers import assert_unchanged, frozen_copy, run, settings
from fixtures.meshes import build_object, grid, snapshot_of
from semantic_uv.analysis.topology import Topology, uv_edge_continuous
from semantic_uv.core.errors import ErrorCode, SUOError
from semantic_uv.core.settings import TopologyPolicy


def messy_object():
    v, f = grid(4)
    f = [list(face) for face in f]
    f[5] = f[5][::-1]                      # 法線が逆の面 (向き不整合エッジ)
    v = list(v) + [(0.25, 0.25, 0.5)]      # 3面目を共有エッジ (6,7) に立てる (非多様体)
    f.append([6, 7, len(v) - 1])
    v.append((0.5, 1.0, 0.0))              # 頂点 22 と同位置
    v.append((0.75, 1.0, 0.0))
    f.append([22, 26, 27])                 # 一直線上の3点 = 面積0の面
    return build_object("Messy", v, f)


def test_split_policy_unwraps_and_cuts_problem_edges():
    o = messy_object()
    frozen = frozen_copy(o)
    plan, vr = run(snapshot_of(o), settings(topology_policy=TopologyPolicy.SPLIT))
    assert vr.ok
    topo = Topology(o)
    from semantic_uv.analysis.topology import nonmanifold_edges
    for e in nonmanifold_edges(topo):
        if len(topo.edge_faces[e]) == 2:
            assert not uv_edge_continuous(topo, plan.objects["Messy"].uv, e)
    assert np.all(np.isfinite(plan.objects["Messy"].uv))
    assert plan.quality.max_area_distortion < 1e3
    assert_unchanged(o, frozen)          # メッシュ・既存データは変更しない


def test_stop_policy_reports_and_stops():
    o = messy_object()
    with pytest.raises(SUOError) as ei:
        run(snapshot_of(o), settings(topology_policy=TopologyPolicy.STOP))
    assert ei.value.code == ErrorCode.INVALID_INPUT
