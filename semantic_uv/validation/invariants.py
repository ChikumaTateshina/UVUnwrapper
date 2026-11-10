"""不変条件の検証。許容差で変更を隠さない (要素単位の完全一致)。"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..analysis.topology import Topology, uv_edge_continuous
from ..core.errors import ErrorCode, Target, ValidationReport
from ..core.models import ObjectData, ObjectPlan
from ..seams.constraints import ResolvedConstraints


def verify_unwritable_exact(obj: ObjectData, plan: ObjectPlan, rep: ValidationReport,
                            stage: str = "invariants") -> None:
    """書込許可集合の外 (未選択面コーナー) は入力UVと完全一致。"""
    src = obj.input_uv()
    keep = ~plan.writable
    if src is None:
        if keep.any():
            rep.error(ErrorCode.VALIDATION_FAILED, "入力UVがないのに未書込コーナーがあります",
                      stage=stage, targets=[Target(obj.key)])
        return
    diff = keep & np.any(plan.uv != src, axis=1)
    pin_diff = keep & (plan.pin != obj.input_pin())
    sel_diff = keep & (plan.select != obj.input_select())
    for mask, what in ((diff, "座標"), (pin_diff, "pin"), (sel_diff, "UV選択属性")):
        if mask.any():
            idx = np.nonzero(mask)[0][:10]
            rep.error(ErrorCode.VALIDATION_FAILED, f"未選択コーナーの{what}が変化しています",
                      stage=stage,
                      targets=[Target(obj.key, face=int(obj.loop_face()[l]), corner=int(l)) for l in idx])


def verify_fixed_loops(obj: ObjectData, plan: ObjectPlan, fixed: np.ndarray,
                       rep: ValidationReport, what: str, stage: str = "invariants") -> None:
    src = obj.input_uv()
    if src is None or not fixed.any():
        return
    diff = fixed & np.any(plan.uv != src, axis=1)
    if diff.any():
        idx = np.nonzero(diff)[0][:10]
        rep.error(ErrorCode.VALIDATION_FAILED, f"{what}が元の座標と一致しません", stage=stage,
                  targets=[Target(obj.key, face=int(obj.loop_face()[l]), corner=int(l)) for l in idx])


def verify_hard_constraints(obj: ObjectData, topo: Topology, plan: ObjectPlan,
                            rc: ResolvedConstraints, rep: ValidationReport,
                            stage: str = "invariants") -> None:
    """REQUIRE は出力UVで不連続、FORBID は連続。書込対象の面に接するエッジのみ判定
    (固定Uの既存状態は変更できないため、違反は警告)。"""
    lf = obj.loop_face()
    wf = set(int(f) for f in np.unique(lf[plan.writable])) if plan.writable.any() else set()
    for e in sorted(rc.must_cut | rc.must_join):
        fs = topo.edge_faces[e]
        if len(fs) != 2:
            continue
        cont = uv_edge_continuous(topo, plan.uv, e)
        violated = (e in rc.must_cut and cont) or (e in rc.must_join and not cont)
        if not violated:
            continue
        a, b = obj.edges[e]
        if e in rc.topology_cuts:
            kind = "非多様体・向き不整合エッジの分割"
        else:
            kind = "REQUIRE/保護シーム" if e in rc.must_cut else "FORBID"
        if fs[0] in wf or fs[1] in wf:
            rep.error(ErrorCode.CONSTRAINT_CONFLICT, f"{kind} 制約に違反しています", stage=stage,
                      targets=[Target(obj.key, edge=(int(a), int(b)))])
        else:
            rep.warn(ErrorCode.CONSTRAINT_CONFLICT,
                     f"固定された未選択部分が {kind} 制約を満たしていません (変更しません)",
                     stage=stage, targets=[Target(obj.key, edge=(int(a), int(b)))])
