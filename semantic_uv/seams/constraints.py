"""エッジ制約の解決 (設計仕様 5章)。

AUTO / PREFER / AVOID はソフト、REQUIRE / FORBID はハード制約。
seam_mode:
  PROTECT: 既存シームを REQUIRE と同等に維持する。
  PREFER : 既存シームに変更コストを課す (切断を優先、非シームの切断に罰則)。
  IGNORE : 既存シームを採点に用いない。明示 REQUIRE/FORBID は解除しない。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Set

import numpy as np

from ..core.errors import ErrorCode, Target, ValidationReport
from ..core.models import EdgeConstraint, ObjectData
from ..core.settings import SeamMode
from ..analysis.topology import Topology


@dataclass
class ResolvedConstraints:
    object_key: str
    must_cut: Set[int] = field(default_factory=set)      # ハード: 必ず切断
    must_join: Set[int] = field(default_factory=set)     # ハード: 切断禁止
    protected_seams: Set[int] = field(default_factory=set)  # PROTECT 由来 (must_cut に含む)
    soft_bias: np.ndarray = None                          # (E,) +で切りやすく, -で切りにくく


def resolve_constraints(obj: ObjectData, topo: Topology, seam_mode: SeamMode,
                        report: ValidationReport) -> ResolvedConstraints:
    E = obj.n_edges
    rc = ResolvedConstraints(obj.key, soft_bias=np.zeros(E))
    cons = obj.edge_constraint
    for e in range(E):
        c = int(cons[e])
        if c == EdgeConstraint.REQUIRE:
            rc.must_cut.add(e)
        elif c == EdgeConstraint.FORBID:
            rc.must_join.add(e)
        elif c == EdgeConstraint.PREFER:
            rc.soft_bias[e] += 1.0
        elif c == EdgeConstraint.AVOID:
            rc.soft_bias[e] -= 1.0
    seams = obj.seams
    if seam_mode == SeamMode.PROTECT:
        for e in np.nonzero(seams)[0]:
            e = int(e)
            if e in rc.must_join:
                report.error(ErrorCode.CONSTRAINT_CONFLICT,
                             "保護対象の既存シームに FORBID が指定されています",
                             stage="constraints", targets=[Target(obj.key, edge=_ev(obj, e))],
                             advice="既存シームを外すか、シームモードを PREFER/IGNORE にしてください")
                continue
            rc.must_cut.add(e)
            rc.protected_seams.add(e)
    elif seam_mode == SeamMode.PREFER:
        rc.soft_bias[seams] += 0.6
        rc.soft_bias[~seams] -= 0.15
    # IGNORE: 何もしない
    both = rc.must_cut & rc.must_join
    if both:
        report.error(ErrorCode.CONSTRAINT_CONFLICT,
                     "同一エッジに REQUIRE と FORBID が指定されています",
                     stage="constraints",
                     targets=[Target(obj.key, edge=_ev(obj, e)) for e in sorted(both)],
                     advice="どちらか一方の制約を外してください")
    # 境界・非多様体エッジの FORBID は意味を持たないが矛盾ではない
    check_forbid_closure(obj, topo, rc, report)
    return rc


def check_forbid_closure(obj: ObjectData, topo: Topology, rc: ResolvedConstraints,
                         report: ValidationReport) -> None:
    """FORBID で結合される面集合が REQUIRE で一周切断される等、
    ハード制約だけでは表現できない矛盾は展開後の検証で検出する。
    ここでは REQUIRE エッジの両側面が FORBID だけで結ばれ、かつ
    その切断が無意味になる (同一頂点上で閉じない) 場合を警告する。"""
    if not rc.must_cut or not rc.must_join:
        return
    # 同じ面ペアを REQUIRE と FORBID の別エッジで結ぶ場合 (三角柱など) は矛盾
    pair_req = {}
    for e in rc.must_cut:
        fs = topo.edge_faces[e]
        if len(fs) == 2:
            pair_req[tuple(sorted(fs))] = e
    for e in rc.must_join:
        fs = topo.edge_faces[e]
        if len(fs) == 2 and tuple(sorted(fs)) in pair_req:
            report.error(ErrorCode.CONSTRAINT_CONFLICT,
                         "同じ面の組に REQUIRE と FORBID が同時に指定されています",
                         stage="constraints",
                         targets=[Target(obj.key, edge=_ev(obj, e)),
                                  Target(obj.key, edge=_ev(obj, pair_req[tuple(sorted(fs))]))])


def _ev(obj: ObjectData, e: int):
    a, b = obj.edges[e]
    return (int(a), int(b))
