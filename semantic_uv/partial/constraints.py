"""部分再展開のハード制約: 境界分離と FORBID、pin。"""
from __future__ import annotations

from ..analysis.topology import Topology
from ..core.errors import ErrorCode, Target, ValidationReport
from ..core.models import ObjectData, PartialScope
from ..seams.constraints import ResolvedConstraints


def check_relocate_constraints(obj: ObjectData, topo: Topology, scope: PartialScope,
                               rc: ResolvedConstraints, rep: ValidationReport) -> None:
    """RELOCATE_SELECTED は S/U 境界を UV 上で切り離す。FORBID と衝突すれば停止し、
    制約を勝手に解除しない。"""
    bad = []
    for e in sorted(scope.boundary_edges[obj.key]):
        if len(topo.edge_faces[e]) == 2 and e in rc.must_join:
            bad.append(e)
    if bad:
        rep.error(ErrorCode.CONSTRAINT_CONFLICT,
                  "選択境界に FORBID 制約があり、選択側を分離できません",
                  stage="partial.constraints",
                  targets=[Target(obj.key, edge=(int(obj.edges[e][0]), int(obj.edges[e][1]))) for e in bad[:10]],
                  advice="FORBID を外すか、境界固定モード (KEEP_BOUNDARY) を使用してください")
