"""意味的グループ化と配置順。

階層: マテリアル → Object → メッシュ接続成分。グループ内では
対称ペアと類似形状を隣接させ、残りは面積の降順。
可読性重み < 充填重み の場合は全体を面積降順にし、グループは同順位の判定にのみ使う。
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from ..core.models import UVGroup, UVIsland
from ..core.settings import SoftWeights
from .similarity import similar_clusters


def build_groups(islands: Sequence[UVIsland], island_material: Sequence[str],
                 island_component: Sequence[int]) -> List[UVGroup]:
    keys: Dict[Tuple[str, str, int], int] = {}
    groups: List[UVGroup] = []
    for i, isl in enumerate(islands):
        key = (island_material[i], isl.object_key, int(island_component[i]))
        if key not in keys:
            keys[key] = len(groups)
            groups.append(UVGroup(gid=len(groups), members=[], object_key=isl.object_key))
        groups[keys[key]].members.append(i)
    for g in groups:
        for i in g.members:
            islands[i].group = g.gid
    return groups


def placement_order(islands: Sequence[UVIsland], groups: List[UVGroup],
                    weights: SoftWeights, areas: Sequence[float]) -> List[int]:
    if weights.readability < weights.packing:
        return sorted(range(len(islands)), key=lambda i: (-areas[i], islands[i].group, islands[i].uid))
    labels = similar_clusters([isl.descriptor for isl in islands])
    group_area = {g.gid: sum(areas[i] for i in g.members) for g in groups}
    order: List[int] = []
    for g in sorted(groups, key=lambda g: (-group_area[g.gid], g.gid)):
        members = sorted(g.members, key=lambda i: (-areas[labels[i]] if labels[i] >= 0 else 0,
                                                   labels[i], -areas[i], islands[i].uid))
        placed = set()
        for i in members:
            if i in placed:
                continue
            order.append(i)
            placed.add(i)
            p = islands[i].symmetry_partner
            if p is not None and p in g.members and p not in placed:
                order.append(p)
                placed.add(p)
        # グループ外の対称相手は後で自身のグループ順で入る
    seen = set()
    out = []
    for i in order:
        if i not in seen:
            out.append(i)
            seen.add(i)
    for i in range(len(islands)):
        if i not in seen:
            out.append(i)
    return out
