"""全体配置の改善。固定候補数の配置順・回転方針を試し、
ソフト目的 (充填倍率と可読性) が最良のものを選ぶ。シード固定で決定的。"""
from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

from ..core.settings import SearchBudget, SoftWeights
from .packing import PackItem, Placement, pack_full


def improve_layout(items: Sequence[PackItem], base_order: List[int], tiles: Sequence[int],
                   gap: float, margin: float, weights: SoftWeights, budget: SearchBudget,
                   group_of: Sequence[int], log: List[str]
                   ) -> Tuple[Optional[List[Placement]], float]:
    rng = random.Random(budget.seed)
    candidates: List[Tuple[str, List[int], bool]] = [("base", list(base_order), False)]
    if weights.packing >= weights.readability:
        candidates.append(("base+rotate", list(base_order), True))
    n_variants = max(0, min(budget.max_candidates // 16, 6))
    # グループの並びだけを入れ替える (グループ内の近接は維持)
    groups: List[List[int]] = []
    for i in base_order:
        g = group_of[i]
        if groups and group_of[groups[-1][0]] == g:
            groups[-1].append(i)
        else:
            groups.append([i])
    for v in range(n_variants):
        gs = list(groups)
        if len(gs) > 1:
            a = rng.randrange(len(gs))
            b = rng.randrange(len(gs))
            gs[a], gs[b] = gs[b], gs[a]
        candidates.append((f"variant{v}", [i for g in gs for i in g], False))

    best: Optional[List[Placement]] = None
    best_s = 0.0
    best_name = ""
    for name, order, rot in candidates:
        placements, s = pack_full(items, order, tiles, gap, margin, prefer_rotate=rot)
        if placements is None:
            log.append(f"layout {name}: 配置不可")
            continue
        # 可読性のため基本順は僅差なら優先 (1% 以内)
        threshold = best_s * (1.01 if name != "base" else 1.0)
        log.append(f"layout {name}: scale={s:.6f}")
        if best is None or s > threshold:
            best, best_s, best_name = placements, s, name
    if best is not None:
        log.append(f"layout 採用: {best_name}")
    return best, best_s
