"""Chart の局所探索 (設計仕様 5章 / R-105)。

高速な初期分割を展開・検証し、問題のある Chart だけに
シーム追加 (分割) と Chart 統合を試す。ハード制約は越えない。
探索の主予算は反復数と候補数。結果はエッジ・面番号の順序だけに依存し決定的。
"""
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..analysis.shape_features import edge_lengths_world
from ..analysis.topology import Topology
from ..core.errors import ErrorCode, SUOError, Target
from ..core.models import EdgeFeatures
from ..core.settings import SearchBudget, SoftWeights, UnwrapBackend
from ..unwrap.chart_unwrap import ChartResult, unwrap_charts
from .constraints import ResolvedConstraints
from .segmentation import chart_cut_edges


@dataclass
class SearchLog:
    lines: List[str] = field(default_factory=list)
    iterations: int = 0
    candidates: int = 0
    timed_out: bool = False

    def add(self, text: str) -> None:
        self.lines.append(text)


def _forbid_units(topo: Topology, faces: Sequence[int], rc: ResolvedConstraints) -> Dict[int, int]:
    """FORBID で結ばれた面のまとまり: face -> unit 代表。"""
    fset = set(faces)
    parent = {f: f for f in faces}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for e in sorted(rc.must_join):
        fs = topo.edge_faces[e]
        if len(fs) == 2 and fs[0] in fset and fs[1] in fset:
            a, b = find(fs[0]), find(fs[1])
            if a != b:
                parent[max(a, b)] = min(a, b)
    return {f: find(f) for f in faces}


def split_chart(topo: Topology, faces: Sequence[int], rc: ResolvedConstraints
                ) -> Optional[Tuple[List[int], List[int]]]:
    """測地的に最も離れた2つの面から領域を成長させ2分割する。"""
    faces = sorted(faces)
    if len(faces) < 2:
        return None
    unit = _forbid_units(topo, faces, rc)
    units = sorted(set(unit.values()))
    if len(units) < 2:
        return None
    obj = topo.obj
    w = obj.world_co()
    cen = {f: w[obj.loop_vert[list(obj.face_loops(f))]].mean(axis=0) for f in faces}
    ucen = {}
    for u in units:
        mem = [f for f in faces if unit[f] == u]
        ucen[u] = np.mean([cen[f] for f in mem], axis=0)
    allc = np.mean(list(ucen.values()), axis=0)
    s0 = max(units, key=lambda u: (float(np.linalg.norm(ucen[u] - allc)), -u))
    s1 = max(units, key=lambda u: (float(np.linalg.norm(ucen[u] - ucen[s0])), -u))
    if s0 == s1:
        return None
    fset = set(faces)
    # unit 隣接グラフ
    adj: Dict[int, Dict[int, float]] = {u: {} for u in units}
    for f in faces:
        for e, g in topo.neighbors(f):
            if g in fset and unit[g] != unit[f]:
                a, b = unit[f], unit[g]
                d = float(np.linalg.norm(ucen[a] - ucen[b]))
                adj[a][b] = min(adj[a].get(b, d), d)
    owner: Dict[int, int] = {}
    heap = [(0.0, 0, s0), (0.0, 1, s1)]
    best = {s0: 0.0, s1: 0.0}
    while heap:
        d, lab, u = heapq.heappop(heap)
        if u in owner:
            continue
        owner[u] = lab
        for v, wgt in sorted(adj[u].items()):
            nd = d + wgt
            if v not in owner and nd < best.get(v, float("inf")):
                best[v] = nd
                heapq.heappush(heap, (nd, lab, v))
    # 非連結の unit は近い方へ
    for u in units:
        if u not in owner:
            owner[u] = 0 if np.linalg.norm(ucen[u] - ucen[s0]) <= np.linalg.norm(ucen[u] - ucen[s1]) else 1
    a = [f for f in faces if owner[unit[f]] == 0]
    b = [f for f in faces if owner[unit[f]] == 1]
    if not a or not b:
        return None
    return a, b


AREA_LIMIT_FACTOR = 4.0   # 面積歪み (max(r,1/r)-1) の上限 = 角度歪み上限 × 係数


def _combined(res: ChartResult) -> float:
    """角度歪みと面積歪みの合成値 (Chart 比較用)。"""
    return res.angle_distortion + res.area_distortion / AREA_LIMIT_FACTOR


def _needs_work(res: ChartResult, limit: float) -> bool:
    return (not res.valid) or res.angle_distortion > limit or \
        res.area_distortion > limit * AREA_LIMIT_FACTOR


def optimize_charts(topo: Topology, feats: EdgeFeatures, rc: ResolvedConstraints,
                    charts: List[List[int]], *, backend: UnwrapBackend,
                    budget: SearchBudget, weights: SoftWeights,
                    distortion_limit: float, base_cut: Set[int],
                    log: SearchLog, deadline: float,
                    allow_merge: bool = True) -> Tuple[List[List[int]], List[ChartResult]]:
    """返り値: (charts, results)。有効解が得られない Chart があれば UNWRAP_FAILED。"""
    obj = topo.obj
    lengths = edge_lengths_world(topo)
    charts = [sorted(c) for c in charts]

    def cuts_for(chs):
        # Chart 間のエッジは各 Chart 内の溶接に影響しないため、Chart 内部の切断だけを渡す
        return base_cut | rc.must_cut

    results = unwrap_charts(topo, charts, cuts_for(charts), backend)
    total_area = sum(r.world_area for r in results if r.valid) or \
        float(topo.world_face_areas()[[f for c in charts for f in c]].sum()) or 1.0

    # --- シーム追加 (分割) -------------------------------------------
    # 1ラウンドで問題 Chart を全て分割候補にし、展開は1回の呼び出しにまとめる。
    it = 0
    max_rounds = budget.max_iterations
    unsplittable: Set[int] = set()   # 分割できない Chart の最小面番号
    for _round in range(max_rounds):
        if time.monotonic() > deadline:
            log.timed_out = True
            break
        bad = [i for i, r in enumerate(results)
               if _needs_work(r, distortion_limit) and charts[i][0] not in unsplittable]
        if not bad:
            break
        bad.sort(key=lambda i: (results[i].valid, -_combined(results[i]), charts[i][0]))
        splits: List[Tuple[int, List[int], List[int]]] = []
        for i in bad:
            sp = split_chart(topo, charts[i], rc)
            if sp is None:
                unsplittable.add(charts[i][0])
                continue
            splits.append((i, sorted(sp[0]), sorted(sp[1])))
        if not splits:
            break
        flat = [c for _, a, b in splits for c in (a, b)]
        subs = unwrap_charts(topo, flat, cuts_for(None), backend)
        it += len(splits)
        log.iterations += len(splits)
        replace: Dict[int, Tuple[List[List[int]], List[ChartResult]]] = {}
        for k, (i, a, b) in enumerate(splits):
            sub = subs[2 * k:2 * k + 2]
            old = results[i]
            accept = (not old.valid) or (all(x.valid for x in sub) and
                                         max(_combined(x) for x in sub) < _combined(old))
            if accept:
                log.add(f"{obj.key}: Chart(面{charts[i][0]}..)を分割 "
                        f"({'無効解の解消' if not old.valid else f'角度歪み {old.angle_distortion:.3f} / 面積歪み {old.area_distortion:.3f}'})")
                replace[i] = ([a, b], sub)
            else:
                unsplittable.add(charts[i][0])
        if not replace:
            break
        new_charts: List[List[int]] = []
        new_results: List[ChartResult] = []
        for i in range(len(charts)):
            if i in replace:
                new_charts += replace[i][0]
                new_results += replace[i][1]
            else:
                new_charts.append(charts[i])
                new_results.append(results[i])
        charts, results = new_charts, new_results

    invalid = [r for r in results if not r.valid]
    if invalid:
        r = invalid[0]
        raise SUOError(ErrorCode.UNWRAP_FAILED,
                       f"有効な展開が得られないChartがあります: {r.reason}",
                       stage="unwrap", targets=[Target(obj.key, face=f) for f in r.faces[:10]],
                       advice="面の退化や非多様体を確認するか、REQUIRE シームを追加してください")

    # --- Chart 統合 ------------------------------------------------
    # 1ラウンドで共有境界の長い順に候補をまとめて展開し、Δcost の良い順に
    # 互いに素な組だけを採用する。
    if allow_merge and len(charts) > 1:
        n0 = len(charts)
        sqrt_area = math.sqrt(total_area)
        tried: Set[Tuple[int, int, int, int]] = set()
        cand_used = 0
        batch = max(4, min(32, budget.max_candidates // 4))
        while cand_used < budget.max_candidates:
            if time.monotonic() > deadline:
                log.timed_out = True
                break
            owner = {f: ci for ci, ch in enumerate(charts) for f in ch}
            shared: Dict[Tuple[int, int], float] = {}
            for e in sorted(chart_cut_edges(topo, charts)):
                if e in rc.must_cut or e in base_cut:
                    continue
                a, b = topo.edge_faces[e]
                ca, cb = owner[a], owner[b]
                key = (min(ca, cb), max(ca, cb))
                shared[key] = shared.get(key, 0.0) + float(lengths[e])
            pairs = sorted(shared.items(),
                           key=lambda kv: (-kv[1], charts[kv[0][0]][0], charts[kv[0][1]][0]))
            cands = []
            for (ca, cb), slen in pairs:
                sig = (charts[ca][0], charts[cb][0], len(charts[ca]), len(charts[cb]))
                if sig in tried:
                    continue
                if cand_used + len(cands) >= budget.max_candidates or len(cands) >= batch:
                    break
                tried.add(sig)
                cands.append((ca, cb, slen))
            if not cands:
                break
            cand_used += len(cands)
            log.candidates += len(cands)
            merged_list = [sorted(charts[ca] + charts[cb]) for ca, cb, _ in cands]
            rs = unwrap_charts(topo, merged_list, cuts_for(None), backend)
            scored = []
            for (ca, cb, slen), merged, r in zip(cands, merged_list, rs):
                if _needs_work(r, distortion_limit):
                    continue
                ra, rb = results[ca], results[cb]
                delta = weights.distortion * (_combined(r) * r.world_area
                                              - _combined(ra) * ra.world_area
                                              - _combined(rb) * rb.world_area) / total_area
                delta -= weights.seam_length * slen / sqrt_area
                delta -= weights.island_count / n0
                if delta < 0:
                    scored.append((delta, charts[ca][0], charts[cb][0], ca, cb, merged, r))
            if not scored:
                continue
            scored.sort(key=lambda x: (x[0], x[1], x[2]))
            used: Set[int] = set()
            merges = []
            for delta, _, _, ca, cb, merged, r in scored:
                if ca in used or cb in used:
                    continue
                used.update((ca, cb))
                merges.append((ca, cb, merged, r))
                log.add(f"{obj.key}: Chart統合 (面{charts[ca][0]}.. + 面{charts[cb][0]}..) Δcost={delta:.4f}")
            keep = [k for k in range(len(charts)) if k not in used]
            charts = [charts[k] for k in keep] + [m[2] for m in merges]
            results = [results[k] for k in keep] + [m[3] for m in merges]
            order = sorted(range(len(charts)), key=lambda k: charts[k][0])
            charts = [charts[k] for k in order]
            results = [results[k] for k in order]
    methods: Dict[str, int] = {}
    for r in results:
        methods[r.method] = methods.get(r.method, 0) + 1
    log.add(f"{obj.key}: 展開方式 " + ", ".join(f"{k} x{v}" for k, v in sorted(methods.items())))
    return charts, results
