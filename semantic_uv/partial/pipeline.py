"""部分再展開 (設計仕様 6章 / 実装指示書 6.2)。

    candidate = exact_copy_of_source_uv(snapshot)
    for region in stable_order(scope.selected_components):
        charts = generate_or_improve_selected_charts(region, constraints)
        uv = unwrap_on_working_copy(charts, constraints)
        KEEP_BOUNDARY:    verify_fixed_boundary(uv)
        RELOCATE_SELECTED: match_density / place_into_free_domain / reserve
        candidate.write_only_selected_corners(uv)

書込は書込許可集合 (S のコーナー) の外へ出ない。固定座標は最後に元の数値を
代入し直し、その最終値で検証する。KEEP_BOUNDARY から RELOCATE_SELECTED へ
自動で切り替えない。
"""
from __future__ import annotations

import math
from typing import List, Set, Tuple

import numpy as np

from ..analysis.topology import uv_island_faces
from ..core.errors import ErrorCode, SUOError, Target, ValidationReport
from ..core.models import (InputSnapshot, LayoutDomain, ObjectPlan, PartialScope,
                           ResultPlan, UVIsland)
from ..core.settings import BoundaryMode, Settings, UnwrapBackend, UVMode
from ..layout import texel_density as td
from ..layout.grouping import build_groups
from ..layout.obstacles import ObstacleIndex
from ..layout.packing import place_free
from ..seams.optimization import SearchLog, optimize_charts
from ..seams.segmentation import segment_charts
from ..unwrap.chart_unwrap import evaluate_chart, unwrap_charts
from .boundaries import keep_boundary_fixed_loops
from .constraints import check_relocate_constraints

DENSITY_TOLERANCE = 0.05


def run_partial(snapshot: InputSnapshot, a, scope: PartialScope, domain: LayoutDomain,
                settings: Settings, plan: ResultPlan, deadline: float):
    from ..core import pipeline as P

    for obj in snapshot.iter_objects():
        src = obj.input_uv()
        plan.objects[obj.key] = ObjectPlan(
            object_key=obj.key, uv=src.copy(), pin=obj.input_pin().copy(),
            select=obj.input_select().copy(), writable=scope.writable_loops[obj.key].copy())
    summary = plan.partial_summary
    summary["selected_faces"] = {k: int(len(v)) for k, v in scope.selected_faces.items()}
    summary["fixed_faces"] = {k: int(len(v)) for k, v in scope.unselected_faces.items()}
    summary["boundary_edges"] = {k: int(len(v)) for k, v in scope.boundary_edges.items()}
    summary["obstacle_tris"] = int(len(domain.obstacles.tris))
    summary["boundary_mode"] = settings.boundary_mode.value
    summary["movable_faces"] = {k: set() for k in snapshot.order}
    summary["fixed_loops"] = {}

    if scope.selected_face_count == 0:
        plan.no_op = True
        plan.notes.append("選択面がありません。変更は行いません (no-op)")
        P._finish(snapshot, a, plan, [], [], settings, domain, [], generated=set())
        return plan

    rep = ValidationReport()
    if settings.boundary_mode == BoundaryMode.RELOCATE_SELECTED:
        for obj in snapshot.iter_objects():
            check_relocate_constraints(obj, a.topos[obj.key], scope, a.rcs[obj.key], rep)
    rep.raise_if_failed("partial.constraints")

    slog = SearchLog()
    if settings.boundary_mode == BoundaryMode.KEEP_BOUNDARY:
        islands = yield from _keep_boundary(snapshot, a, scope, settings, plan, slog)
        generated: Set[Tuple[str, int]] = set()
    else:
        islands = yield from _relocate(snapshot, a, scope, domain, settings, plan, slog, deadline)
        generated = {(i.object_key, f) for i in islands if i.movable for f in i.faces}
    plan.search_log.extend(slog.lines)
    mats, comps = P.island_material_and_component(snapshot, a.topos, islands) if islands else ([], [])
    groups = build_groups(islands, mats, comps) if islands else []
    P._finish(snapshot, a, plan, islands, groups, settings, domain, [], generated)
    # 生成面の向き検査は再配置した島のみ。境界固定は元の向きに従う。
    _density_report(snapshot, a, plan, islands, domain, settings)
    if not any(np.any(plan.objects[k].uv != snapshot.objects[k].input_uv()) for k in snapshot.order):
        plan.no_op = True
        plan.notes.append("有効な変更はありませんでした (全コーナーが固定されているか、改善候補が既存より良くありません)")
    yield 1.0, "完了"
    return plan


# ------------------------------------------------------------------
def _keep_boundary(snapshot, a, scope, settings, plan, slog):
    from ..core import pipeline as P
    islands: List[UVIsland] = []
    total = sum(len(c) for c in scope.components.values()) or 1
    done = 0
    for obj in snapshot.iter_objects():
        topo, rc = a.topos[obj.key], a.rcs[obj.key]
        src = obj.input_uv()
        op = plan.objects[obj.key]
        fixed_all = np.zeros(obj.n_loops, dtype=bool)
        for comp in scope.components[obj.key]:
            for sub in uv_island_faces(topo, src, comp):
                fixed = keep_boundary_fixed_loops(topo, sub, src, rc.must_cut)
                fixed |= scope.pinned_loops[obj.key]
                loops = np.asarray([l for f in sub for l in obj.face_loops(f)], dtype=np.int64)
                fixed_all[loops] |= fixed[loops]
                if fixed[loops].all():
                    plan.notes.append(f"{obj.key}: 面{sub[0]}..の全コーナーが境界固定のため変更なし")
                    continue
                pinned = {int(l): (float(src[l, 0]), float(src[l, 1])) for l in loops if fixed[l]}
                res = unwrap_charts(topo, [sub], rc.must_cut, UnwrapBackend.BUILTIN_LSCM,
                                    pinned_loops=pinned, weld_uv=src, initial_uv=src)[0]
                if not res.valid:
                    raise SUOError(ErrorCode.UNWRAP_FAILED,
                                   f"境界を固定したままでは有効な展開が得られません ({res.reason})",
                                   stage="partial.keep_boundary",
                                   targets=[Target(obj.key, face=f) for f in sub[:10]],
                                   advice="選択範囲を変更するか、明示的に RELOCATE_SELECTED を選んでください "
                                          "(自動では切り替えません)")
                if settings.uv_mode == UVMode.IMPROVE:
                    ok, _, ma, _, ad, _, _ = evaluate_chart(topo, sub, src, require_positive=False)
                    if ok and (ma + 0.5 * ad) <= (res.angle_distortion + 0.5 * res.area_distortion):
                        slog.add(f"{obj.key}: 面{sub[0]}..は既存UVの方が良いため維持")
                        continue
                for l in loops:
                    if fixed[l]:
                        op.uv[l] = src[l]          # 固定値は元の数値を代入し直す
                    else:
                        op.uv[l] = res.loop_uv[int(l)]
                slog.add(f"{obj.key}: 面{sub[0]}..の内部 {int((~fixed[loops]).sum())} コーナーを再計算 ({res.method})")
                islands.append(P.make_island(obj, topo, sub, op.uv, "improved", movable=False))
            done += 1
            yield 0.1 + 0.7 * done / total, f"{obj.key}: 境界固定で再展開"
        plan.partial_summary["fixed_loops"][obj.key] = fixed_all
    return islands


# ------------------------------------------------------------------
def _relocate(snapshot, a, scope, domain, settings, plan, slog, deadline):
    from ..core import pipeline as P
    islands: List[UVIsland] = []
    pinned_flags: List[bool] = []
    total = sum(len(c) for c in scope.components.values()) or 1
    done = 0
    for obj in snapshot.iter_objects():
        topo, rc, feats = a.topos[obj.key], a.rcs[obj.key], a.feats[obj.key]
        src = obj.input_uv()
        pins = scope.pinned_loops[obj.key]
        bcut = scope.boundary_edges[obj.key]
        for comp in scope.components[obj.key]:
            if settings.uv_mode == UVMode.IMPROVE:
                units = uv_island_faces(topo, src, comp)
            else:
                units = [comp]
            for unit in units:
                loops = [l for f in unit for l in obj.face_loops(f)]
                has_pin = bool(pins[loops].any())
                if settings.uv_mode == UVMode.IMPROVE:
                    ok, _, ma, _, _, _, _ = evaluate_chart(topo, unit, src, require_positive=False)
                    if ok and (ma <= settings.distortion_limit or has_pin):
                        islands.append(P.make_island(obj, topo, unit, src, "kept", movable=not has_pin))
                        pinned_flags.append(has_pin)
                        continue
                if has_pin:
                    # pin を含む単位は分割せず、pin 位置を保ったまま内蔵ソルバーで解く
                    pinned = {int(l): (float(src[l, 0]), float(src[l, 1])) for l in loops if pins[l]}
                    res = unwrap_charts(topo, [unit], rc.must_cut | bcut, UnwrapBackend.BUILTIN_LSCM,
                                        pinned_loops=pinned)[0]
                    if not res.valid:
                        raise SUOError(ErrorCode.UNWRAP_FAILED,
                                       f"pin を保持した展開が得られません ({res.reason})",
                                       stage="partial.relocate",
                                       targets=[Target(obj.key, face=f) for f in unit[:10]])
                    islands.append(P.make_island(obj, topo, unit, res.loop_uv, "generated", movable=False))
                    pinned_flags.append(True)
                    continue
                charts = segment_charts(topo, feats, rc, unit, settings.chart_angle_limit_deg,
                                        extra_cut=bcut)
                charts, results = optimize_charts(
                    topo, feats, rc, charts, backend=settings.backend, budget=settings.search,
                    weights=settings.weights, distortion_limit=settings.distortion_limit,
                    base_cut=bcut, log=slog, deadline=deadline)
                for r in results:
                    islands.append(P.make_island(obj, topo, r.faces, r.loop_uv, "generated"))
                    pinned_flags.append(False)
            done += 1
            yield 0.1 + 0.5 * done / total, f"{obj.key}: 選択側を分離して展開"
    for obj in snapshot.iter_objects():
        cut = scope.boundary_edges[obj.key]
        plan.objects[obj.key].candidate_seams |= {e for e in cut if len(a.topos[obj.key].edge_faces[e]) == 2}

    # 密度: 指定値、または固定Uの有効な島から推定。U が無ければ全体配置の決定に従う。
    k = domain.density_k
    has_u = len(domain.obstacles.tris) > 0
    if k is None and not has_u:
        if any(pinned_flags):
            raise SUOError(ErrorCode.NO_FEASIBLE_LAYOUT,
                           "固定領域がない場合の全体配置では pin 付きの島を動かせません",
                           stage="partial.relocate", advice="pin を外すか、密度を明示指定してください")
        groups, k = P.layout_full(snapshot, a.topos, islands, domain, settings, plan.search_log)
        _write_islands(plan, islands)
        for isl in islands:
            plan.partial_summary["movable_faces"][isl.object_key].update(isl.faces)
        return islands
    if k is None:
        raise SUOError(ErrorCode.NO_FEASIBLE_LAYOUT, "固定領域から代表密度を推定できません",
                       stage="partial.density", advice="密度を明示指定してください")
    plan.partial_summary["density_k"] = k
    plan.partial_summary["density_px"] = td.px_per_unit(k, domain.resolution)
    index = ObstacleIndex(domain.obstacles, domain.gap_uv)
    s = math.sqrt(k)
    order = sorted(range(len(islands)),
                   key=lambda i: (not pinned_flags[i], -islands[i].world_area, islands[i].uid))
    for n, i in enumerate(order):
        isl = islands[i]
        obj = snapshot.objects[isl.object_key]
        topo = a.topos[isl.object_key]
        lt = P.island_local_tris(topo, isl)
        if pinned_flags[i]:
            # 固定点を動かす配置変換は禁止: その場で衝突を検査
            if index.collides(isl.uv[lt], domain.gap_uv):
                raise SUOError(ErrorCode.NO_FEASIBLE_LAYOUT,
                               "pin により移動できない選択島が固定UVと衝突します",
                               stage="partial.relocate",
                               targets=[Target(isl.object_key, face=f) for f in isl.faces[:10]],
                               advice="pin を外すか、選択範囲を変更してください (制約は自動で外しません)")
            index.add(isl.uv[lt], isl.uid)
            continue
        item = P.normalized_item(obj, topo, isl, i, free_rotation=isl.origin != "kept",
                                 allow_rotate=True)
        p = place_free(item, s, domain.tiles, index, domain.gap_uv, domain.edge_margin_uv,
                       settings.search.placement_candidates)
        if p is None:
            raise SUOError(ErrorCode.NO_FEASIBLE_LAYOUT,
                           "指定予算で実行可能な空き領域が見つかりません (密度を維持したまま)",
                           stage="partial.relocate",
                           targets=[Target(isl.object_key, face=f) for f in isl.faces[:10]],
                           advice="許可タイルの追加を明示指定する、密度設定を変更する、または選択を変更してください")
        isl.uv = p.apply(item.uv)
        isl.tile = p.tile
        index.add(isl.uv[item.tris], isl.uid)   # 先に配置した島も後続の障害物
        yield 0.6 + 0.3 * (n + 1) / max(len(order), 1), "空き領域へ配置"
    _write_islands(plan, islands)
    for isl in islands:
        plan.partial_summary["movable_faces"][isl.object_key].update(isl.faces)
    return islands


def _write_islands(plan: ResultPlan, islands: List[UVIsland]) -> None:
    for isl in islands:
        op = plan.objects[isl.object_key]
        assert op.writable[isl.loops].all(), "書込許可集合の外への書込"
        op.uv[isl.loops] = isl.uv
        op.island_tiles[isl.faces[0]] = isl.tile


def _density_report(snapshot, a, plan, islands, domain, settings) -> None:
    k_ref = domain.density_k
    devs = []
    for isl in islands:
        lt = None
        from ..core import pipeline as P
        lt = P.island_local_tris(a.topos[isl.object_key], isl)
        op = plan.objects[isl.object_key]
        uv = op.uv[isl.loops]
        t = uv[lt]
        ua = float(np.abs(0.5 * ((t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1])
                                 - (t[:, 1, 1] - t[:, 0, 1]) * (t[:, 2, 0] - t[:, 0, 0]))).sum())
        k = td.k_of(ua, isl.world_area)
        if k and k_ref:
            devs.append((td.deviation(k, k_ref), isl))
    if not devs:
        return
    worst = max(d for d, _ in devs)
    plan.partial_summary["density_deviation_max"] = worst
    plan.partial_summary["density_achieved"] = worst <= DENSITY_TOLERANCE
    if worst > DENSITY_TOLERANCE:
        if settings.boundary_mode == BoundaryMode.KEEP_BOUNDARY:
            plan.warnings.append(f"境界固定のため密度統一を達成できません (最大偏差 {worst * 100:.1f}%)。"
                                 "境界固定を優先しています")
        else:
            plan.warnings.append(f"密度偏差が目標 ±5% を超えています (最大 {worst * 100:.1f}%)")
    plan.partial_summary["note_u_density"] = "未選択部分の密度は変更していません"
