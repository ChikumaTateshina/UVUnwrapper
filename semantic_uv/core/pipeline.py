"""純粋な処理パイプライン (実装指示書 3章の関数契約)。

capture_input / commit_plan / rollback / restore は Blender 側
(blender.mesh_adapter, persistence.transaction) にある。ここは Blender を import しない。

    validate_input(snapshot, settings) -> ValidationReport
    build_scope(snapshot, settings)    -> PartialScope | FullScope
    build_domain(snapshot, scope, settings) -> LayoutDomain
    compute_plan(snapshot, scope, domain, settings) -> ResultPlan
    validate_plan(snapshot, plan, settings) -> ValidationReport
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, Generator, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from ..analysis.materials import material_name
from ..analysis.shape_features import shape_descriptor
from ..analysis.symmetry import find_mirror_pairs
from ..analysis.topology import (Topology, connected_components, degenerate_faces,
                                 nonmanifold_edges, uv_edge_continuous, uv_island_faces)
from ..layout import texel_density as td
from ..layout.domain import island_tile, make_domain
from ..layout.grouping import build_groups, placement_order
from ..layout.optimization import improve_layout
from ..layout.orientation import choose_rotation
from ..layout.packing import PackItem, pack_full, place_free
from ..layout.obstacles import ObstacleIndex
from ..analysis.shape_features import rotation
from ..seams.constraints import ResolvedConstraints, resolve_constraints
from ..seams.optimization import SearchLog, optimize_charts
from ..seams.scoring import compute_edge_features
from ..seams.segmentation import segment_charts
from ..unwrap.chart_unwrap import evaluate_chart
from ..validation.invariants import (verify_fixed_loops, verify_hard_constraints,
                                     verify_unwritable_exact)
from ..validation.quality import compute_quality
from ..validation.uv_validity import validate_uv
from .errors import ErrorCode, SUOError, Target, ValidationReport
from .models import (EdgeFeatures, FullScope, InputSnapshot, LayoutDomain, ObjectData,
                     ObjectPlan, ObstacleSet, PartialScope, ResultPlan, UVGroup, UVIsland)
from .settings import (DensityMode, Scope, Settings, SharedMeshPolicy, UVMode,
                       validate_settings)

ProgressIter = Generator[Tuple[float, str], None, ResultPlan]
DENSITY_TOLERANCE = 0.05


# ===================================================================
@dataclass
class Analysis:
    topos: Dict[str, Topology]
    rcs: Dict[str, ResolvedConstraints]
    feats: Dict[str, EdgeFeatures]
    report: ValidationReport


_analysis_cache: Dict[Tuple[str, str], Analysis] = {}


def analyze(snapshot: InputSnapshot, settings: Settings) -> Analysis:
    key = (snapshot.fingerprint(), settings.fingerprint())
    cached = _analysis_cache.get(key)
    if cached is not None:
        return cached
    rep = ValidationReport()
    topos, rcs, feats = {}, {}, {}
    for obj in snapshot.iter_objects():
        topo = Topology(obj)
        topos[obj.key] = topo
        rc = resolve_constraints(obj, topo, settings.seam_mode, rep)
        rcs[obj.key] = rc
        feats[obj.key] = compute_edge_features(obj, topo, rc)
    a = Analysis(topos, rcs, feats, rep)
    _analysis_cache.clear()
    _analysis_cache[key] = a
    return a


# ===================================================================
def validate_input(snapshot: InputSnapshot, settings: Settings) -> ValidationReport:
    rep = ValidationReport()
    for p in validate_settings(settings):
        rep.error(ErrorCode.INVALID_INPUT, p, stage="settings")
    for name, reason in snapshot.excluded:
        rep.info(ErrorCode.INVALID_INPUT, f"対象外: {name} ({reason})", stage="input")
    if not snapshot.order:
        rep.error(ErrorCode.INVALID_INPUT, "対象となる Mesh Object が選択されていません", stage="input")
        return rep
    a = analyze(snapshot, settings)
    rep.extend(a.report)
    mesh_owner: Dict[str, str] = {}
    partial = settings.scope == Scope.SELECTED_FACES
    for obj in snapshot.iter_objects():
        tgt = [Target(obj.key)]
        if not obj.editable:
            rep.error(ErrorCode.NOT_EDITABLE, "リンクまたは読取専用のデータです", stage="input", targets=tgt)
        m = np.asarray(obj.matrix_world)[:3, :3]
        scales = np.linalg.norm(m, axis=0)
        if np.any(scales <= 1e-12) or abs(np.linalg.det(m)) <= 1e-18:
            rep.error(ErrorCode.INVALID_INPUT, "ゼロスケールの変換を持つObjectです", stage="input",
                      targets=tgt, advice="スケールを0以外にしてください")
        if obj.mirror_transform:
            rep.info(ErrorCode.INVALID_INPUT, "負スケールの変換を考慮して面積と向きを計算します",
                     stage="input", targets=tgt)
        # 共有 Mesh
        others_selected = obj.mesh_key in mesh_owner
        if obj.mesh_users > 1 or others_selected:
            if settings.shared_mesh_policy == SharedMeshPolicy.STOP:
                users = ", ".join(obj.mesh_user_keys) if obj.mesh_user_keys else str(obj.mesh_users)
                rep.error(ErrorCode.SHARED_MESH,
                          f"Mesh Data '{obj.mesh_key}' が複数の Object で共有されています ({users})",
                          stage="input", targets=tgt,
                          advice="共有Mesh独立化を明示的に許可するか、対象を変更してください")
            else:
                rep.warn(ErrorCode.SHARED_MESH,
                         f"Mesh Data '{obj.mesh_key}' を確定時に独立化します (非対象ユーザーは変更しません)",
                         stage="input", targets=tgt)
        mesh_owner.setdefault(obj.mesh_key, obj.key)
        has_uv = obj.input_uv() is not None
        if (partial or settings.uv_mode == UVMode.IMPROVE) and not has_uv:
            rep.error(ErrorCode.INVALID_INPUT,
                      "入力UVがありません" + ("（部分再展開には入力UVが必要です）" if partial else ""),
                      stage="input", targets=tgt,
                      advice="全体生成 (ALL_FACES / GENERATE) を実行してください")
        topo = a.topos[obj.key]
        if partial:
            region = set(np.nonzero(obj.face_select & ~obj.face_hide)[0].tolist())
        else:
            region = set(range(obj.n_faces))
        if not region:
            continue
        nm = [e for e in nonmanifold_edges(topo)
              if any(f in region for f in topo.edge_faces[e])]
        if nm:
            rep.error(ErrorCode.INVALID_INPUT, "非多様体または向きが不整合なエッジがあります",
                      stage="input",
                      targets=[Target(obj.key, edge=(int(obj.edges[e][0]), int(obj.edges[e][1]))) for e in nm[:10]],
                      advice="トポロジーを修正してから実行してください (自動修復は行いません)")
        dg = [f for f in degenerate_faces(topo) if f in region]
        if dg:
            rep.error(ErrorCode.INVALID_INPUT, "退化した面があります", stage="input",
                      targets=[Target(obj.key, face=f) for f in dg[:10]],
                      advice="面積0の面や重複頂点を修正してください (自動修復は行いません)")
    return rep


def build_scope(snapshot: InputSnapshot, settings: Settings) -> Union[FullScope, PartialScope]:
    if settings.scope == Scope.ALL_FACES:
        return FullScope({o.key: np.arange(o.n_faces, dtype=np.int64) for o in snapshot.iter_objects()})
    from ..partial.selection import build_partial_scope
    a = analyze(snapshot, settings)
    return build_partial_scope(snapshot, a.topos, settings.boundary_mode.value)


def build_domain(snapshot: InputSnapshot, scope, settings: Settings) -> LayoutDomain:
    k_spec = td.k_from_px(settings.density_px_per_unit, settings.resolution) \
        if settings.density_mode == DensityMode.SPECIFIED else None
    if not isinstance(scope, PartialScope):
        return make_domain(settings, density_k=k_spec, density_source="specified" if k_spec else "auto")
    a = analyze(snapshot, settings)
    rep = ValidationReport()
    tris: List[np.ndarray] = []
    owners: List[Tuple[str, int]] = []
    pairs = []
    allowed = set(settings.effective_tiles())
    for obj in snapshot.iter_objects():
        topo = a.topos[obj.key]
        uv = obj.input_uv()
        U = scope.unselected_faces[obj.key]
        if uv is None or len(U) == 0:
            continue
        lf_bad = [f for f in U if not np.all(np.isfinite(uv[list(obj.face_loops(int(f)))]))]
        if lf_bad:
            rep.error(ErrorCode.INVALID_INPUT, "固定される未選択UVに非有限の座標があります",
                      stage="domain", targets=[Target(obj.key, face=int(f)) for f in lf_bad[:10]],
                      advice="該当面を選択に追加してください (未選択部分は修復しません)")
            continue
        for comp in uv_island_faces(topo, uv, U.tolist()):
            tids = [t for f in comp for t in topo.face_tris(f)]
            if not tids:
                continue
            t_uv = uv[topo.tris[tids]]
            tile = island_tile(t_uv.reshape(-1, 2))
            if tile is None or tile not in allowed:
                rep.error(ErrorCode.INVALID_INPUT,
                          "固定UVが許可タイル外にあります" + (f" (タイル {tile})" if tile else " (タイル境界をまたいでいます)"),
                          stage="domain", targets=[Target(obj.key, face=comp[0])],
                          advice="許可タイルを設定に追加するか、該当面を選択してください")
            tris.append(t_uv)
            owners.extend((obj.key, int(topo.tri_face[t])) for t in tids)
            wa = float(topo.world_tri_areas()[tids].sum())
            ua = float(np.abs(_signed(t_uv)).sum())
            pairs.append((ua, wa))
    rep.raise_if_failed("domain")
    obst = ObstacleSet(np.concatenate(tris) if tris else np.zeros((0, 3, 2)), owners)
    if k_spec:
        return make_domain(settings, obst, k_spec, "specified")
    k_fixed = td.representative_k(pairs)
    return make_domain(settings, obst, k_fixed, "fixed" if k_fixed else "auto")


def _signed(t: np.ndarray) -> np.ndarray:
    return 0.5 * ((t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1])
                  - (t[:, 1, 1] - t[:, 0, 1]) * (t[:, 2, 0] - t[:, 0, 0]))


# ===================================================================
# 島と配置の共通処理
def make_island(obj: ObjectData, topo: Topology, faces: Sequence[int],
                loop_uv: Dict[int, Tuple[float, float]] | np.ndarray, origin: str,
                movable: bool = True) -> UVIsland:
    loops = np.asarray([l for f in faces for l in obj.face_loops(f)], dtype=np.int64)
    if isinstance(loop_uv, dict):
        uv = np.asarray([loop_uv[int(l)] for l in loops], dtype=np.float64)
    else:
        uv = np.asarray(loop_uv[loops], dtype=np.float64)
    wa = float(topo.world_face_areas()[list(faces)].sum())
    isl = UVIsland(object_key=obj.key, faces=sorted(int(f) for f in faces), loops=loops, uv=uv,
                   world_area=wa, locked=not movable, movable=movable, origin=origin)
    isl.descriptor = shape_descriptor(topo, faces)
    return isl


def island_local_tris(topo: Topology, isl: UVIsland) -> np.ndarray:
    pos = {int(l): i for i, l in enumerate(isl.loops)}
    tids = [t for f in isl.faces for t in topo.face_tris(f)]
    if not tids:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray([[pos[int(l)] for l in topo.tris[t]] for t in tids], dtype=np.int64)


def island_uv_area(isl: UVIsland, tris: np.ndarray) -> float:
    return isl.uv_area(tris)


def normalized_item(obj: ObjectData, topo: Topology, isl: UVIsland, index: int,
                    free_rotation: bool, allow_rotate: bool) -> PackItem:
    """島を k=1 (A_uv = A_world) に正規化し、方向を決めた PackItem。"""
    tris = island_local_tris(topo, isl)
    ua = isl.uv_area(tris)
    s = math.sqrt(isl.world_area / ua) if ua > 0 and isl.world_area > 0 else 1.0
    uv = (isl.uv - isl.uv.mean(axis=0)) * s
    world = obj.world_co()[obj.loop_vert[isl.loops]]
    theta = choose_rotation(uv, world, allow_free=free_rotation)
    uv = uv @ rotation(theta).T
    uv = uv - uv.min(axis=0)
    return PackItem(index=index, uv=uv, tris=tris, allow_rotate=allow_rotate)


def assign_symmetry(snapshot: InputSnapshot, islands: List[UVIsland]) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    by_obj: Dict[str, List[int]] = {}
    for i, isl in enumerate(islands):
        if isl.movable:
            by_obj.setdefault(isl.object_key, []).append(i)
    for key, idxs in by_obj.items():
        obj = snapshot.objects[key]
        found = find_mirror_pairs(obj, [islands[i].faces for i in idxs])
        for a, b, _axis, _conf in found:
            ia, ib = idxs[a], idxs[b]
            islands[ia].symmetry_partner = ib
            islands[ib].symmetry_partner = ia
            pairs.append((ia, ib))
    return pairs


def island_material_and_component(snapshot: InputSnapshot, topos: Dict[str, Topology],
                                  islands: List[UVIsland]) -> Tuple[List[str], List[int]]:
    comp_of: Dict[str, Dict[int, int]] = {}
    for key in {i.object_key for i in islands}:
        topo = topos[key]
        m: Dict[int, int] = {}
        for ci, comp in enumerate(connected_components(topo, range(snapshot.objects[key].n_faces))):
            for f in comp:
                m[f] = ci
        comp_of[key] = m
    mats, comps = [], []
    for isl in islands:
        obj = snapshot.objects[isl.object_key]
        counts: Dict[int, float] = {}
        areas = topos[isl.object_key].world_face_areas()
        for f in isl.faces:
            mi = int(obj.face_material[f])
            counts[mi] = counts.get(mi, 0.0) + float(areas[f])
        best = max(sorted(counts), key=lambda k: counts[k]) if counts else 0
        mats.append(material_name(obj, best))
        comps.append(comp_of[isl.object_key].get(isl.faces[0], 0))
    return mats, comps


def layout_full(snapshot: InputSnapshot, topos: Dict[str, Topology], islands: List[UVIsland],
                domain: LayoutDomain, settings: Settings, log: List[str],
                keep_direction: Optional[Set[int]] = None,
                order_override: Optional[List[int]] = None) -> Tuple[List[UVGroup], float]:
    """全島を共通倍率で配置し island.uv / island.tile を更新する。返り値: (groups, k)。"""
    keep_direction = keep_direction or set()
    items = []
    for i, isl in enumerate(islands):
        obj = snapshot.objects[isl.object_key]
        keep = i in keep_direction
        items.append(normalized_item(obj, topos[isl.object_key], isl, i,
                                     free_rotation=not keep, allow_rotate=not keep))
    mats, comps = island_material_and_component(snapshot, topos, islands)
    groups = build_groups(islands, mats, comps)
    areas = [isl.world_area for isl in islands]
    order = order_override or placement_order(islands, groups, settings.weights, areas)
    gap, margin = domain.gap_uv, domain.edge_margin_uv
    if domain.density_k:
        s = math.sqrt(domain.density_k)
        placements, scale = pack_full(items, order, domain.tiles, gap, margin, fixed_scale=s)
        if placements is None:
            raise SUOError(ErrorCode.NO_FEASIBLE_LAYOUT,
                           f"指定密度 ({settings.density_px_per_unit:g} px/単位) では許可タイルに収まりません"
                           if domain.density_source == "specified" else "指定密度では許可タイルに収まりません",
                           stage="layout",
                           advice="密度を下げる、許可タイルを追加する、または対象を減らしてください")
    else:
        placements, scale = improve_layout(items, order, domain.tiles, gap, margin,
                                           settings.weights, settings.search,
                                           [isl.group for isl in islands], log)
        if placements is None:
            raise SUOError(ErrorCode.NO_FEASIBLE_LAYOUT,
                           "余白を確保したまま許可タイルへ配置できません (指定予算で実行可能解なし)",
                           stage="layout",
                           advice="余白を小さくする、許可タイルを追加する、またはシームを減らしてください")
    for p in placements:
        isl = islands[p.index]
        isl.uv = p.apply(items[p.index].uv)
        isl.tile = p.tile
    return groups, scale * scale


# ===================================================================
def compute_plan(snapshot: InputSnapshot, scope, domain: LayoutDomain, settings: Settings,
                 progress: Optional[Callable[[float, str], None]] = None) -> ResultPlan:
    it = compute_plan_iter(snapshot, scope, domain, settings)
    while True:
        try:
            frac, msg = next(it)
            if progress:
                progress(frac, msg)
        except StopIteration as stop:
            return stop.value


def compute_plan_iter(snapshot: InputSnapshot, scope, domain: LayoutDomain,
                      settings: Settings) -> ProgressIter:
    """段階ごとに (進捗, メッセージ) を yield する。取消は generator.close() で行い、
    作業コピーのみを扱うため元データは変化しない。"""
    a = analyze(snapshot, settings)
    deadline = time.monotonic() + settings.search.time_limit_s
    yield 0.05, "解析完了"
    plan = ResultPlan(objects={}, order=list(snapshot.order),
                      input_fingerprint=snapshot.fingerprint(),
                      object_fingerprints={k: o.fingerprint() for k, o in snapshot.objects.items()},
                      settings_fingerprint=settings.fingerprint())
    if isinstance(scope, PartialScope):
        from ..partial.pipeline import run_partial
        result = yield from run_partial(snapshot, a, scope, domain, settings, plan, deadline)
        return result
    result = yield from _run_full(snapshot, a, domain, settings, plan, deadline)
    return result


def _init_object_plans(snapshot: InputSnapshot, plan: ResultPlan, writable_all: bool) -> None:
    for obj in snapshot.iter_objects():
        src = obj.input_uv()
        uv = src.copy() if src is not None else np.zeros((obj.n_loops, 2))
        writable = np.ones(obj.n_loops, dtype=bool) if writable_all else np.zeros(obj.n_loops, dtype=bool)
        plan.objects[obj.key] = ObjectPlan(object_key=obj.key, uv=uv,
                                           pin=obj.input_pin().copy(), select=obj.input_select().copy(),
                                           writable=writable)


def _run_full(snapshot: InputSnapshot, a: Analysis, domain: LayoutDomain, settings: Settings,
              plan: ResultPlan, deadline: float) -> ProgressIter:
    _init_object_plans(snapshot, plan, writable_all=True)
    slog = SearchLog()
    islands: List[UVIsland] = []
    kept: Set[int] = set()
    n_obj = len(snapshot.order)
    improve = settings.uv_mode == UVMode.IMPROVE
    any_reunwrap = False
    for oi, obj in enumerate(snapshot.iter_objects()):
        topo, rc, feats = a.topos[obj.key], a.rcs[obj.key], a.feats[obj.key]
        faces = list(range(obj.n_faces))
        if improve:
            src = obj.input_uv()
            groups_bad: List[List[int]] = []
            for comp in _improve_units(obj, topo, rc, src, faces):
                ok, reason, ma, _, _, _, _ = evaluate_chart(topo, comp, src, require_positive=False)
                violates = _violates_hard(topo, rc, src, comp)
                if ok and ma <= settings.distortion_limit and not violates:
                    kept.add(len(islands))
                    islands.append(make_island(obj, topo, comp, src, "kept"))
                else:
                    why = reason or ("ハード制約違反" if violates else f"歪み {ma:.3f}")
                    slog.add(f"{obj.key}: 島(面{comp[0]}..)を再展開: {why}")
                    groups_bad.append(comp)
            _warn_seam_mismatch(obj, topo, src, plan)
            for comp in groups_bad:
                any_reunwrap = True
                charts = segment_charts(topo, feats, rc, comp, settings.chart_angle_limit_deg)
                charts, results = optimize_charts(
                    topo, feats, rc, charts, backend=settings.backend, budget=settings.search,
                    weights=settings.weights, distortion_limit=settings.distortion_limit,
                    base_cut=set(), log=slog, deadline=deadline)
                for r in results:
                    islands.append(make_island(obj, topo, r.faces, r.loop_uv, "improved"))
        else:
            charts = segment_charts(topo, feats, rc, faces, settings.chart_angle_limit_deg)
            charts, results = optimize_charts(
                topo, feats, rc, charts, backend=settings.backend, budget=settings.search,
                weights=settings.weights, distortion_limit=settings.distortion_limit,
                base_cut=set(), log=slog, deadline=deadline)
            for r in results:
                islands.append(make_island(obj, topo, r.faces, r.loop_uv, "generated"))
        yield 0.1 + 0.6 * (oi + 1) / n_obj, f"{obj.key}: 展開完了"
    if slog.timed_out:
        plan.warnings.append("時間上限に達したため局所探索を打ち切りました (検証済みの解のみ使用)")
    plan.search_log.extend(slog.lines)
    plan.search_log.append(f"探索: 反復 {slog.iterations}, 候補 {slog.candidates}")

    sym_pairs = assign_symmetry(snapshot, islands)
    layout_log: List[str] = []
    baseline_ok = False
    if improve:
        baseline_ok = _baseline_valid(snapshot, a, settings, domain)
        if not any_reunwrap and baseline_ok:
            plan.notes.append("既存UVは全ての検査に合格しており、改善の必要はありません (出力は入力と同一)")
            _finish(snapshot, a, plan, islands, [], settings, domain, sym_pairs, generated=set())
            return plan
    if improve and any_reunwrap:
        placed = _try_place_into_existing(snapshot, a, islands, kept, domain, settings, layout_log)
        if placed is not None:
            groups = placed
            # 位置を保持した既存島は余白検査の対象外 (既存の余白不足は変更しない)
            mv: Dict[str, Set[int]] = {o: set() for o in snapshot.order}
            for i, isl in enumerate(islands):
                if i not in kept:
                    mv[isl.object_key].update(isl.faces)
            plan.partial_summary["movable_faces"] = mv
        else:
            order = sorted(kept, key=lambda i: (float(islands[i].uv[:, 1].mean()),
                                                float(islands[i].uv[:, 0].mean())))
            rest = [i for i in range(len(islands)) if i not in kept]
            groups, _ = layout_full(snapshot, a.topos, islands, domain, settings, layout_log,
                                    keep_direction=kept, order_override=order + rest)
    elif improve:
        # 島は正常だが配置・密度が不正: 方向を保って再配置
        order = sorted(range(len(islands)), key=lambda i: (float(islands[i].uv[:, 1].mean()),
                                                            float(islands[i].uv[:, 0].mean())))
        groups, _ = layout_full(snapshot, a.topos, islands, domain, settings, layout_log,
                                keep_direction=set(range(len(islands))), order_override=order)
    else:
        groups, _ = layout_full(snapshot, a.topos, islands, domain, settings, layout_log)
    plan.search_log.extend(layout_log)
    yield 0.85, "配置完了"
    for isl in islands:
        op = plan.objects[isl.object_key]
        op.uv[isl.loops] = isl.uv
        op.pin[isl.loops] = False
        op.island_tiles[isl.faces[0]] = isl.tile
    generated = {(i.object_key, f) for i in islands if i.origin != "kept" for f in i.faces}
    _finish(snapshot, a, plan, islands, groups, settings, domain, sym_pairs, generated)
    if improve and baseline_ok and plan.baseline_quality is not None and \
            plan.quality.total_cost > plan.baseline_quality.total_cost + 1e-12:
        # 正常な基準UVより悪化する候補は採用しない
        plan.notes.append("改善候補の総合コストが既存UVより悪いため、既存UVを維持します")
        _init_object_plans(snapshot, plan, writable_all=True)
        _finish(snapshot, a, plan, [], [], settings, domain, [], generated=set())
    yield 1.0, "完了"
    return plan


def _improve_units(obj, topo, rc, src, faces) -> List[List[int]]:
    """入力UVの島。FORBID なのに分断されている島同士はまとめて再展開単位にする。"""
    comps = uv_island_faces(topo, src, faces)
    owner = {f: i for i, c in enumerate(comps) for f in c}
    parent = list(range(len(comps)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for e in sorted(rc.must_join):
        fs = topo.edge_faces[e]
        if len(fs) == 2:
            ra, rb = find(owner[fs[0]]), find(owner[fs[1]])
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    merged: Dict[int, List[int]] = {}
    for i, c in enumerate(comps):
        merged.setdefault(find(i), []).extend(c)
    return [sorted(v) for _, v in sorted(merged.items())]


def _violates_hard(topo: Topology, rc: ResolvedConstraints, uv: np.ndarray, faces) -> bool:
    fset = set(faces)
    for e in rc.must_cut:
        fs = topo.edge_faces[e]
        if len(fs) == 2 and fs[0] in fset and fs[1] in fset and uv_edge_continuous(topo, uv, e):
            return True
    for e in rc.must_join:
        fs = topo.edge_faces[e]
        if len(fs) == 2 and (fs[0] in fset or fs[1] in fset) and not uv_edge_continuous(topo, uv, e):
            return True
    return False


def _warn_seam_mismatch(obj: ObjectData, topo: Topology, uv: np.ndarray, plan: ResultPlan) -> None:
    mism = 0
    for e in range(obj.n_edges):
        if len(topo.edge_faces[e]) != 2:
            continue
        if bool(obj.seams[e]) == uv_edge_continuous(topo, uv, e):
            mism += 1
    if mism:
        plan.warnings.append(f"{obj.key}: シームフラグと実際のUV分断が {mism} 本のエッジで一致しません "
                             "(改善モードは実際のUV連続性を優先します)")


def _baseline_valid(snapshot: InputSnapshot, a: Analysis, settings: Settings,
                    domain: LayoutDomain) -> bool:
    objs = list(snapshot.iter_objects())
    uvs = {o.key: o.input_uv() for o in objs}
    wr = {o.key: np.ones(o.n_loops, dtype=bool) for o in objs}
    rep = validate_uv(objs, a.topos, uvs, wr, domain.tiles, domain.gap_uv, domain.edge_margin_uv)
    if not rep.ok:
        return False
    q = compute_quality(objs, a.topos, uvs, len(domain.tiles), settings.weights)
    return q.density_deviation_max <= DENSITY_TOLERANCE


def _try_place_into_existing(snapshot, a, islands, kept, domain, settings, log):
    """正常な既存島を動かさずに、再展開した島だけを空き領域へ置く。"""
    if not kept:
        return None
    kept_list = sorted(kept)
    tris, owners, pairs = [], [], []
    for i in kept_list:
        isl = islands[i]
        topo = a.topos[isl.object_key]
        lt = island_local_tris(topo, isl)
        t = isl.uv[lt]
        tile = island_tile(t.reshape(-1, 2))
        if tile is None or tile not in domain.tiles:
            return None
        tris.append(t)
        owners.extend([(isl.object_key, isl.faces[0])] * len(t))
        pairs.append((isl.uv_area(lt), isl.world_area))
    k = td.representative_k(pairs)
    if not k or any(td.deviation(ua / wa, k) > DENSITY_TOLERANCE for ua, wa in pairs if wa > 0):
        return None
    # 既存島同士の重複があれば保持できない
    from ..validation.overlaps import find_overlaps
    if find_overlaps(np.concatenate(tris), max_pairs=1):
        return None
    index = ObstacleIndex(ObstacleSet(np.concatenate(tris), owners), domain.gap_uv)
    s = math.sqrt(k)
    new = [i for i in range(len(islands)) if i not in kept]
    new.sort(key=lambda i: (-islands[i].world_area, islands[i].uid))
    for i in new:
        isl = islands[i]
        obj = snapshot.objects[isl.object_key]
        item = normalized_item(obj, a.topos[isl.object_key], isl, i, True, True)
        p = place_free(item, s, domain.tiles, index, domain.gap_uv, domain.edge_margin_uv,
                       settings.search.placement_candidates)
        if p is None:
            log.append("既存配置を保った空き領域配置に失敗、全体を再配置します")
            return None
        isl.uv = p.apply(item.uv)
        isl.tile = p.tile
        index.add(isl.uv[item.tris], isl.uid)
    for i in kept_list:
        islands[i].tile = island_tile(islands[i].uv) or 1001
    mats, comps = island_material_and_component(snapshot, a.topos, islands)
    log.append("正常な既存島の位置を保持し、再展開した島を空き領域へ配置しました")
    return build_groups(islands, mats, comps)


def _finish(snapshot, a, plan: ResultPlan, islands, groups, settings, domain, sym_pairs,
            generated) -> None:
    objs = list(snapshot.iter_objects())
    uvs = {k: p.uv for k, p in plan.objects.items()}
    gmap = {(isl.object_key, isl.faces[0]): isl.group for isl in islands if isl.group >= 0}
    pairs = [((islands[i].object_key, islands[i].faces[0]), (islands[j].object_key, islands[j].faces[0]))
             for i, j in sym_pairs]
    plan.islands = islands
    plan.groups = groups
    plan.quality = compute_quality(objs, a.topos, uvs, len(domain.tiles), settings.weights,
                                   target_k=None, groups=gmap, pairs=pairs)
    src = {o.key: o.input_uv() for o in objs}
    if all(v is not None for v in src.values()):
        plan.baseline_quality = compute_quality(objs, a.topos, src, len(domain.tiles), settings.weights)
    for obj in objs:
        topo = a.topos[obj.key]
        op = plan.objects[obj.key]
        op.candidate_seams = {e for e in range(obj.n_edges)
                              if len(topo.edge_faces[e]) == 2 and not uv_edge_continuous(topo, op.uv, e)}
        added = op.candidate_seams - set(np.nonzero(obj.seams)[0].tolist())
        removed = set(np.nonzero(obj.seams)[0].tolist()) - op.candidate_seams
        if settings.seam_mode.value == "PREFER" and (added or removed):
            plan.notes.append(f"{obj.key}: 既存シームから {len(added)} 本追加・{len(removed)} 本削除 "
                              "(歪み・統合コストの改善のため、シームモード PREFER)")
    plan.partial_summary.setdefault("generated_faces", {})
    gf: Dict[str, Set[int]] = {}
    for key, f in generated:
        gf.setdefault(key, set()).add(f)
    plan.partial_summary["generated_faces"] = gf


# ===================================================================
def validate_plan(snapshot: InputSnapshot, plan: ResultPlan, settings: Settings) -> ValidationReport:
    a = analyze(snapshot, settings)
    rep = ValidationReport()
    if plan.input_fingerprint != snapshot.fingerprint():
        rep.error(ErrorCode.STALE_INPUT, "計算後に入力が変化しています", stage="validate")
        return rep
    objs = list(snapshot.iter_objects())
    partial = settings.scope == Scope.SELECTED_FACES
    for obj in objs:
        op = plan.objects[obj.key]
        if op.uv.shape != (obj.n_loops, 2):
            rep.error(ErrorCode.VALIDATION_FAILED, "結果のコーナー数が入力と一致しません",
                      stage="validate", targets=[Target(obj.key)])
            continue
        if obj.input_uv() is not None:
            verify_unwritable_exact(obj, op, rep)
        verify_hard_constraints(obj, a.topos[obj.key], op, a.rcs[obj.key], rep)
        if partial:
            pins = obj.input_pin() & op.writable
            verify_fixed_loops(obj, op, pins, rep, "pin されたコーナー")
            fixed = plan.partial_summary.get("fixed_loops", {}).get(obj.key)
            if fixed is not None:
                verify_fixed_loops(obj, op, fixed, rep, "固定境界コーナー")
    if not rep.ok:
        return rep
    uvs = {k: p.uv for k, p in plan.objects.items()}
    wr = {k: p.writable for k, p in plan.objects.items()}
    movable = plan.partial_summary.get("movable_faces")
    gen = plan.partial_summary.get("generated_faces")
    rep.extend(validate_uv(objs, a.topos, uvs, wr, settings.effective_tiles(), settings.gap_uv,
                           settings.gap_uv, movable_faces=movable, generated_faces=gen))
    return rep
