"""オペレーター。最適化ロジックは持たず、パイプライン・トランザクションを呼ぶだけ。"""
from __future__ import annotations

from typing import Optional

import bmesh
import bpy
from bpy.props import EnumProperty

from ..blender.context import ContextState, object_mode
from ..blender.mesh_adapter import CONSTRAINT_ATTR, capture_input
from ..core import pipeline as P
from ..core.errors import ErrorCode, SUOError
from ..core.models import EdgeConstraint, TxState
from ..persistence import restore as R
from ..persistence import transaction as T
from . import properties


class Session:
    """プレビュー中の作業コピー (メモリ内のみ)。元データには影響しない。"""
    snapshot = None
    settings = None
    plan = None
    state: TxState = TxState.IDLE
    messages: list = []
    running = False
    cancel_requested = False

    @classmethod
    def clear(cls):
        cls.snapshot = cls.settings = cls.plan = None
        cls.state = TxState.IDLE

    @classmethod
    def set_messages(cls, lines):
        cls.messages = list(lines)[:40]


def _report_error(op, err: SUOError):
    op.report({"ERROR"}, err.describe())
    lines = [err.describe()]
    if err.report is not None:
        lines += [i.describe() for i in err.report.issues]
    Session.set_messages(lines)


def _redraw(context):
    for win in context.window_manager.windows:
        for area in win.screen.areas:
            if area.type in {"IMAGE_EDITOR", "VIEW_3D"}:
                area.tag_redraw()


def _prepare(context):
    settings = properties.to_settings(context)
    snapshot = capture_input(context, settings)
    rep = P.validate_input(snapshot, settings)
    rep.raise_if_failed("input")
    scope = P.build_scope(snapshot, settings)
    domain = P.build_domain(snapshot, scope, settings)
    return settings, snapshot, scope, domain, rep


class SUO_OT_analyze(bpy.types.Operator):
    bl_idname = "suo.analyze"
    bl_label = "Analyze"
    bl_description = "対象・入力UV・制約・トポロジーを検査します (データは変更しません)"

    def execute(self, context):
        try:
            with object_mode(context):
                settings, snapshot, scope, domain, rep = _prepare(context)
        except SUOError as e:
            _report_error(self, e)
            return {"CANCELLED"}
        lines = [f"対象 {len(snapshot.order)} Object"]
        for obj in snapshot.iter_objects():
            lines.append(f"  {obj.key}: 面 {obj.n_faces}, UV入力 {obj.input_uv_name or 'なし'}")
        if hasattr(scope, "selected_face_count"):
            lines.append(f"選択面 {scope.selected_face_count} / 固定障害物三角形 {len(domain.obstacles.tris)}")
        lines += [i.describe() for i in rep.issues]
        Session.set_messages(lines)
        self.report({"INFO"}, "解析完了: " + lines[0])
        return {"FINISHED"}


class SUO_OT_generate(bpy.types.Operator):
    bl_idname = "suo.generate"
    bl_label = "Generate"
    bl_description = "作業コピー上でUVを計算し、検証済みの結果をプレビューします (Escで取消)"

    _timer = None
    _gen = None
    _ctx: Optional[ContextState] = None

    def _start(self, context):
        self._ctx = ContextState(context)
        if context.mode != "OBJECT":
            for name in self._ctx.in_edit:
                ob = bpy.data.objects.get(name)
                if ob is not None:
                    ob.update_from_editmode()
            bpy.ops.object.mode_set(mode="OBJECT")
        settings, snapshot, scope, domain, rep = _prepare(context)
        Session.clear()
        Session.state = TxState.COMPUTING
        Session.cancel_requested = False
        self._settings, self._snapshot, self._scope = settings, snapshot, scope
        self._gen = P.compute_plan_iter(snapshot, scope, domain, settings)

    def _finish(self, context, plan):
        vr = P.validate_plan(self._snapshot, plan, self._settings)
        if not vr.ok:
            Session.clear()
            raise SUOError(vr.errors[0].code, "結果が検証に合格しません: " + vr.errors[0].message,
                           stage="validate", targets=vr.errors[0].targets, report=vr)
        Session.snapshot, Session.settings, Session.plan = self._snapshot, self._settings, plan
        Session.state = TxState.PREVIEW
        lines = ["プレビュー: 検証合格 (制約違反 0)"]
        lines += [f"出力先: 新規UVレイヤー ({self._settings.result_prefix}NNN) / 対象: "
                  + ", ".join(plan.write_targets())]
        lines += [f"{a}: {b}" for a, b in plan.quality.as_rows()]
        ps = plan.partial_summary
        if "selected_faces" in ps:
            lines.append(f"選択面 {sum(ps['selected_faces'].values())} / 固定面 {sum(ps['fixed_faces'].values())} "
                         f"/ 境界エッジ {sum(ps['boundary_edges'].values())} / 障害物三角形 {ps.get('obstacle_tris', 0)}")
            if "density_achieved" in ps:
                lines.append(f"密度達成: {'可' if ps['density_achieved'] else '不可'} "
                             f"(最大偏差 {ps['density_deviation_max'] * 100:.1f}%)")
        lines += ["注記: " + n for n in plan.notes] + ["警告: " + w for w in plan.warnings]
        lines += [i.describe() for i in vr.warnings]
        Session.set_messages(lines)

    def _end(self, context):
        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        try:
            wm.progress_end()
        except Exception:
            pass
        Session.running = False
        if self._ctx is not None:
            self._ctx.restore(context)
        _redraw(context)

    def execute(self, context):
        """非対話実行 (スクリプト・テスト用)。"""
        try:
            self._start(context)
            while True:
                try:
                    next(self._gen)
                except StopIteration as stop:
                    self._finish(context, stop.value)
                    break
        except SUOError as e:
            Session.clear()
            _report_error(self, e)
            if self._ctx:
                self._ctx.restore(context)
            return {"CANCELLED"}
        if self._ctx:
            self._ctx.restore(context)
        self.report({"INFO"}, "プレビューを作成しました")
        return {"FINISHED"}

    def invoke(self, context, event):
        try:
            self._start(context)
        except SUOError as e:
            Session.clear()
            _report_error(self, e)
            if self._ctx:
                self._ctx.restore(context)
            return {"CANCELLED"}
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.01, window=context.window)
        wm.progress_begin(0, 100)
        Session.running = True
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "ESC" or Session.cancel_requested:
            self._gen.close()
            Session.clear()
            Session.set_messages(["計算を取り消しました (元データは変更されていません)"])
            self._end(context)
            self.report({"WARNING"}, "取消しました")
            return {"CANCELLED"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        try:
            frac, msg = next(self._gen)
            context.window_manager.progress_update(int(frac * 100))
            Session.set_messages([f"計算中 {int(frac * 100)}%: {msg}"])
            _redraw(context)
            return {"RUNNING_MODAL"}
        except StopIteration as stop:
            try:
                self._finish(context, stop.value)
            except SUOError as e:
                _report_error(self, e)
                self._end(context)
                return {"CANCELLED"}
            self._end(context)
            self.report({"INFO"}, "プレビューを作成しました")
            return {"FINISHED"}
        except SUOError as e:
            Session.clear()
            _report_error(self, e)
            self._end(context)
            return {"CANCELLED"}
        except MemoryError:
            Session.clear()
            _report_error(self, SUOError(ErrorCode.OUT_OF_MEMORY, "メモリ不足のため中止しました",
                                         stage="compute", advice="対象を減らしてください"))
            self._end(context)
            return {"CANCELLED"}


class SUO_OT_compare(bpy.types.Operator):
    bl_idname = "suo.compare"
    bl_label = "Compare"
    bl_description = "入力UVとプレビュー結果の指標を比較表示します"

    @classmethod
    def poll(cls, context):
        return Session.plan is not None

    def execute(self, context):
        plan = Session.plan
        lines = ["指標: 入力UV → 結果"]
        base = plan.baseline_quality
        for (name, new), old in zip(plan.quality.as_rows(),
                                    base.as_rows() if base else [(None, "-")] * 20):
            lines.append(f"  {name}: {old[1]} → {new}")
        lines.append("※グループ分散・方向一致率は可読性の代理指標です")
        Session.set_messages(lines)
        return {"FINISHED"}


class SUO_OT_apply(bpy.types.Operator):
    bl_idname = "suo.apply"
    bl_label = "Apply"
    bl_description = "プレビュー結果を新規UVレイヤーへ全対象まとめて確定します (元UVは変更しません)"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return Session.plan is not None and Session.state == TxState.PREVIEW

    def execute(self, context):
        try:
            with object_mode(context):
                Session.state = TxState.COMMITTING
                rec = T.commit_plan(context, Session.snapshot, Session.plan, Session.settings)
        except SUOError as e:
            Session.state = TxState.PREVIEW if e.code == ErrorCode.COMMIT_FAILED else Session.state
            if e.code == ErrorCode.STALE_INPUT:
                Session.clear()
            _report_error(self, e)
            return {"CANCELLED"}
        layer = next(iter(rec.entries.values()))["created_layer"]
        Session.set_messages([f"確定しました: {layer} ({len(rec.objects)} 対象, 記録 {rec.tx_id})",
                              "既存のアクティブ／レンダーUVは変更していません"])
        Session.clear()
        Session.state = TxState.COMMITTED
        _redraw(context)
        return {"FINISHED"}


class SUO_OT_cancel(bpy.types.Operator):
    bl_idname = "suo.cancel"
    bl_label = "Cancel"
    bl_description = "計算中なら取消し、プレビューを破棄します"

    def execute(self, context):
        if Session.running:
            Session.cancel_requested = True
        Session.clear()
        Session.set_messages(["プレビューを破棄しました (元データは変更されていません)"])
        _redraw(context)
        return {"FINISHED"}


class SUO_OT_restore(bpy.types.Operator):
    bl_idname = "suo.restore"
    bl_label = "Restore"
    bl_description = "直近の本アドオンの確定を全対象まとめて取り消します"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            with object_mode(context):
                rep = R.restore()
        except SUOError as e:
            _report_error(self, e)
            return {"CANCELLED"}
        if rep.ok:
            Session.set_messages(["Restore 完了"])
            return {"FINISHED"}
        Session.set_messages(["Restore に失敗しました。残存変更:"] + rep.residual)
        self.report({"ERROR"}, "Restore に失敗しました")
        return {"CANCELLED"}


class SUO_OT_recover_backup(bpy.types.Operator):
    bl_idname = "suo.recover_backup"
    bl_label = "バックアップを別コピーへ回収"
    bl_description = "直近の確定前バックアップを、Objectの複製に復元します (元Objectは変更しません)"

    def execute(self, context):
        rec = T.latest_committed()
        if rec is None:
            self.report({"ERROR"}, "回収できる記録がありません")
            return {"CANCELLED"}
        names = []
        try:
            with object_mode(context):
                for k in rec.objects:
                    names.append(R.recover_to_copy(rec, k))
        except SUOError as e:
            _report_error(self, e)
            return {"CANCELLED"}
        Session.set_messages(["回収しました: " + ", ".join(names)])
        return {"FINISHED"}


class SUO_OT_apply_seams(bpy.types.Operator):
    bl_idname = "suo.apply_seams"
    bl_label = "候補シームを実シームへ反映"
    bl_description = "プレビュー結果の候補シームを元メッシュのシームフラグへ反映します (Restore 対象)"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return Session.plan is not None and Session.state == TxState.PREVIEW

    def execute(self, context):
        plan = Session.plan
        try:
            with object_mode(context):
                fresh = capture_input(context, Session.settings, list(Session.snapshot.order))
                if fresh.fingerprint() != plan.input_fingerprint:
                    raise SUOError(ErrorCode.STALE_INPUT, "入力が変更されています", stage="seams",
                                   advice="再計算してください")
                rec = T.commit_seams(context, plan, plan.write_targets())
        except SUOError as e:
            _report_error(self, e)
            return {"CANCELLED"}
        Session.clear()
        Session.set_messages([f"シームを反映しました (記録 {rec.tx_id})。結果は再計算してください"])
        return {"FINISHED"}


class SUO_OT_set_edge_constraint(bpy.types.Operator):
    bl_idname = "suo.set_edge_constraint"
    bl_label = "エッジ制約を設定"
    bl_description = "Edit Mode で選択したエッジに手動制約を設定します"
    bl_options = {"REGISTER", "UNDO"}

    constraint: EnumProperty(name="制約", items=[
        ("AUTO", "AUTO", ""), ("PREFER", "PREFER", "シームにしたい"),
        ("REQUIRE", "REQUIRE", "必ずシーム (ハード制約)"), ("AVOID", "AVOID", "シームを避けたい"),
        ("FORBID", "FORBID", "シーム禁止 (ハード制約)")])

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH"

    def execute(self, context):
        value = int(EdgeConstraint[self.constraint])
        count = 0
        for ob in context.objects_in_mode:
            bm = bmesh.from_edit_mesh(ob.data)
            layer = bm.edges.layers.int.get(CONSTRAINT_ATTR) or bm.edges.layers.int.new(CONSTRAINT_ATTR)
            for e in bm.edges:
                if e.select:
                    e[layer] = value
                    count += 1
            bmesh.update_edit_mesh(ob.data, loop_triangles=False, destructive=False)
        self.report({"INFO"}, f"{count} 本のエッジに {self.constraint} を設定しました")
        return {"FINISHED"}


classes = (SUO_OT_analyze, SUO_OT_generate, SUO_OT_compare, SUO_OT_apply, SUO_OT_cancel,
           SUO_OT_restore, SUO_OT_recover_backup, SUO_OT_apply_seams, SUO_OT_set_edge_constraint)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
