"""全対象の原子的確定とロールバック (設計仕様 9章 / 実装指示書 8章)。

順序:
 1. Apply 時に全入力指紋・編集可能性・出力レイヤー容量・共有参照を再検証
 2. 全変更対象のバックアップを取得し、復元可能性を検証
 3. 永続 TransactionRecord を準備し、作成資源・参照変更を順に記録
 4. 全対象へ反映して事後不変検証。成功したときだけ COMMITTED
 5. 任意の例外では逆順に戻し、全対象を再検査 (ロールバック中は中断しない)
プロセスクラッシュ時の完全原子性は保証しない (保存済み .blend と永続バックアップから回収)。
"""
from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import bpy
import numpy as np

from ..blender import uv_adapter
from ..blender.mesh_adapter import capture_input, mesh_users
from ..core.errors import ErrorCode, SUOError, Target
from ..core.models import InputSnapshot, ResultPlan, TransactionRecord, TxState
from ..core.settings import Settings, SharedMeshPolicy
from . import serialization as ser
from .snapshot import backup_mesh, seams_array, topology_fingerprint, verify_backup

# 異常注入用フック: (stage, object_key) を受け取り、例外を送出できる。
FAULT_HOOK: Optional[Callable[[str, str], None]] = None


def _fault(stage: str, key: str) -> None:
    if FAULT_HOOK is not None:
        FAULT_HOOK(stage, key)


@dataclass
class RestoreReport:
    ok: bool
    state: str
    messages: List[str] = field(default_factory=list)
    residual: List[str] = field(default_factory=list)


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def record_to_dict(rec: TransactionRecord) -> dict:
    return {"tx_id": rec.tx_id, "state": rec.state.value, "created": rec.created,
            "objects": rec.objects, "entries": rec.entries, "log": rec.log, "kind": rec.kind}


def record_from_dict(d: dict) -> TransactionRecord:
    return TransactionRecord(tx_id=d["tx_id"], state=TxState(d["state"]), created=d["created"],
                             objects=list(d["objects"]), entries=dict(d["entries"]),
                             log=list(d.get("log", [])), kind=d.get("kind", "UV_RESULT"))


def persist(rec: TransactionRecord) -> None:
    ser.store(rec.tx_id, ser.dumps(record_to_dict(rec)))


def load_record(tx_id: str) -> Optional[TransactionRecord]:
    text = ser.load(tx_id)
    return record_from_dict(ser.loads(text)) if text else None


def latest_committed() -> Optional[TransactionRecord]:
    recs = []
    for tx in ser.list_ids():
        try:
            r = load_record(tx)
        except ValueError:
            continue
        if r is not None and r.state == TxState.COMMITTED:
            recs.append(r)
    recs.sort(key=lambda r: (r.created, r.tx_id))
    return recs[-1] if recs else None


# ===================================================================
def _check_before_commit(context, snapshot: InputSnapshot, plan: ResultPlan,
                         settings: Settings) -> List[str]:
    if plan.input_fingerprint != snapshot.fingerprint():
        raise SUOError(ErrorCode.STALE_INPUT, "結果と入力スナップショットが一致しません", stage="commit.precheck")
    missing = [k for k in snapshot.order if k not in bpy.data.objects]
    if missing:
        raise SUOError(ErrorCode.STALE_INPUT, "対象 Object が見つかりません", stage="commit.precheck",
                       targets=[Target(k) for k in missing], advice="再計算してください")
    fresh = capture_input(context, settings, object_names=list(snapshot.order))
    stale = [k for k in snapshot.order
             if fresh.objects[k].fingerprint() != plan.object_fingerprints.get(k)]
    if stale:
        raise SUOError(ErrorCode.STALE_INPUT,
                       "プレビュー後に入力 (トポロジー・UV・選択・変換・シーム等) が変更されました",
                       stage="commit.precheck", targets=[Target(k) for k in stale],
                       advice="再計算してから Apply してください")
    targets = plan.write_targets()
    for k in targets:
        obj = bpy.data.objects[k]
        od = fresh.objects[k]
        if not od.editable:
            raise SUOError(ErrorCode.NOT_EDITABLE, "編集できないデータです", stage="commit.precheck",
                           targets=[Target(k)])
        if len(obj.data.uv_layers) >= uv_adapter.compat.max_uv_layers():
            raise SUOError(ErrorCode.COMMIT_FAILED, "UV レイヤー数が上限のため出力レイヤーを追加できません",
                           stage="commit.precheck", targets=[Target(k)],
                           advice="不要な UV マップを削除してください")
        users = mesh_users(obj.data)
        if len(users) > 1 and settings.shared_mesh_policy == SharedMeshPolicy.STOP:
            raise SUOError(ErrorCode.SHARED_MESH, "共有 Mesh のため確定できません",
                           stage="commit.precheck", targets=[Target(k)],
                           advice="共有Mesh独立化を明示的に許可してください")
    return targets


def commit_plan(context, snapshot: InputSnapshot, plan: ResultPlan,
                settings: Settings) -> TransactionRecord:
    targets = _check_before_commit(context, snapshot, plan, settings)
    if not targets or plan.no_op:
        raise SUOError(ErrorCode.NO_OP, "書き込む変更がありません", stage="commit.precheck")
    rec = TransactionRecord(tx_id=uuid.uuid4().hex[:12], state=TxState.COMMITTING, created=_now(),
                            objects=list(targets))
    for k in targets:
        obj = bpy.data.objects[k]
        b = backup_mesh(obj)
        if not verify_backup(obj, b):
            raise SUOError(ErrorCode.COMMIT_FAILED, "バックアップを検証できません",
                           stage="commit.backup", targets=[Target(k)])
        rec.entries[k] = {"backup": b, "created_layer": None, "single_user": None,
                          "result_fingerprint": None, "post_layers": None}
    persist(rec)
    try:
        meshes = [bpy.data.objects[k].data for k in targets]
        name = uv_adapter.unique_layer_name(meshes, settings.result_prefix)
        for k in targets:
            obj = bpy.data.objects[k]
            entry = rec.entries[k]
            _fault("before_single_user", k)
            others = [u for u in mesh_users(obj.data) if u != k]
            if others:
                orig = obj.data
                copy = orig.copy()
                obj.data = copy
                entry["single_user"] = {"original_mesh": orig.name, "copy_mesh": copy.name}
                rec.log.append(f"{k}: Mesh '{orig.name}' を '{copy.name}' として独立化")
                persist(rec)
            _fault("before_layer", k)
            uv_adapter.create_layer(obj.data, name)
            entry["created_layer"] = name
            persist(rec)
            _fault("before_write", k)
            op = plan.objects[k]
            uv_adapter.write_layer(obj.data, name, op.uv, op.pin, op.select)
            _fault("after_write", k)
        # 事後不変検証
        for k in targets:
            obj = bpy.data.objects[k]
            entry = rec.entries[k]
            op = plan.objects[k]
            uv, pin, sel = uv_adapter.read_layer(obj.data, name)
            if not (np.array_equal(uv, op.uv.astype(np.float32)) and np.array_equal(pin, op.pin)
                    and np.array_equal(sel, op.select)):
                raise SUOError(ErrorCode.COMMIT_FAILED, "書込結果が計画と一致しません",
                               stage="commit.verify", targets=[Target(k)])
            for lb in entry["backup"]["layers"]:
                if uv_adapter.layer_fingerprint(obj.data, lb["name"]) != lb["fingerprint"]:
                    raise SUOError(ErrorCode.COMMIT_FAILED, f"元UV '{lb['name']}' が変化しました",
                                   stage="commit.verify", targets=[Target(k)])
            act, ren = uv_adapter.active_render_names(obj.data)
            if (act, ren) != (entry["backup"]["active"], entry["backup"]["render"]) and entry["backup"]["layers"]:
                raise SUOError(ErrorCode.COMMIT_FAILED, "アクティブ／レンダーUV設定が変化しました",
                               stage="commit.verify", targets=[Target(k)])
            if not np.array_equal(seams_array(obj.data),
                                  ser.unpack_array(entry["backup"]["seams"])):
                raise SUOError(ErrorCode.COMMIT_FAILED, "シームが変化しました", stage="commit.verify",
                               targets=[Target(k)])
            _fault("verify", k)
            entry["result_fingerprint"] = uv_adapter.layer_fingerprint(obj.data, name)
            entry["post_layers"] = {l.name: uv_adapter.layer_fingerprint(obj.data, l.name)
                                    for l in obj.data.uv_layers}
            entry["post_mesh"] = obj.data.name
            entry["post_topology"] = topology_fingerprint(obj.data)
        rec.state = TxState.COMMITTED
        rec.log.append(f"{_now()}: {len(targets)} 対象へ '{name}' を確定")
        persist(rec)
        return rec
    except BaseException as ex:  # 取消・容量不足を含む全ての異常
        rb = rollback(rec)
        if isinstance(ex, SUOError):
            msg = ex.message
            stage = ex.stage
        else:
            msg = f"{type(ex).__name__}: {ex}"
            stage = "commit"
        raise SUOError(ErrorCode.COMMIT_FAILED, f"確定に失敗しました: {msg}", stage=stage,
                       changed=not rb.ok, rollback_state=rb.state,
                       advice="ロールバック済みです" if rb.ok else
                       "残存変更があります: " + "; ".join(rb.residual)) from ex


def rollback(rec: TransactionRecord) -> RestoreReport:
    """作成物・参照変更を全対象で逆順に戻す。"""
    rec.state = TxState.ROLLING_BACK
    rep = RestoreReport(ok=True, state=TxState.ROLLING_BACK.value)
    for k in reversed(rec.objects):
        entry = rec.entries.get(k)
        if entry is None:
            continue
        try:
            _undo_entry(k, entry, rep)
        except Exception as ex:  # ロールバック自体の異常は成功と表示しない
            rep.ok = False
            rep.residual.append(f"{k}: {type(ex).__name__}: {ex}")
    for k in rec.objects:
        entry = rec.entries.get(k)
        if entry is None or k not in bpy.data.objects:
            continue
        problems = _verify_restored(bpy.data.objects[k], entry["backup"])
        if problems:
            rep.ok = False
            rep.residual.extend(f"{k}: {p}" for p in problems)
    rec.state = TxState.ROLLED_BACK if rep.ok else TxState.ROLLBACK_FAILED
    rep.state = rec.state.value
    rec.log.append(f"{_now()}: ロールバック {'完了' if rep.ok else '失敗'}")
    try:
        if rep.ok:
            ser.delete(rec.tx_id)   # 未確定の記録は残さない
        else:
            persist(rec)
    except Exception as ex:
        rep.residual.append(f"復元記録の更新に失敗: {ex}")
    return rep


def _undo_entry(k: str, entry: dict, rep: RestoreReport) -> None:
    obj = bpy.data.objects.get(k)
    if obj is None:
        rep.residual.append(f"{k}: Object が存在しません")
        rep.ok = False
        return
    name = entry.get("created_layer")
    su = entry.get("single_user")
    if su:
        orig = bpy.data.meshes.get(su["original_mesh"])
        copy = bpy.data.meshes.get(su["copy_mesh"])
        if orig is None:
            raise RuntimeError(f"元の Mesh '{su['original_mesh']}' が見つかりません")
        obj.data = orig
        if copy is not None and copy.users == 0:
            bpy.data.meshes.remove(copy)
    elif name:
        uv_adapter.remove_layer(obj.data, name)
    b = entry["backup"]
    uv_adapter.set_active_render(obj.data, b["active"], b["render"])
    entry["created_layer"] = None


def _verify_restored(obj, backup: dict) -> List[str]:
    problems = []
    me = obj.data
    if me.name != backup["mesh"]:
        problems.append(f"Mesh 参照が元と異なります ({me.name} != {backup['mesh']})")
        return problems
    names = [l.name for l in me.uv_layers]
    if names != [l["name"] for l in backup["layers"]]:
        problems.append(f"UV レイヤー構成が元と異なります {names}")
        return problems
    for lb in backup["layers"]:
        if uv_adapter.layer_fingerprint(me, lb["name"]) != lb["fingerprint"]:
            problems.append(f"UV '{lb['name']}' が元と異なります")
    act, ren = uv_adapter.active_render_names(me)
    if backup["layers"] and (act, ren) != (backup["active"], backup["render"]):
        problems.append("アクティブ／レンダーUV設定が元と異なります")
    if not np.array_equal(seams_array(me), ser.unpack_array(backup["seams"])):
        problems.append("シームが元と異なります")
    return problems


# ===================================================================
def commit_seams(context, plan: ResultPlan, keys: List[str]) -> TransactionRecord:
    """候補シームを実シームへ反映する明示操作。Restore の対象。"""
    rec = TransactionRecord(tx_id=uuid.uuid4().hex[:12], state=TxState.COMMITTING, created=_now(),
                            objects=list(keys), kind="SEAMS")
    for k in keys:
        obj = bpy.data.objects[k]
        rec.entries[k] = {"backup": backup_mesh(obj), "created_layer": None, "single_user": None}
    persist(rec)
    try:
        for k in keys:
            obj = bpy.data.objects[k]
            me = obj.data
            s = np.zeros(len(me.edges), dtype=bool)
            for e in plan.objects[k].candidate_seams:
                s[e] = True
            _fault("seams", k)
            me.edges.foreach_set("use_seam", s)
            me.update()
            rec.entries[k]["post_seams"] = ser.pack_array(seams_array(me))
            rec.entries[k]["post_mesh"] = me.name
            rec.entries[k]["post_topology"] = topology_fingerprint(me)
        rec.state = TxState.COMMITTED
        persist(rec)
        return rec
    except BaseException as ex:
        for k in reversed(keys):
            obj = bpy.data.objects.get(k)
            if obj is not None:
                obj.data.edges.foreach_set("use_seam", ser.unpack_array(rec.entries[k]["backup"]["seams"]))
        rec.state = TxState.ROLLED_BACK
        ser.delete(rec.tx_id)
        raise SUOError(ErrorCode.COMMIT_FAILED, f"シーム反映に失敗しました: {ex}", stage="commit.seams",
                       rollback_state=rec.state.value) from ex
