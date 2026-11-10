"""Restore: 直近の確定を全対象まとめて取り消す。

トポロジーと確定後データの指紋を全対象で先に検証し、ユーザーが後から
変更した結果を無断で上書きしない。競合時は何も変更せず RESTORE_CONFLICT。
"""
from __future__ import annotations

from typing import List, Optional

import bpy
import numpy as np

from ..blender import uv_adapter
from ..core.errors import ErrorCode, SUOError, Target
from ..core.models import TransactionRecord, TxState
from . import serialization as ser
from .snapshot import seams_array, topology_fingerprint
from .transaction import RestoreReport, _now, _undo_entry, _verify_restored, latest_committed, persist


def find_conflicts(rec: TransactionRecord) -> List[str]:
    problems: List[str] = []
    for k in rec.objects:
        entry = rec.entries[k]
        obj = bpy.data.objects.get(k)
        if obj is None:
            problems.append(f"{k}: Object が存在しません")
            continue
        me = obj.data
        if me.name != entry.get("post_mesh"):
            problems.append(f"{k}: Mesh 参照が確定後から変更されています")
            continue
        if topology_fingerprint(me) != entry.get("post_topology"):
            problems.append(f"{k}: トポロジーが確定後に変更されています")
            continue
        if rec.kind == "SEAMS":
            if not np.array_equal(seams_array(me), ser.unpack_array(entry["post_seams"])):
                problems.append(f"{k}: シームが確定後に変更されています")
            continue
        name = entry.get("created_layer")
        if not name or name not in me.uv_layers:
            problems.append(f"{k}: 結果レイヤーが見つかりません")
            continue
        if uv_adapter.layer_fingerprint(me, name) != entry.get("result_fingerprint"):
            problems.append(f"{k}: 結果UV '{name}' が確定後に手動で変更されています")
        post = entry.get("post_layers") or {}
        if sorted(l.name for l in me.uv_layers) != sorted(post):
            problems.append(f"{k}: UV レイヤー構成が確定後に変更されています")
        else:
            for lname, fp in post.items():
                if lname != name and uv_adapter.layer_fingerprint(me, lname) != fp:
                    problems.append(f"{k}: UV '{lname}' が確定後に変更されています")
        su = entry.get("single_user")
        if su:
            orig = bpy.data.meshes.get(su["original_mesh"])
            if orig is None:
                problems.append(f"{k}: 元の共有 Mesh '{su['original_mesh']}' が見つかりません")
            elif topology_fingerprint(orig) != entry["backup"]["topology"]:
                problems.append(f"{k}: 元の共有 Mesh が確定後に変更されています")
    return problems


def restore(rec: Optional[TransactionRecord] = None) -> RestoreReport:
    rec = rec or latest_committed()
    if rec is None:
        raise SUOError(ErrorCode.RESTORE_CONFLICT, "復元できる確定記録がありません", stage="restore")
    if rec.state != TxState.COMMITTED:
        raise SUOError(ErrorCode.RESTORE_CONFLICT, f"確定済みではない記録です ({rec.state.value})",
                       stage="restore")
    conflicts = find_conflicts(rec)
    if conflicts:
        raise SUOError(ErrorCode.RESTORE_CONFLICT, "Restore を拒否しました: " + "; ".join(conflicts[:5]),
                       stage="restore", targets=[Target(c.split(":")[0]) for c in conflicts[:10]],
                       advice="「バックアップを別コピーへ回収」で元のUVを取り出せます")
    rep = RestoreReport(ok=True, state="RESTORING")
    for k in reversed(rec.objects):
        entry = rec.entries[k]
        try:
            if rec.kind == "SEAMS":
                bpy.data.objects[k].data.edges.foreach_set(
                    "use_seam", ser.unpack_array(entry["backup"]["seams"]))
                bpy.data.objects[k].data.update()
            else:
                _undo_entry(k, entry, rep)
        except Exception as ex:
            rep.ok = False
            rep.residual.append(f"{k}: {type(ex).__name__}: {ex}")
    for k in rec.objects:
        problems = _verify_restored(bpy.data.objects[k], rec.entries[k]["backup"])
        if problems:
            rep.ok = False
            rep.residual.extend(f"{k}: {p}" for p in problems)
    rec.state = TxState.RESTORED if rep.ok else TxState.ROLLBACK_FAILED
    rep.state = rec.state.value
    rec.log.append(f"{_now()}: Restore {'完了' if rep.ok else '失敗'}")
    persist(rec)
    return rep


def recover_to_copy(rec: TransactionRecord, key: str) -> str:
    """保存済みバックアップの UV を、元 Object の複製 (別コピー) に復元する。"""
    entry = rec.entries[key]
    b = entry["backup"]
    src = bpy.data.objects.get(key)
    if src is None:
        raise SUOError(ErrorCode.RESTORE_CONFLICT, "元 Object がありません", stage="recover")
    new_obj = src.copy()
    new_obj.data = src.data.copy()
    new_obj.name = f"{key}_SUO_recovered"
    for coll in src.users_collection:
        coll.objects.link(new_obj)
    me = new_obj.data
    if topology_fingerprint(me) != b["topology"]:
        bpy.data.objects.remove(new_obj)
        raise SUOError(ErrorCode.RESTORE_CONFLICT, "トポロジーが変更されているため回収できません",
                       stage="recover", targets=[Target(key)])
    while len(me.uv_layers):
        me.uv_layers.remove(me.uv_layers[0])
    for lb in b["layers"]:
        layer = me.uv_layers.new(name=lb["name"], do_init=False)
        uv_adapter.write_layer(me, layer.name, ser.unpack_array(lb["uv"]).astype(np.float64),
                               ser.unpack_array(lb["pin"]), ser.unpack_array(lb["select"]))
    uv_adapter.set_active_render(me, b["active"], b["render"])
    me.edges.foreach_set("use_seam", ser.unpack_array(b["seams"]))
    return new_obj.name
