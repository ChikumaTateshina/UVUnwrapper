"""バージョン付き形式での永続化。

復元情報は .blend に保存される Text データブロック (フェイクユーザー付き) に
JSON として格納し、明示削除まで残す。大きな配列は zlib + base64。
"""
from __future__ import annotations

import base64
import json
import zlib
from typing import List, Optional

import numpy as np

FORMAT_VERSION = 1
TEXT_PREFIX = ".SUO_TX_"


def pack_array(arr: np.ndarray) -> dict:
    a = np.ascontiguousarray(arr)
    return {"dtype": str(a.dtype), "shape": list(a.shape),
            "data": base64.b64encode(zlib.compress(a.tobytes(), 6)).decode("ascii")}


def unpack_array(d: dict) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(d["data"]))
    return np.frombuffer(raw, dtype=np.dtype(d["dtype"])).reshape(d["shape"]).copy()


def dumps(record_dict: dict) -> str:
    payload = dict(record_dict)
    payload["format_version"] = FORMAT_VERSION
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def loads(text: str) -> dict:
    d = json.loads(text)
    v = d.get("format_version")
    if v != FORMAT_VERSION:
        raise ValueError(f"未対応の復元データ形式です (version={v})")
    return d


# ------------------------------------------------ Blender Text 格納
def store(tx_id: str, text: str) -> str:
    import bpy
    name = TEXT_PREFIX + tx_id
    t = bpy.data.texts.get(name) or bpy.data.texts.new(name)
    t.clear()
    t.write(text)
    t.use_fake_user = True
    return name


def load(tx_id: str) -> Optional[str]:
    import bpy
    t = bpy.data.texts.get(TEXT_PREFIX + tx_id)
    return t.as_string() if t is not None else None


def delete(tx_id: str) -> None:
    import bpy
    t = bpy.data.texts.get(TEXT_PREFIX + tx_id)
    if t is not None:
        bpy.data.texts.remove(t)


def list_ids() -> List[str]:
    import bpy
    return sorted(t.name[len(TEXT_PREFIX):] for t in bpy.data.texts if t.name.startswith(TEXT_PREFIX))
