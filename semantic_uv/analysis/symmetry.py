"""ローカル X/Y/Z 鏡映による対称部品の検出。

初期版は Object ローカル座標の軸平面に対する鏡映のみを扱う。
位置 (頂点集合) と面構造 (面数・頂点数) の両方を確認し、
信頼度が閾値未満なら対称関係を適用しない。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.models import ObjectData

CONFIDENCE_THRESHOLD = 0.95


def _local_points(obj: ObjectData, faces: Sequence[int]) -> np.ndarray:
    verts = sorted(set(int(v) for f in faces for v in obj.loop_vert[list(obj.face_loops(f))]))
    return obj.co[verts]


def mirror_confidence(obj: ObjectData, faces_a: Sequence[int], faces_b: Sequence[int],
                      axis: int, tol: float) -> float:
    """faces_a を axis で鏡映した頂点が faces_b の頂点に一致する割合。"""
    pa = _local_points(obj, faces_a)
    pb = _local_points(obj, faces_b)
    if len(pa) == 0 or len(pa) != len(pb) or len(faces_a) != len(faces_b):
        return 0.0
    m = pa.copy()
    m[:, axis] = -m[:, axis]
    # 量子化ハッシュで最近傍照合 (近傍セルも確認)
    cell = max(tol, 1e-9)
    grid: Dict[Tuple[int, int, int], List[int]] = {}
    for i, p in enumerate(pb):
        grid.setdefault(tuple(np.floor(p / cell).astype(int)), []).append(i)
    matched = 0
    used = set()
    for p in m:
        base = np.floor(p / cell).astype(int)
        hit = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in grid.get((base[0] + dx, base[1] + dy, base[2] + dz), ()):
                        if j not in used and np.linalg.norm(pb[j] - p) <= tol:
                            used.add(j)
                            hit = True
                            break
                    if hit:
                        break
                if hit:
                    break
        matched += hit
    return matched / len(pa)


def find_mirror_pairs(obj: ObjectData, parts: List[Sequence[int]],
                      rel_tol: float = 1e-4) -> List[Tuple[int, int, int, float]]:
    """parts (面集合のリスト) の中から鏡映ペアを返す: (i, j, axis, confidence)。
    i < j、各 part は高々1ペア。"""
    if not parts:
        return []
    co = obj.co
    diag = float(np.linalg.norm(co.max(axis=0) - co.min(axis=0))) if len(co) else 1.0
    tol = max(diag * rel_tol, 1e-9)
    centroids = [(_local_points(obj, p).mean(axis=0) if len(p) else np.zeros(3)) for p in parts]
    sizes = [len(p) for p in parts]
    paired = set()
    out = []
    for i in range(len(parts)):
        if i in paired:
            continue
        best: Optional[Tuple[int, int, float]] = None
        for j in range(i + 1, len(parts)):
            if j in paired or sizes[i] != sizes[j]:
                continue
            for axis in range(3):
                mc = centroids[i].copy()
                mc[axis] = -mc[axis]
                if np.linalg.norm(mc - centroids[j]) > tol * 10 + diag * 1e-3:
                    continue
                # 鏡映面上にある自己対称部品は自分自身と対になるので除外
                if abs(centroids[i][axis]) <= tol:
                    continue
                conf = mirror_confidence(obj, parts[i], parts[j], axis, tol)
                if conf >= CONFIDENCE_THRESHOLD and (best is None or conf > best[2]):
                    best = (j, axis, conf)
        if best is not None:
            j, axis, conf = best
            paired.update((i, j))
            out.append((i, j, axis, conf))
    return out
