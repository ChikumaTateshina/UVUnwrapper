"""形状類似性。面積・周長・主軸長・面数・曲がりで候補を絞る。"""
from __future__ import annotations

from typing import List

import numpy as np

SIMILAR_TOL = 0.05


def descriptor_distance(a: np.ndarray, b: np.ndarray) -> float:
    """相対差の最大値 (0 = 同一)。面数は一致を要求。"""
    if a is None or b is None:
        return float("inf")
    if int(a[5]) != int(b[5]):
        return float("inf")
    d = 0.0
    for i in (0, 1, 2, 3, 4):
        m = max(abs(a[i]), abs(b[i]))
        if m <= 1e-12:
            continue
        d = max(d, abs(a[i] - b[i]) / m)
    d = max(d, abs(a[6] - b[6]))
    return d


def similar_clusters(descriptors: List[np.ndarray], tol: float = SIMILAR_TOL) -> List[int]:
    """貪欲クラスタリング。返り値: 各要素のクラスタ番号 (最初の代表の index)。"""
    labels = [-1] * len(descriptors)
    reps: List[int] = []
    for i, d in enumerate(descriptors):
        for r in reps:
            if descriptor_distance(d, descriptors[r]) <= tol:
                labels[i] = r
                break
        if labels[i] < 0:
            labels[i] = i
            reps.append(i)
    return labels
