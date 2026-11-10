"""マテリアル境界と Base Color の OKLab 色差。"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from ..core.models import ObjectData
from .topology import Topology


def linear_to_oklab(rgb: Tuple[float, float, float]) -> Tuple[float, float, float]:
    r, g, b = rgb
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = (math.copysign(abs(x) ** (1 / 3), x) for x in (l, m, s))
    return (0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_)


def oklab_delta(c1: Optional[Tuple[float, float, float]],
                c2: Optional[Tuple[float, float, float]]) -> float:
    """色が不明なら nan。"""
    if c1 is None or c2 is None:
        return float("nan")
    a, b = linear_to_oklab(c1), linear_to_oklab(c2)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def material_boundaries(topo: Topology):
    """返り値: (is_boundary[E] bool, color_delta[E])。"""
    obj: ObjectData = topo.obj
    E = obj.n_edges
    boundary = np.zeros(E, dtype=bool)
    delta = np.full(E, np.nan)
    for e, fs in enumerate(topo.edge_faces):
        if len(fs) != 2:
            continue
        m1, m2 = int(obj.face_material[fs[0]]), int(obj.face_material[fs[1]])
        if m1 == m2:
            delta[e] = 0.0
            continue
        boundary[e] = True
        c1 = obj.materials[m1].base_color if 0 <= m1 < len(obj.materials) else None
        c2 = obj.materials[m2].base_color if 0 <= m2 < len(obj.materials) else None
        delta[e] = oklab_delta(c1, c2)
    return boundary, delta


def material_name(obj: ObjectData, index: int) -> str:
    if 0 <= index < len(obj.materials):
        return obj.materials[index].name
    return f"<slot {index}>"
