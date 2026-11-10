"""高速な島単位の重複判定を、総当たりの交差面積判定と突き合わせる。"""
import math
import random

import numpy as np

import helpers  # noqa: F401  (sys.path)
from semantic_uv.validation.injectivity import island_self_overlap, overlapping_island_pairs
from semantic_uv.validation.overlaps import find_overlaps


def grid_island(n, x0, y0, s, jitter=0.0, rng=None):
    pts = {}
    for j in range(n + 1):
        for i in range(n + 1):
            dx = dy = 0.0
            if rng is not None and 0 < i < n and 0 < j < n:
                dx, dy = rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)
            pts[(i, j)] = (x0 + s * i / n + dx, y0 + s * j / n + dy)
    tris = []
    for j in range(n):
        for i in range(n):
            a, b, c, d = pts[(i, j)], pts[(i + 1, j)], pts[(i + 1, j + 1)], pts[(i, j + 1)]
            tris.append((a, b, c))
            tris.append((a, c, d))
    return np.asarray(tris, dtype=np.float64)


def test_self_overlap_matches_bruteforce():
    rng = random.Random(1)
    agree = 0
    for trial in range(60):
        t = grid_island(4, 0, 0, 1.0, jitter=rng.uniform(0, 0.3), rng=rng)
        sa = 0.5 * ((t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1]) - (t[:, 1, 1] - t[:, 0, 1]) * (t[:, 2, 0] - t[:, 0, 0]))
        if (sa <= 0).any():
            continue  # 向きの揃った島だけが前提
        brute = bool(find_overlaps(t, max_pairs=1))
        assert island_self_overlap(t) == brute
        agree += 1
    assert agree > 10


def test_folded_strip_detected():
    # 帯を折り返して自分自身に重ねる (全三角形は正の向き、境界が交差)
    t = grid_island(1, 0, 0, 1.0)
    pts = []
    for k in range(6):
        ang = k * 1.3
        x, y = math.cos(ang), math.sin(ang)
        x2, y2 = 2 * math.cos(ang), 2 * math.sin(ang)
        pts.append(((x, y), (x2, y2)))
    tris = []
    for k in range(5):
        (a, b), (c, d) = pts[k], pts[k + 1]
        tris.append((a, b, d))
        tris.append((a, d, c))
    tris = np.asarray(tris)
    assert island_self_overlap(tris) == bool(find_overlaps(tris, max_pairs=1)) == True


def test_island_pairs_match_bruteforce():
    rng = random.Random(7)
    for trial in range(40):
        isl = [grid_island(2, rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0.1, 0.6)) for _ in range(4)]
        fast = set(overlapping_island_pairs(isl))
        brute = set()
        for i in range(4):
            for j in range(i + 1, 4):
                if find_overlaps(np.concatenate([isl[i], isl[j]]), max_pairs=10_000):
                    # 同一島内の重複はないので、見つかれば島間
                    brute.add((i, j))
        assert fast == brute, (trial, fast, brute)


def test_touching_and_nested():
    a = grid_island(2, 0, 0, 1.0)
    b = grid_island(2, 1.0, 0, 1.0)      # 辺で接する
    c = grid_island(2, 0.25, 0.25, 0.5)  # a に内包
    assert overlapping_island_pairs([a, b]) == []
    assert overlapping_island_pairs([a, c]) == [(0, 1)]
    assert overlapping_island_pairs([a, a.copy()]) == [(0, 1)]   # 完全スタック
