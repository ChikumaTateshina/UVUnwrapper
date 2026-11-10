"""共通UV空間へのパッキング。

pack_full: 全体配置。共通の長さ倍率 s を二分探索し、全島を
  スカイライン左下詰めで許可タイルへ配置する。外接矩形は島間余白込みで
  重ならないため、島同士の余白は保証される。島はタイル境界をまたがない。
place_free: 固定障害物のある空き領域への配置 (RELOCATE_SELECTED)。
  保守的占有グリッドの相関 (FFT) で輪郭ベースの候補を求め、
  三角形の厳密な距離判定で確定する。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .domain import tile_origin
from .obstacles import ObstacleIndex, rasterize


@dataclass
class PackItem:
    """原点基準の島形状 (長さ倍率 1 = A_uv が A_world と等しい状態)。"""
    index: int
    uv: np.ndarray            # (n,2) 回転・正規化済み
    tris: np.ndarray          # (T,3) uv 内 index
    allow_rotate: bool = True

    def bbox(self) -> Tuple[float, float]:
        e = np.ptp(self.uv, axis=0)
        return float(e[0]), float(e[1])


@dataclass
class Placement:
    index: int
    tile: int
    rot90: int                 # 0..3
    scale: float
    offset: Tuple[float, float]  # 絶対UV上の bbox 最小点

    def apply(self, uv: np.ndarray) -> np.ndarray:
        q = rotate90(uv, self.rot90) * self.scale
        q = q - q.min(axis=0)
        return q + np.asarray(self.offset)


def rotate90(uv: np.ndarray, k: int) -> np.ndarray:
    k %= 4
    if k == 0:
        return uv.copy()
    if k == 1:
        return np.c_[-uv[:, 1], uv[:, 0]]
    if k == 2:
        return -uv
    return np.c_[uv[:, 1], -uv[:, 0]]


# ------------------------------------------------------------- skyline
class _Skyline:
    def __init__(self, margin: float):
        self.m = margin
        self.segs: List[List[float]] = [[margin, margin, 1.0 - 2 * margin]]  # x, y, w

    def find(self, w: float, h: float, gap: float) -> Optional[Tuple[float, float, int]]:
        m = self.m
        best = None
        for i in range(len(self.segs)):
            x = self.segs[i][0]
            if x + w > 1 - m + 1e-12:
                break
            need = w + gap
            y = 0.0
            j = i
            covered = 0.0
            while j < len(self.segs) and covered < need - 1e-15:
                y = max(y, self.segs[j][1])
                covered = self.segs[j][0] + self.segs[j][2] - x
                j += 1
            if covered < w - 1e-12:
                continue
            if y + h > 1 - m + 1e-12:
                continue
            cand = (y + h, x, i)
            if best is None or cand < best:
                best = cand
        if best is None:
            return None
        top, x, i = best
        return x, top - h, i

    def place(self, x: float, y: float, w: float, h: float, gap: float) -> None:
        new = [x, y + h + gap, w + gap]
        right = x + w + gap
        out: List[List[float]] = []
        for s in self.segs:
            sx, sy, sw = s
            ex = sx + sw
            if ex <= x or sx >= right:
                out.append(s)
                continue
            if sx < x:
                out.append([sx, sy, x - sx])
            if ex > right:
                out.append([right, sy, ex - right])
        out.append(new)
        out.sort(key=lambda s: s[0])
        merged: List[List[float]] = []
        for s in out:
            if s[2] <= 1e-15:
                continue
            if merged and abs(merged[-1][1] - s[1]) < 1e-15 and \
                    abs(merged[-1][0] + merged[-1][2] - s[0]) < 1e-12:
                merged[-1][2] += s[2]
            else:
                merged.append(s)
        self.segs = merged


def skyline_pack(items: Sequence[PackItem], order: Sequence[int], tiles: Sequence[int],
                 scale: float, gap: float, margin: float, prefer_rotate: bool
                 ) -> Optional[List[Placement]]:
    sk = [_Skyline(margin) for _ in tiles]
    by_index = {it.index: it for it in items}
    out: List[Placement] = []
    t_cur = 0
    for idx in order:
        it = by_index[idx]
        w0, h0 = it.bbox()
        w0, h0 = w0 * scale, h0 * scale
        options = [(0, w0, h0)]
        if it.allow_rotate:
            rot = (1, h0, w0)
            options = [rot, options[0]] if prefer_rotate else [options[0], rot]
        placed = False
        for ti in range(t_cur, len(tiles)):
            best = None
            for k, w, h in options:
                r = sk[ti].find(w, h, gap)
                if r is None:
                    continue
                x, y, _ = r
                cand = (y + h, x, k, w, h)
                if best is None or cand[0] < best[0] - 1e-12 or \
                        (abs(cand[0] - best[0]) <= 1e-12 and cand[1] < best[1] - 1e-12):
                    best = cand
                if not prefer_rotate and k == 0:
                    break  # 元の向きで置けるなら優先
            if best is None:
                continue
            top, x, k, w, h = best
            y = top - h
            sk[ti].place(x, y, w, h, gap)
            ou, ov = tile_origin(tiles[ti])
            out.append(Placement(idx, tiles[ti], k, scale, (ou + x, ov + y)))
            placed = True
            # グループを同一/近隣タイルに保つため、タイルは前進のみ
            t_cur = ti
            break
        if not placed:
            return None
    return out


def pack_full(items: Sequence[PackItem], order: Sequence[int], tiles: Sequence[int],
              gap: float, margin: float, prefer_rotate: bool = False,
              fixed_scale: Optional[float] = None, iterations: int = 28
              ) -> Tuple[Optional[List[Placement]], float]:
    """返り値: (placements, scale)。fixed_scale 指定時はその倍率のみ試す。"""
    if not items:
        return [], 1.0
    if fixed_scale is not None:
        return skyline_pack(items, order, tiles, fixed_scale, gap, margin, prefer_rotate), fixed_scale
    usable = len(tiles) * (1 - 2 * margin) ** 2
    area = 0.0
    max_ext = 0.0
    for it in items:
        uvt = it.uv[it.tris]
        a = np.abs(0.5 * ((uvt[:, 1, 0] - uvt[:, 0, 0]) * (uvt[:, 2, 1] - uvt[:, 0, 1])
                          - (uvt[:, 1, 1] - uvt[:, 0, 1]) * (uvt[:, 2, 0] - uvt[:, 0, 0]))).sum()
        area += a
        max_ext = max(max_ext, min(it.bbox()))
    if area <= 0:
        return None, 0.0
    hi = math.sqrt(usable / area)
    if max_ext > 0:
        hi = min(hi, (1 - 2 * margin) / max_ext)
    lo = 0.0
    best: Optional[List[Placement]] = None
    best_s = 0.0
    s = hi
    r = skyline_pack(items, order, tiles, s, gap, margin, prefer_rotate)
    if r is not None:
        return r, s
    probe = hi * 1e-3
    r = skyline_pack(items, order, tiles, probe, gap, margin, prefer_rotate)
    if r is None:
        return None, 0.0
    lo, best, best_s = probe, r, probe
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        r = skyline_pack(items, order, tiles, mid, gap, margin, prefer_rotate)
        if r is not None:
            lo, best, best_s = mid, r, mid
        else:
            hi = mid
        if hi - lo <= hi * 1e-4:
            break
    return best, best_s


# ---------------------------------------------------------- free space
def place_free(item: PackItem, scale: float, tiles: Sequence[int], index: ObstacleIndex,
               gap: float, margin: float, budget: int,
               rotations: Sequence[int] = (0, 1, 2, 3)) -> Optional[Placement]:
    """空き領域への配置。見つからなければ None (指定予算で実行可能解なし)。"""
    cell = max(gap / 2.0, 1.0 / 1024) if gap > 0 else 1.0 / 1024
    n = int(math.ceil(1.0 / cell))
    tried = 0
    for ti, tile in enumerate(tiles):
        ou, ov = tile_origin(tile)
        obst = index.tris_in_tile(tile)
        occ = rasterize(obst, (ou, ov), cell, n, gap / 2.0) if len(obst) else np.zeros((n, n), bool)
        for k in rotations if item.allow_rotate else (0,):
            q = rotate90(item.uv, k) * scale
            q = q - q.min(axis=0)
            w, h = np.ptp(q, axis=0)
            if w > 1 - 2 * margin + 1e-12 or h > 1 - 2 * margin + 1e-12:
                continue
            # 島の原点を格子点に合わせ、膨張分 p セルの余白を付けてラスタ化 (保守的)
            p = int(math.ceil((gap / 2.0) / cell)) + 2
            size = int(math.ceil(max(w, h) / cell)) + 2 * p + 2
            foot = rasterize(q[item.tris], (-p * cell, -p * cell), cell, size, gap / 2.0)
            used = np.argwhere(foot)
            if len(used) == 0:
                continue
            foot = foot[:used[:, 0].max() + 1, :used[:, 1].max() + 1]
            # タイル外側 p セルは空き (端余白は算術的に判定)
            occ_p = np.zeros((n + 2 * p, n + 2 * p), dtype=bool)
            occ_p[p:p + n, p:p + n] = occ
            counts = _correlate(occ_p, foot)
            # counts[x] は島原点がタイルのセル x にある場合の重なり数
            ix_min = int(math.ceil(margin / cell - 1e-9))
            ix_max = min(int(math.floor((1 - margin - w) / cell + 1e-9)), n - 1)
            iy_max = min(int(math.floor((1 - margin - h) / cell + 1e-9)), n - 1)
            if ix_max < ix_min or iy_max < ix_min:
                continue
            ok_x = np.arange(ix_min, ix_max + 1)
            ok_y = np.arange(ix_min, iy_max + 1)
            sub = counts[np.ix_(ok_x, ok_y)]
            free = np.argwhere(sub == 0)
            if len(free) == 0:
                continue
            # 左下優先 (y, x)
            free = free[np.lexsort((free[:, 0], free[:, 1]))]
            for fx, fy in free:
                tried += 1
                if tried > budget:
                    return None
                off = (ou + ok_x[fx] * cell, ov + ok_y[fy] * cell)
                placed = q + np.asarray(off)
                if not np.all(placed.max(axis=0) <= (ou + 1 - margin + 1e-12, ov + 1 - margin + 1e-12)):
                    continue
                if not index.collides(placed[item.tris], gap):
                    return Placement(item.index, tile, k, scale, off)
    return None


def _correlate(occ: np.ndarray, foot: np.ndarray) -> np.ndarray:
    """counts[x,y] = Σ occ[x+i, y+j] * foot[i,j] (範囲外は占有扱い)。"""
    n = occ.shape[0]
    fw, fh = foot.shape
    pad = np.ones((n + fw, n + fh), dtype=np.float64)
    pad[:n, :n] = occ
    A = np.fft.rfft2(pad)
    F = np.zeros_like(pad)
    F[:fw, :fh] = foot[::-1, ::-1]
    B = np.fft.rfft2(F)
    full = np.fft.irfft2(A * B, s=pad.shape)
    out = full[fw - 1:fw - 1 + n, fh - 1:fh - 1 + n]
    return np.rint(out).astype(np.int64)
