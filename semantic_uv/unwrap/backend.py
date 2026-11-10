"""展開バックエンド。

- 内蔵 LSCM (最小二乗等角写像): 自由境界 (2点pin) と境界固定の双方に対応。
- 境界固定のフォールバックとして一様重み Tutte 写像。
- Blender 標準 Angle Based / Conformal は blender.compatibility から
  register_external_backend() で登録された関数を作業コピー上で呼ぶ。
返値は呼び出し側で独立に検証する。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..analysis.topology import Topology


@dataclass
class UnwrapJob:
    """1 Chart 分の展開問題。UV頂点 (溶接グループ) 単位。"""
    positions: np.ndarray                    # (n,3) ワールド座標
    polys: List[List[int]]                   # 面 -> UV頂点 index (loop順)
    tris: np.ndarray                         # (T,3) UV頂点 index
    pinned: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    initial: Optional[np.ndarray] = None     # 改善モードの初期解 (n,2)


ExternalBackend = Callable[[List[UnwrapJob], str], Optional[List[Optional[np.ndarray]]]]
_external: Optional[ExternalBackend] = None


def register_external_backend(fn: Optional[ExternalBackend]) -> None:
    global _external
    _external = fn


def external_available() -> bool:
    return _external is not None


def unwrap_external(jobs: List[UnwrapJob], method: str) -> Optional[List[Optional[np.ndarray]]]:
    if _external is None:
        return None
    return _external(jobs, method)


# ---------------------------------------------------------------------
def build_job(topo: Topology, faces: Sequence[int], group: np.ndarray, n_groups: int,
              ) -> Tuple[UnwrapJob, np.ndarray]:
    """weld_groups の結果から UnwrapJob を作る。返り値の2つ目は
    UV頂点 index -> 代表 loop。"""
    obj = topo.obj
    w = obj.world_co()
    rep = -np.ones(n_groups, dtype=np.int64)
    positions = np.zeros((n_groups, 3))
    polys: List[List[int]] = []
    for f in faces:
        poly = []
        for l in obj.face_loops(f):
            g = int(group[l])
            if rep[g] < 0:
                rep[g] = l
                positions[g] = w[obj.loop_vert[l]]
            poly.append(g)
        polys.append(poly)
    fset = set(int(f) for f in faces)
    tris = [tuple(int(group[l]) for l in topo.tris[t]) for f in faces for t in topo.face_tris(f)]
    tri_arr = np.asarray(tris, dtype=np.int64) if tris else np.zeros((0, 3), dtype=np.int64)
    return UnwrapJob(positions, polys, tri_arr), rep


def _local_frames(P: np.ndarray, tris: np.ndarray):
    """各三角形を自身の平面に置いた 2D 座標 (p0=原点, p1=x軸上)。"""
    a, b, c = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
    e1 = b - a
    e2 = c - a
    l1 = np.linalg.norm(e1, axis=1)
    l1s = np.where(l1 > 0, l1, 1.0)
    x = e1 / l1s[:, None]
    n = np.cross(e1, e2)
    ln = np.linalg.norm(n, axis=1)
    lns = np.where(ln > 0, ln, 1.0)
    y = np.cross(n / lns[:, None], x)
    z0 = np.zeros(len(tris), dtype=complex)
    z1 = l1 + 0j
    z2 = np.einsum("ij,ij->i", e2, x) + 1j * np.einsum("ij,ij->i", e2, y)
    area2 = ln  # 2 * area
    return z0, z1, z2, area2


def _lscm_system(job: UnwrapJob):
    """返り値: rows, cols, vals (2T x 2n の疎行列 COO)。変数順は [u0,v0,u1,v1,...]。"""
    tris = job.tris
    z0, z1, z2, d = _local_frames(job.positions, tris)
    valid = d > 0
    W = np.stack([z2 - z1, z0 - z2, z1 - z0], axis=1)  # (T,3) complex
    s = np.where(valid, 1.0 / np.sqrt(np.where(valid, d, 1.0)), 0.0)
    W = W * s[:, None]
    T = len(tris)
    rows, cols, vals = [], [], []
    for j in range(3):
        a = W[:, j].real
        b = W[:, j].imag
        vid = tris[:, j]
        r_re = 2 * np.arange(T)
        r_im = r_re + 1
        rows += [r_re, r_re, r_im, r_im]
        cols += [2 * vid, 2 * vid + 1, 2 * vid, 2 * vid + 1]
        vals += [a, -b, b, a]
    return (np.concatenate(rows), np.concatenate(cols), np.concatenate(vals), 2 * T)


def _solve_least_squares(rows, cols, vals, n_rows, n_vars, fixed: Dict[int, float],
                         x0: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """|A x|^2 を固定変数付きで最小化。小規模は密、大規模は前処理付き CG。"""
    free_mask = np.ones(n_vars, dtype=bool)
    xfix = np.zeros(n_vars)
    for k, v in fixed.items():
        free_mask[k] = False
        xfix[k] = v
    free_idx = np.nonzero(free_mask)[0]
    nf = len(free_idx)
    if nf == 0:
        return xfix
    remap = -np.ones(n_vars, dtype=np.int64)
    remap[free_idx] = np.arange(nf)
    # b = -A_fixed x_fixed
    fixed_entries = ~free_mask[cols]
    b = np.zeros(n_rows)
    np.add.at(b, rows[fixed_entries], -vals[fixed_entries] * xfix[cols[fixed_entries]])
    fr = rows[~fixed_entries]
    fc = remap[cols[~fixed_entries]]
    fv = vals[~fixed_entries]

    if nf <= 1200:
        A = np.zeros((n_rows, nf))
        np.add.at(A, (fr, fc), fv)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        out = xfix.copy()
        out[free_idx] = sol
        return out if np.all(np.isfinite(out)) else None

    # CG on normal equations
    def Ax(x):
        return np.bincount(fr, weights=fv * x[fc], minlength=n_rows)

    def ATy(y):
        return np.bincount(fc, weights=fv * y[fr], minlength=nf)

    diag = np.bincount(fc, weights=fv * fv, minlength=nf)
    diag[diag == 0] = 1.0
    rhs = ATy(b)
    x = x0[free_idx].copy() if x0 is not None else np.zeros(nf)
    r = rhs - ATy(Ax(x))
    z = r / diag
    p = z.copy()
    rz = float(r @ z)
    norm_rhs = float(np.linalg.norm(rhs)) or 1.0
    max_iter = min(20000, 50 * int(math.sqrt(nf)) + 2000)
    for _ in range(max_iter):
        if math.sqrt(abs(rz)) <= 1e-12 * norm_rhs or np.linalg.norm(r) <= 1e-11 * norm_rhs:
            break
        Ap = ATy(Ax(p))
        denom = float(p @ Ap)
        if denom <= 0 or not math.isfinite(denom):
            break
        alpha = rz / denom
        x += alpha * p
        r -= alpha * Ap
        z = r / diag
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    out = xfix.copy()
    out[free_idx] = x
    return out if np.all(np.isfinite(out)) else None


def _pick_free_pins(job: UnwrapJob) -> Dict[int, Tuple[float, float]]:
    """主軸方向の両端2点を (0,0),(1,0) に固定する。"""
    P = job.positions
    used = np.unique(job.tris) if len(job.tris) else np.arange(len(P))
    Q = P[used] - P[used].mean(axis=0)
    _, vecs = np.linalg.eigh(Q.T @ Q)
    axis = vecs[:, -1]
    proj = Q @ axis
    i0 = int(used[int(np.argmin(proj))])
    i1 = int(used[int(np.argmax(proj))])
    if i0 == i1:
        i1 = int(used[-1]) if int(used[0]) == i0 else int(used[0])
    d = float(np.linalg.norm(P[i1] - P[i0])) or 1.0
    return {i0: (0.0, 0.0), i1: (d, 0.0)}


def lscm(job: UnwrapJob) -> Optional[np.ndarray]:
    n = len(job.positions)
    if n == 0 or len(job.tris) == 0:
        return None
    pins = dict(job.pinned)
    if len(pins) < 2:
        pins = dict(_pick_free_pins(job)) if not pins else _extend_single_pin(job, pins)
    rows, cols, vals, nr = _lscm_system(job)
    fixed = {}
    for k, (u, v) in pins.items():
        fixed[2 * k] = float(u)
        fixed[2 * k + 1] = float(v)
    x0 = None
    if job.initial is not None:
        x0 = job.initial.reshape(-1).astype(np.float64)
    else:
        x0 = _planar_guess(job, pins).reshape(-1)
    x = _solve_least_squares(rows, cols, vals, nr, 2 * n, fixed, x0)
    if x is None:
        return None
    uv = x.reshape(n, 2)
    for k, (u, v) in pins.items():   # 固定値を最後に元の数値で代入し直す
        uv[k] = (u, v)
    return uv


def _extend_single_pin(job: UnwrapJob, pins):
    """pin 1点のみ: 位置はそのまま、もう1点を主軸方向に仮固定し、
    後段の配置で回転・拡縮しない (pin 位置は保持される)。"""
    (k0, (u0, v0)), = pins.items()
    auto = _pick_free_pins(job)
    far = max(auto, key=lambda k: np.linalg.norm(job.positions[k] - job.positions[k0]))
    if far == k0:
        return pins
    d = float(np.linalg.norm(job.positions[far] - job.positions[k0]))
    out = dict(pins)
    out[far] = (u0 + d, v0)
    return out


def _planar_guess(job: UnwrapJob, pins) -> np.ndarray:
    P = job.positions
    Q = P - P.mean(axis=0)
    _, vecs = np.linalg.eigh(Q.T @ Q)
    uv = np.c_[Q @ vecs[:, -1], Q @ vecs[:, -2]]
    return uv


def tutte(job: UnwrapJob) -> Optional[np.ndarray]:
    """境界固定の一様重み調和写像。pinned が境界を与える。"""
    n = len(job.positions)
    if len(job.pinned) < 3:
        return None
    nbr: List[set] = [set() for _ in range(n)]
    for poly in job.polys:
        m = len(poly)
        for i in range(m):
            a, b = poly[i], poly[(i + 1) % m]
            if a != b:
                nbr[a].add(b)
                nbr[b].add(a)
    rows, cols, vals = [], [], []
    r = 0
    for i in range(n):
        if i in job.pinned or not nbr[i]:
            continue
        deg = float(len(nbr[i]))
        for comp in (0, 1):
            rows.append(r)
            cols.append(2 * i + comp)
            vals.append(deg)
            for j in sorted(nbr[i]):
                rows.append(r)
                cols.append(2 * j + comp)
                vals.append(-1.0)
            r += 1
    fixed = {}
    for k, (u, v) in job.pinned.items():
        fixed[2 * k] = float(u)
        fixed[2 * k + 1] = float(v)
    if r == 0:
        uv = np.zeros((n, 2))
        for k, (u, v) in job.pinned.items():
            uv[k] = (u, v)
        return uv
    x = _solve_least_squares(np.asarray(rows), np.asarray(cols), np.asarray(vals, dtype=float),
                             r, 2 * n, fixed,
                             job.initial.reshape(-1) if job.initial is not None else None)
    if x is None:
        return None
    uv = x.reshape(n, 2)
    for k, (u, v) in job.pinned.items():
        uv[k] = (u, v)
    return uv
