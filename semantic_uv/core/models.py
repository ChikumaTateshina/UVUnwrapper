"""内部データモデル (設計仕様 4章)。

ID は処理スナップショット内で有効な (object_key, local_id)。
UV は頂点ではなく面コーナー (loop) 単位で扱う。
このモジュールは Blender を import しない。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


class EdgeConstraint(int, Enum):
    AUTO = 0
    PREFER = 1
    REQUIRE = 2
    AVOID = 3
    FORBID = 4


@dataclass
class MaterialInfo:
    name: str
    # 静的に取得できた Base Color (linear RGB)。複雑なノードは None = 不明。
    base_color: Optional[Tuple[float, float, float]] = None


@dataclass
class ObjectData:
    """1 Mesh Object のスナップショット (純データ)。"""
    key: str
    mesh_key: str
    matrix_world: np.ndarray                 # (4,4)
    co: np.ndarray                           # (V,3) ローカル座標
    edges: np.ndarray                        # (E,2) 頂点インデックス
    face_loop_start: np.ndarray              # (F,)
    face_loop_total: np.ndarray              # (F,)
    loop_vert: np.ndarray                    # (L,)
    loop_edge: np.ndarray                    # (L,)
    face_material: np.ndarray                # (F,)
    face_select: np.ndarray                  # (F,) bool
    face_hide: np.ndarray                    # (F,) bool
    seams: np.ndarray                        # (E,) bool
    edge_constraint: np.ndarray              # (E,) int (EdgeConstraint)
    materials: List[MaterialInfo] = field(default_factory=list)
    uv_layer_names: List[str] = field(default_factory=list)
    uv: Dict[str, np.ndarray] = field(default_factory=dict)          # name -> (L,2) float64
    uv_pin: Dict[str, np.ndarray] = field(default_factory=dict)      # name -> (L,) bool
    uv_select: Dict[str, np.ndarray] = field(default_factory=dict)   # name -> (L,) bool
    active_uv: Optional[str] = None
    render_uv: Optional[str] = None
    input_uv_name: Optional[str] = None
    # Blender の loop_triangles (T,3)。描画・ベイクと同じ三角形で展開・重複判定するため。
    # None の場合は analysis.topology.triangulate の分割を使う。
    tri_loops: Optional[np.ndarray] = None
    # Blender 側情報 (純データとしてのみ保持)
    editable: bool = True
    mesh_users: int = 1
    mesh_user_keys: Tuple[str, ...] = ()

    # ---- 派生値 -------------------------------------------------------
    @property
    def n_faces(self) -> int:
        return int(len(self.face_loop_start))

    @property
    def n_loops(self) -> int:
        return int(len(self.loop_vert))

    @property
    def n_edges(self) -> int:
        return int(len(self.edges))

    def face_loops(self, f: int) -> range:
        s = int(self.face_loop_start[f])
        return range(s, s + int(self.face_loop_total[f]))

    def world_co(self) -> np.ndarray:
        cached = getattr(self, "_world_co", None)
        if cached is None:
            m = np.asarray(self.matrix_world, dtype=np.float64)
            h = np.c_[self.co, np.ones(len(self.co))]
            cached = (h @ m.T)[:, :3]
            object.__setattr__(self, "_world_co", cached)
        return cached

    @property
    def mirror_transform(self) -> bool:
        return float(np.linalg.det(np.asarray(self.matrix_world)[:3, :3])) < 0.0

    def loop_face(self) -> np.ndarray:
        cached = getattr(self, "_loop_face", None)
        if cached is None:
            cached = np.repeat(np.arange(self.n_faces), self.face_loop_total)
            object.__setattr__(self, "_loop_face", cached)
        return cached

    def input_uv(self) -> Optional[np.ndarray]:
        if self.input_uv_name is None:
            return None
        return self.uv.get(self.input_uv_name)

    def input_pin(self) -> np.ndarray:
        if self.input_uv_name and self.input_uv_name in self.uv_pin:
            return self.uv_pin[self.input_uv_name]
        return np.zeros(self.n_loops, dtype=bool)

    def input_select(self) -> np.ndarray:
        if self.input_uv_name and self.input_uv_name in self.uv_select:
            return self.uv_select[self.input_uv_name]
        return np.zeros(self.n_loops, dtype=bool)

    def topology_hash(self) -> str:
        h = hashlib.sha256()
        for arr in (self.edges, self.face_loop_start, self.face_loop_total,
                    self.loop_vert, self.loop_edge):
            h.update(np.ascontiguousarray(arr, dtype=np.int64).tobytes())
        h.update(str(len(self.co)).encode())
        return h.hexdigest()

    def fingerprint(self) -> str:
        """Apply 直前の再検証に使う入力指紋。"""
        h = hashlib.sha256()
        h.update(self.topology_hash().encode())
        h.update(np.ascontiguousarray(self.co, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(self.matrix_world, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(self.seams, dtype=np.bool_).tobytes())
        h.update(np.ascontiguousarray(self.edge_constraint, dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(self.face_select, dtype=np.bool_).tobytes())
        h.update(np.ascontiguousarray(self.face_hide, dtype=np.bool_).tobytes())
        h.update(np.ascontiguousarray(self.face_material, dtype=np.int64).tobytes())
        for name in self.uv_layer_names:
            h.update(name.encode("utf-8"))
            h.update(np.ascontiguousarray(self.uv[name], dtype=np.float64).tobytes())
            if name in self.uv_pin:
                h.update(np.ascontiguousarray(self.uv_pin[name], dtype=np.bool_).tobytes())
        h.update(str(self.input_uv_name).encode("utf-8"))
        return h.hexdigest()


@dataclass
class InputSnapshot:
    objects: Dict[str, ObjectData]
    order: List[str]                       # 安定した処理順
    excluded: List[Tuple[str, str]] = field(default_factory=list)  # (name, reason)
    context_state: dict = field(default_factory=dict)  # モード・アクティブ・選択 (Blender側で利用)

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for key in self.order:
            h.update(key.encode("utf-8"))
            h.update(self.objects[key].fingerprint().encode())
        return h.hexdigest()

    def iter_objects(self):
        for key in self.order:
            yield self.objects[key]


@dataclass
class EdgeFeatures:
    """エッジ単位の特徴量。配列は ObjectData.edges と同じ順。"""
    object_key: str
    faces: List[Tuple[int, ...]]           # 接続面
    length: np.ndarray
    dihedral: np.ndarray                   # ラジアン, 境界/非多様体は nan
    concavity: np.ndarray                  # +凸 / -凹 (符号付き)
    material_boundary: np.ndarray          # bool
    color_delta: np.ndarray                # OKLab ΔE, 不明は nan
    existing_seam: np.ndarray              # bool
    constraint: np.ndarray                 # EdgeConstraint
    score: np.ndarray                      # シーム候補スコア (大きいほど切りたい)


@dataclass
class UVChart:
    object_key: str
    faces: List[int]
    world_area: float = 0.0
    distortion: float = 0.0
    group: int = -1


@dataclass
class UVIsland:
    """出力UVの1アイランド。loops は object 内 loop index。"""
    object_key: str
    faces: List[int]
    loops: np.ndarray                      # (n,)
    uv: np.ndarray                         # (n,2) loops と同順
    world_area: float = 0.0
    locked: bool = False                   # 固定 (U, pin, 境界固定) なら移動不可
    movable: bool = True
    tile: int = 1001
    origin: str = "generated"              # generated / kept / fixed / improved
    group: int = -1
    descriptor: Optional[np.ndarray] = None
    symmetry_partner: Optional[int] = None

    @property
    def uid(self) -> Tuple[str, int]:
        return (self.object_key, int(min(self.faces)) if self.faces else -1)

    def uv_area(self, tris: np.ndarray) -> float:
        """tris: (T,3) island 内ローカルインデックス"""
        if len(tris) == 0:
            return 0.0
        a, b, c = self.uv[tris[:, 0]], self.uv[tris[:, 1]], self.uv[tris[:, 2]]
        cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        return float(np.sum(np.abs(cross)) * 0.5)


@dataclass
class UVGroup:
    gid: int
    members: List[int]                     # island index
    material: Optional[int] = None
    object_key: Optional[str] = None
    priority: float = 0.0
    orientation: float = 0.0
    parent: Optional[int] = None


@dataclass
class FullScope:
    """ALL_FACES。全面が書込可能。"""
    selected_faces: Dict[str, np.ndarray]  # object_key -> 全面 index
    kind: str = "FULL"

    def writable_faces(self, key: str) -> np.ndarray:
        return self.selected_faces.get(key, np.zeros(0, dtype=np.int64))


@dataclass
class PartialScope:
    selected_faces: Dict[str, np.ndarray]          # S
    unselected_faces: Dict[str, np.ndarray]        # U (隠れ面を含む)
    writable_loops: Dict[str, np.ndarray]          # bool (L,)
    boundary_edges: Dict[str, Set[int]]            # S/U 境界 + S側メッシュ外周
    boundary_loops: Dict[str, np.ndarray]          # bool (L,): S側境界コーナー
    pinned_loops: Dict[str, np.ndarray]            # bool (L,)
    components: Dict[str, List[List[int]]]         # Sの連結成分 (面)
    boundary_mode: str = "KEEP_BOUNDARY"
    kind: str = "PARTIAL"

    @property
    def selected_face_count(self) -> int:
        return int(sum(len(v) for v in self.selected_faces.values()))

    def writable_faces(self, key: str) -> np.ndarray:
        return self.selected_faces.get(key, np.zeros(0, dtype=np.int64))


@dataclass
class ObstacleSet:
    """UV三角形の固定障害物 (タイルローカルではなく絶対UV座標)。"""
    tris: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 2)))
    owners: List[Tuple[str, int]] = field(default_factory=list)  # (object_key, face)


@dataclass
class LayoutDomain:
    tiles: Tuple[int, ...]
    resolution: int
    gap_uv: float
    edge_margin_uv: float
    obstacles: ObstacleSet = field(default_factory=ObstacleSet)
    density_k: Optional[float] = None      # 目標 k = A_uv / A_world (None は自動)
    density_source: str = "auto"


@dataclass
class ObjectPlan:
    object_key: str
    uv: np.ndarray                         # (L,2) 全コーナー候補
    pin: np.ndarray                        # (L,) bool
    select: np.ndarray                     # (L,) bool
    writable: np.ndarray                   # (L,) bool 書込許可集合
    candidate_seams: Set[int] = field(default_factory=set)
    island_tiles: Dict[int, int] = field(default_factory=dict)  # 代表面 -> タイル

    def changed_loops(self, source_uv: Optional[np.ndarray]) -> int:
        if source_uv is None:
            return int(self.writable.sum())
        diff = np.any(self.uv != source_uv, axis=1)
        return int(diff.sum())


@dataclass
class QualityMetrics:
    mean_angle_distortion: float = 0.0
    max_angle_distortion: float = 0.0
    mean_area_distortion: float = 0.0
    max_area_distortion: float = 0.0
    density_deviation_max: float = 0.0
    fill_ratio: float = 0.0
    island_count: int = 0
    seam_length: float = 0.0
    group_dispersion: float = 0.0
    orientation_agreement: float = 1.0
    total_cost: float = 0.0

    def as_rows(self) -> List[Tuple[str, str]]:
        return [
            ("平均角度歪み", f"{self.mean_angle_distortion:.4f}"),
            ("最大角度歪み", f"{self.max_angle_distortion:.4f}"),
            ("平均面積歪み", f"{self.mean_area_distortion:.4f}"),
            ("最大面積歪み", f"{self.max_area_distortion:.4f}"),
            ("密度偏差(最大)", f"{self.density_deviation_max * 100:.2f}%"),
            ("充填率", f"{self.fill_ratio * 100:.2f}%"),
            ("島数", str(self.island_count)),
            ("シーム長", f"{self.seam_length:.4f}"),
            ("グループ分散(代理指標)", f"{self.group_dispersion:.4f}"),
            ("方向一致率(代理指標)", f"{self.orientation_agreement * 100:.1f}%"),
            ("総合コスト", f"{self.total_cost:.4f}"),
        ]


@dataclass
class ResultPlan:
    """Blender参照を含まない計算結果。"""
    objects: Dict[str, ObjectPlan]
    order: List[str]
    input_fingerprint: str
    object_fingerprints: Dict[str, str]
    settings_fingerprint: str
    islands: List[UVIsland] = field(default_factory=list)
    groups: List[UVGroup] = field(default_factory=list)
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    baseline_quality: Optional[QualityMetrics] = None
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    search_log: List[str] = field(default_factory=list)
    no_op: bool = False
    partial_summary: Dict[str, object] = field(default_factory=dict)

    def write_targets(self) -> List[str]:
        return [k for k in self.order if self.objects[k].writable.any()]


class TxState(str, Enum):
    IDLE = "IDLE"
    SNAPSHOTTED = "SNAPSHOTTED"
    COMPUTING = "COMPUTING"
    VALIDATED = "VALIDATED"
    PREVIEW = "PREVIEW"
    COMMITTING = "COMMITTING"
    COMMITTED = "COMMITTED"
    ROLLING_BACK = "ROLLING_BACK"
    ROLLED_BACK = "ROLLED_BACK"
    ROLLBACK_FAILED = "ROLLBACK_FAILED"
    RESTORED = "RESTORED"


@dataclass
class TransactionRecord:
    tx_id: str
    state: TxState
    created: str
    objects: List[str]
    # object_key -> 変更前後ハッシュ・作成資源・参照変更
    entries: Dict[str, dict] = field(default_factory=dict)
    log: List[str] = field(default_factory=list)
    kind: str = "UV_RESULT"   # UV_RESULT / SEAMS
    format_version: int = 1
