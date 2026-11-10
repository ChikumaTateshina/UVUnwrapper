"""型付き設定モデル (実装指示書 3章)。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class Scope(str, Enum):
    ALL_FACES = "ALL_FACES"
    SELECTED_FACES = "SELECTED_FACES"


class UVMode(str, Enum):
    GENERATE = "GENERATE"
    IMPROVE = "IMPROVE"


class BoundaryMode(str, Enum):
    KEEP_BOUNDARY = "KEEP_BOUNDARY"
    RELOCATE_SELECTED = "RELOCATE_SELECTED"


class SeamMode(str, Enum):
    PROTECT = "PROTECT"
    PREFER = "PREFER"
    IGNORE = "IGNORE"


class TileMode(str, Enum):
    SINGLE = "SINGLE"
    UDIM = "UDIM"


class Preset(str, Enum):
    BALANCED = "BALANCED"
    TEXTURE_PAINTING = "TEXTURE_PAINTING"
    COMPACT_PACKING = "COMPACT_PACKING"


class SharedMeshPolicy(str, Enum):
    STOP = "STOP"
    MAKE_SINGLE_USER = "MAKE_SINGLE_USER"


class TopologyPolicy(str, Enum):
    """非多様体・向き不整合エッジと退化面の扱い。メッシュはどちらでも変更しない。
    SPLIT: 問題エッジで UV を強制分割し、退化面は向き・退化の判定から除外して警告。
    STOP : 問題箇所を報告して停止 (設計仕様 R-007 の厳密な動作)。"""
    SPLIT = "SPLIT"
    STOP = "STOP"


class UnwrapBackend(str, Enum):
    """BLENDER_* は作業コピー上で bpy.ops.uv.unwrap を使う。
    Blender外 (単体テスト) や境界固定時は内蔵LSCMを使う。"""
    BLENDER_ANGLE_BASED = "BLENDER_ANGLE_BASED"
    BLENDER_CONFORMAL = "BLENDER_CONFORMAL"
    BUILTIN_LSCM = "BUILTIN_LSCM"


class DensityMode(str, Enum):
    AUTO = "AUTO"          # 全体: 共通倍率を領域に適合 / 部分再配置: 固定領域から推定
    SPECIFIED = "SPECIFIED"  # px / ワールド単位 で明示


@dataclass(frozen=True)
class SoftWeights:
    """ソフト目的の重み。ハード制約はプリセットに依存しない。"""
    distortion: float = 1.0
    seam_length: float = 0.3
    island_count: float = 0.2
    slenderness: float = 0.2
    material_split: float = 0.3
    readability: float = 0.5   # グループ近接・方向統一
    packing: float = 0.5       # 充填率


PRESET_WEIGHTS: Dict[Preset, SoftWeights] = {
    Preset.BALANCED: SoftWeights(),
    Preset.TEXTURE_PAINTING: SoftWeights(distortion=1.5, seam_length=0.5,
                                         island_count=0.4, readability=0.9,
                                         packing=0.3),
    Preset.COMPACT_PACKING: SoftWeights(distortion=0.8, seam_length=0.2,
                                        island_count=0.1, readability=0.2,
                                        packing=1.0),
}


@dataclass(frozen=True)
class SearchBudget:
    """局所探索の主予算は候補数・反復数。時間とメモリは安全弁。"""
    max_candidates: int = 64
    max_iterations: int = 24
    placement_candidates: int = 4000
    seed: int = 0
    time_limit_s: float = 120.0
    memory_limit_mb: int = 4096


@dataclass(frozen=True)
class Settings:
    scope: Scope = Scope.ALL_FACES
    uv_mode: UVMode = UVMode.GENERATE
    boundary_mode: BoundaryMode = BoundaryMode.KEEP_BOUNDARY
    seam_mode: SeamMode = SeamMode.PROTECT
    tile_mode: TileMode = TileMode.SINGLE
    allowed_tiles: Tuple[int, ...] = (1001,)
    resolution: int = 2048
    gap_px: float = 8.0
    density_mode: DensityMode = DensityMode.AUTO
    density_px_per_unit: float = 0.0
    preset: Preset = Preset.BALANCED
    search: SearchBudget = field(default_factory=SearchBudget)
    shared_mesh_policy: SharedMeshPolicy = SharedMeshPolicy.STOP
    topology_policy: TopologyPolicy = TopologyPolicy.SPLIT
    backend: UnwrapBackend = UnwrapBackend.BLENDER_ANGLE_BASED
    # シーム生成の閾値 (度)。Chartの法線コーン半角。
    chart_angle_limit_deg: float = 66.0
    # 改善モードでChartを再展開とみなす角度歪みの上限 (0..1)
    distortion_limit: float = 0.25
    # 入力UVレイヤー名 (object_key -> 名前)。未指定はアクティブUV。
    input_uv: Tuple[Tuple[str, str], ...] = ()
    result_prefix: str = "SUO_Result_"

    @property
    def weights(self) -> SoftWeights:
        return PRESET_WEIGHTS[self.preset]

    @property
    def gap_uv(self) -> float:
        return self.gap_px / float(self.resolution)

    def effective_tiles(self) -> Tuple[int, ...]:
        if self.tile_mode == TileMode.SINGLE:
            return (1001,)
        return tuple(sorted(set(self.allowed_tiles)))

    def input_uv_for(self, object_key: str) -> Optional[str]:
        for key, name in self.input_uv:
            if key == object_key:
                return name
        return None

    def to_dict(self) -> dict:
        def conv(v):
            if isinstance(v, Enum):
                return v.value
            if isinstance(v, dict):
                return {k: conv(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [conv(x) for x in v]
            return v
        return conv(asdict(self))

    def fingerprint(self) -> str:
        data = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(data).hexdigest()


def validate_settings(settings: Settings) -> List[str]:
    """設定自体の不整合を返す (空なら妥当)。"""
    problems: List[str] = []
    if settings.resolution <= 0:
        problems.append("resolution は正の整数である必要があります")
    if settings.gap_px < 0:
        problems.append("gap_px は0以上である必要があります")
    if settings.tile_mode == TileMode.UDIM:
        if not settings.allowed_tiles:
            problems.append("UDIM では許可タイルを明示指定してください")
        for t in settings.allowed_tiles:
            if not is_valid_udim(t):
                problems.append(f"不正なUDIM番号: {t}")
    if settings.density_mode == DensityMode.SPECIFIED and settings.density_px_per_unit <= 0:
        problems.append("密度を指定する場合は正の値を入力してください")
    return problems


def is_valid_udim(tile: int) -> bool:
    if tile < 1001:
        return False
    u = (tile - 1001) % 10
    v = (tile - 1001) // 10
    return 0 <= u <= 9 and v >= 0


def udim_to_offset(tile: int) -> Tuple[int, int]:
    """UDIM = 1001 + u + 10*v"""
    return (tile - 1001) % 10, (tile - 1001) // 10


def offset_to_udim(u: int, v: int) -> int:
    return 1001 + u + 10 * v
