"""UI 設定 (Scene / Object プロパティ) と型付き Settings への変換。"""
from __future__ import annotations

import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, IntProperty,
                       PointerProperty, StringProperty)

from ..core.settings import (BoundaryMode, DensityMode, Preset, Scope, SearchBudget, SeamMode,
                             Settings, SharedMeshPolicy, TileMode, UnwrapBackend, UVMode)


class SUO_Settings(bpy.types.PropertyGroup):
    scope: EnumProperty(name="処理範囲", items=[
        ("ALL_FACES", "全面", "全ての面を展開"),
        ("SELECTED_FACES", "選択面", "選択面のみを再展開 (入力UVが必要)")], default="ALL_FACES")
    uv_mode: EnumProperty(name="生成／改善", items=[
        ("GENERATE", "新規生成", "UVを新規に生成"),
        ("IMPROVE", "既存改善", "既存UVを初期解として改善")], default="GENERATE")
    boundary_mode: EnumProperty(name="境界モード", items=[
        ("KEEP_BOUNDARY", "境界固定", "選択境界と未選択側を固定し、内部のみ再計算"),
        ("RELOCATE_SELECTED", "選択側を分離して再配置",
         "未選択側固定、選択側は境界分離して空き領域へ配置 (3Dメッシュは切断しません)")],
        default="KEEP_BOUNDARY")
    seam_mode: EnumProperty(name="既存シーム", items=[
        ("PROTECT", "保護", "既存シームを維持"),
        ("PREFER", "優先", "既存シームを優先し、変更箇所と理由を表示"),
        ("IGNORE", "無視", "既存シームを採点に使わない (明示制約は維持)")], default="PROTECT")
    tile_mode: EnumProperty(name="タイル", items=[
        ("SINGLE", "単一タイル (0〜1)", ""), ("UDIM", "UDIM (明示指定)", "")], default="SINGLE")
    allowed_tiles: StringProperty(name="許可タイル", default="1001",
                                  description="UDIM番号をカンマ区切りで指定 (例: 1001,1002,1011)")
    resolution: IntProperty(name="タイル解像度", default=2048, min=16, max=65536)
    gap_px: FloatProperty(name="余白 (px)", default=8.0, min=0.0, max=512.0)
    density_mode: EnumProperty(name="密度", items=[
        ("AUTO", "自動", "全体: 共通倍率を領域に適合 / 部分再配置: 固定領域から推定"),
        ("SPECIFIED", "指定", "px/ワールド単位 を指定")], default="AUTO")
    density_px: FloatProperty(name="密度 (px/単位)", default=512.0, min=0.0)
    preset: EnumProperty(name="プリセット", items=[
        ("BALANCED", "Balanced", ""), ("TEXTURE_PAINTING", "Texture Painting", ""),
        ("COMPACT_PACKING", "Compact Packing", "")], default="BALANCED")
    backend: EnumProperty(name="展開方式", items=[
        ("BLENDER_ANGLE_BASED", "Angle Based (標準)", ""),
        ("BLENDER_CONFORMAL", "Conformal (標準)", ""),
        ("BUILTIN_LSCM", "内蔵 LSCM", "")], default="BLENDER_ANGLE_BASED")
    angle_limit: FloatProperty(name="Chart 角度上限", default=66.0, min=1.0, max=89.0)
    distortion_limit: FloatProperty(name="歪み上限", default=0.25, min=0.01, max=1.0)
    max_candidates: IntProperty(name="候補数", default=64, min=1, max=10000)
    max_iterations: IntProperty(name="反復数", default=24, min=1, max=10000)
    placement_candidates: IntProperty(name="配置候補数", default=4000, min=10, max=1000000)
    seed: IntProperty(name="シード", default=0, min=0)
    time_limit: FloatProperty(name="時間上限 (秒)", default=120.0, min=1.0)
    allow_single_user: BoolProperty(name="共有Meshの独立化を許可", default=False,
                                    description="複数Objectで共有されたMeshを確定時に独立化します")
    show_preview: BoolProperty(name="プレビュー表示", default=True)


def parse_tiles(text: str):
    out = []
    for part in text.replace(" ", "").split(","):
        if part:
            out.append(int(part))
    return tuple(out)


def to_settings(context) -> Settings:
    p = context.scene.suo_settings
    input_uv = []
    for ob in context.selected_objects:
        if ob.type == "MESH" and ob.suo_input_uv:
            input_uv.append((ob.name, ob.suo_input_uv))
    try:
        tiles = parse_tiles(p.allowed_tiles)
    except ValueError:
        tiles = ()
    return Settings(
        scope=Scope(p.scope), uv_mode=UVMode(p.uv_mode), boundary_mode=BoundaryMode(p.boundary_mode),
        seam_mode=SeamMode(p.seam_mode), tile_mode=TileMode(p.tile_mode),
        allowed_tiles=tiles if p.tile_mode == "UDIM" else (1001,),
        resolution=p.resolution, gap_px=p.gap_px, density_mode=DensityMode(p.density_mode),
        density_px_per_unit=p.density_px, preset=Preset(p.preset),
        search=SearchBudget(max_candidates=p.max_candidates, max_iterations=p.max_iterations,
                            placement_candidates=p.placement_candidates, seed=p.seed,
                            time_limit_s=p.time_limit),
        shared_mesh_policy=SharedMeshPolicy.MAKE_SINGLE_USER if p.allow_single_user else SharedMeshPolicy.STOP,
        backend=UnwrapBackend(p.backend), chart_angle_limit_deg=p.angle_limit,
        distortion_limit=p.distortion_limit, input_uv=tuple(sorted(input_uv)))


classes = (SUO_Settings,)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.suo_settings = PointerProperty(type=SUO_Settings)
    bpy.types.Object.suo_input_uv = StringProperty(name="入力UV", default="",
                                                   description="空欄はアクティブUV")


def unregister():
    del bpy.types.Object.suo_input_uv
    del bpy.types.Scene.suo_settings
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
