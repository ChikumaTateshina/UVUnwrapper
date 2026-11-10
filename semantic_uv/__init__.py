"""Semantic UV Organizer — 機械学習を用いない Blender 用 UV 展開・整理 Extension。

純粋な解析・配置モジュール (core / analysis / seams / unwrap / partial /
layout / validation) は Blender なしで import できる。Blender 依存部
(blender / persistence / ui) は register() 時にのみ読み込む。
"""

try:
    import bpy  # noqa: F401
    _HAS_BPY = True
except ImportError:  # 単体テスト環境
    _HAS_BPY = False


def register():
    from .blender import compatibility
    from .ui import properties, operators, panels, overlays
    compatibility.register()
    properties.register()
    operators.register()
    panels.register()
    overlays.register()


def unregister():
    from .blender import compatibility
    from .ui import properties, operators, panels, overlays
    overlays.unregister()
    panels.unregister()
    operators.unregister()
    properties.unregister()
    compatibility.unregister()
