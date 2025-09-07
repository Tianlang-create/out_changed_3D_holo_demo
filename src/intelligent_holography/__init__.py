"""Intelligent-3D-Holography Core Package

提供实时 3D 全息图生成网络 (rtholo)、自适应光场调谐模块 (ALFT) 及常用工具函数。"""

from importlib import import_module as _imp

# 按需加载，避免硬依赖
_rtholo = _imp("rtholo")
_alft = _imp("alft")
_utils = _imp("utils")
_misc = _imp("intelligent_holography._misc")

rtholo = _rtholo.rtholo
AdaptiveLightFieldTuner = _alft.AdaptiveLightFieldTuner
noop = _misc.noop
Placeholder = _misc.Placeholder
Fibonacci = _misc.Fibonacci


__all__ = [
    "rtholo",
    "AdaptiveLightFieldTuner",
    "noop",
    "Placeholder",
    "Fibonacci",
]