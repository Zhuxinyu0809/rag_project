"""
生成器模块

提供各种LLM答案生成方法
"""

from .api_generator import APIGenerator

# 未来可以添加：
# from .local_generator import LocalGenerator

__all__ = [
    'APIGenerator',
    # 'LocalGenerator',
]