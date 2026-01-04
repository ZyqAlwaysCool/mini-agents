"""
嵌入向量接口定义
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class BaseEmbedder(ABC):
    """向量生成器抽象"""

    name: str = "base"
    dim: int = 0

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """批量生成向量"""
        raise NotImplementedError
