'''
Description: 向量存储基类
Author: zyq
Date: 2026-01-05 14:47:00
LastEditors: zyq
LastEditTime: 2026-01-07 09:28:07
'''

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ..models import DocumentChunk, RetrievalHit


class BaseVectorStore(ABC):
    @abstractmethod
    def ensure_collection(self, name: str, vector_size: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def upsert(self, collection: str, chunks: List[DocumentChunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(self, collection: str, vector: List[float], filters: Dict[str, Any], limit: int) -> List[RetrievalHit]:
        raise NotImplementedError

    @abstractmethod
    def clear(self, collection: str) -> None:
        raise NotImplementedError
