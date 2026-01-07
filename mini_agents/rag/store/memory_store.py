'''
Description: 内存向量存储+余弦相似度判定
Author: zyq
Date: 2026-01-06 09:45:01
LastEditors: zyq
LastEditTime: 2026-01-07 09:28:22
'''

from __future__ import annotations

from typing import Any, Dict, List
import math

from .base import BaseVectorStore
from ..models import DocumentChunk, RetrievalHit, ChunkMetadata, QueryCandidate


class InMemoryVectorStore(BaseVectorStore):
    def __init__(self):
        self._data: Dict[str, List[DocumentChunk]] = {}

    def ensure_collection(self, name: str, vector_size: int) -> None:
        self._data.setdefault(name, [])

    def upsert(self, collection: str, chunks: List[DocumentChunk]) -> None:
        arr = self._data.setdefault(collection, [])
        arr.extend(chunks)

    def query(self, collection: str, vector: List[float], filters: Dict[str, Any], limit: int) -> List[RetrievalHit]:
        arr = self._data.get(collection, [])
        scored: List[RetrievalHit] = []
        for ck in arr:
            sim = self._cosine(vector, ck.embedding or [])
            scored.append(
                RetrievalHit(
                    chunk=ck,
                    score=sim,
                    reason=f"cosine={sim:.2f}",
                    source_query=QueryCandidate(query_text=filters.get("_query_text", ""), source=filters.get("_source", "user")),
                )
            )
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:limit]

    def clear(self, collection: str) -> None:
        self._data[collection] = []

    def _cosine(self, a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
