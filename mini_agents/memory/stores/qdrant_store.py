'''
Description: Qdrant 向量存储实现，可对接本地或云端 Qdrant 服务, 长期/语义向量存储方案
Author: zyq
Date: 2025-12-31 11:05:48
LastEditors: zyq
LastEditTime: 2026-01-04 09:26:11
'''
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Any
import math

from ..base import MemoryRecord, MemoryQuery, MemoryHit, BaseMemoryStore, MemoryType
from ...core.exceptions import MemoryException
from ..embedder.base import BaseEmbedder

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except Exception as e:
    QdrantClient = None  # type: ignore
    qmodels = None  # type: ignore
    _import_err = e
else:
    _import_err = None


class QdrantStore(BaseMemoryStore):
    def __init__(
        self,
        url: str,
        api_key: Optional[str],
        collection: str = "agent_memory",
        embedder: Optional[BaseEmbedder] = None,
        vector_size: Optional[int] = None,
        prefer_grpc: bool = False,
        min_score: float = 0.3,
    ):
        if _import_err:
            raise MemoryException(f"缺少 qdrant-client 依赖: {_import_err}")
        if not url:
            raise MemoryException("QdrantStore 初始化失败：url 为空")
        self.url = url
        self.api_key = api_key
        self.collection = collection
        self.embedder = embedder
        self.vector_size = vector_size or (embedder.dim if embedder else None)
        if not self.vector_size:
            raise MemoryException("QdrantStore 初始化失败：未提供向量维度")
        self.min_score = min_score
        self.client = QdrantClient(url=self.url, api_key=self.api_key, prefer_grpc=prefer_grpc)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            self.client.get_collection(self.collection)
        except Exception:
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=qmodels.VectorParams(size=self.vector_size, distance=qmodels.Distance.COSINE),
                on_disk_payload=True,
            )

    def add(self, record: MemoryRecord) -> str:
        vector = record.embedding
        if vector is None:
            if not self.embedder:
                raise MemoryException("QdrantStore 写入失败：缺少嵌入向量与 embedder")
            vector = self.embedder.embed([record.content])[0]
        payload = self._build_payload(record)
        self.client.upsert(
            collection_name=self.collection,
            points=[
                qmodels.PointStruct(
                    id=record.id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )
        return record.id

    def search(self, query: MemoryQuery) -> List[MemoryHit]:
        if not self.embedder:
            raise MemoryException("QdrantStore 检索失败：未配置 embedder")
        if not query.text:
            return []
        q_vector = self.embedder.embed([query.text])[0]
        flt = self._build_filter(query)
        res = self.client.search(
            collection_name=self.collection,
            query_vector=q_vector,
            query_filter=flt,
            limit=query.top_k,
        )
        hits: List[MemoryHit] = []
        now = datetime.utcnow()
        for scored in res:
            payload = scored.payload or {}
            rec = self._payload_to_record(payload)
            if rec.expire_at and rec.expire_at <= now:
                continue
            score = scored.score or 0.0
            hits.append(MemoryHit(record=rec, score=score, source_type=rec.type, reason=f"相似度={score:.2f}"))
        return hits

    def delete(self, record_id: str) -> None:
        self.client.delete(collection_name=self.collection, points_selector=qmodels.PointIdsList(points=[record_id]))

    def clear(self, user_id: Optional[str] = None) -> None:
        if user_id:
            flt = qmodels.Filter(must=[qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=user_id))])
            self.client.delete(collection_name=self.collection, points_selector=qmodels.FilterSelector(filter=flt))
        else:
            self.client.delete(collection_name=self.collection, points_selector=qmodels.FilterSelector(filter=qmodels.Filter()))

    def forget(self, now: datetime) -> List[str]:
        removed: List[str] = []
        # 按 expire_at 清理
        flt = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="expire_at_ts",
                    range=qmodels.Range(
                        lte=now.timestamp(),
                    ),
                )
            ]
        )
        points, _ = self.client.scroll(collection_name=self.collection, scroll_filter=flt, limit=100)
        if points:
            ids = [p.id for p in points]
            self.client.delete(collection_name=self.collection, points_selector=qmodels.PointIdsList(points=ids))
            removed.extend(ids)
        return removed

    def stats(self) -> Dict[str, Any]:
        info = self.client.get_collection(self.collection)
        return {"collection": self.collection, "vectors_count": info.points_count if hasattr(info, "points_count") else None}

    def _build_payload(self, record: MemoryRecord) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "type": record.type,
            "content": record.content,
            "metadata": record.metadata or {},
            "importance": record.importance,
            "score": record.score,
            "access_count": record.access_count,
            "created_at": record.created_at.isoformat(),
            "created_at_ts": record.created_at.timestamp(),
            "tags": record.tags or [],
        }
        if record.metadata:
            payload["user_id"] = record.metadata.get("user_id")
            payload["role"] = record.metadata.get("role")
        if record.last_accessed_at:
            payload["last_accessed_at"] = record.last_accessed_at.isoformat()
        if record.expire_at:
            payload["expire_at"] = record.expire_at.isoformat()
            payload["expire_at_ts"] = record.expire_at.timestamp()
        return payload

    def _payload_to_record(self, payload: Dict[str, Any]) -> MemoryRecord:
        return MemoryRecord(
            type=payload.get("type", "long_term"),
            content=payload.get("content", ""),
            metadata=payload.get("metadata", {}),
            importance=payload.get("importance", 0.5),
            score=payload.get("score", 1.0),
            access_count=payload.get("access_count", 0),
            created_at=datetime.fromisoformat(payload.get("created_at")),
            last_accessed_at=datetime.fromisoformat(payload["last_accessed_at"]) if payload.get("last_accessed_at") else None,
            expire_at=datetime.fromisoformat(payload["expire_at"]) if payload.get("expire_at") else None,
            tags=payload.get("tags", []),
        )

    def _build_filter(self, query: MemoryQuery):
        must: List[Any] = []
        if query.user_id:
            must.append(qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=query.user_id)))
        if query.type_scope:
            must.append(qmodels.FieldCondition(key="type", match=qmodels.MatchAny(any=query.type_scope)))
        filters = query.filters or {}
        tags = filters.get("tags")
        if tags:
            must.append(qmodels.FieldCondition(key="tags", match=qmodels.MatchAny(any=tags)))
        created_after = filters.get("created_after")
        if created_after:
            must.append(
                qmodels.FieldCondition(
                    key="created_at_ts",
                    range=qmodels.Range(gte=created_after.timestamp()),
                )
            )
        if must:
            return qmodels.Filter(must=must)
        return None
