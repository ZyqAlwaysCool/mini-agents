'''
Description: qdrant向量存储
Author: zyq
Date: 2026-01-05 14:47:43
LastEditors: zyq
LastEditTime: 2026-01-07 09:28:50
'''

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from loguru import logger

from .base import BaseVectorStore
from ..models import DocumentChunk, ChunkMetadata, RetrievalHit, QueryCandidate

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except Exception as e:  # pragma: no cover - 运行时缺依赖才触发
    QdrantClient = None  # type: ignore
    qmodels = None  # type: ignore
    _import_err = e
else:
    _import_err = None


class QdrantVectorStore(BaseVectorStore):
    def __init__(self, url: str, api_key: Optional[str], min_score: float = 0.3):
        if _import_err:
            raise ImportError(f"缺少 qdrant-client 依赖: {_import_err}")
        if not url:
            raise ValueError("Qdrant 初始化失败：url 为空")
        self.client = QdrantClient(url=url, api_key=api_key)
        self.min_score = min_score

    def ensure_collection(self, name: str, vector_size: int) -> None:
        try:
            self.client.get_collection(name)
        except Exception:
            logger.info(f"创建/重建 Qdrant 集合: {name}")
            self.client.recreate_collection(
                collection_name=name,
                vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
                on_disk_payload=True,
            )

    def upsert(self, collection: str, chunks: List[DocumentChunk]) -> None:
        points = []
        for ck in chunks:
            if ck.embedding is None:
                raise ValueError("缺少向量，无法写入 Qdrant")
            payload = self._build_payload(ck)
            point_id = ck.md5 or ck.id
            points.append(qmodels.PointStruct(id=point_id, vector=ck.embedding, payload=payload))
        if not points:
            return
        self.client.upsert(collection_name=collection, points=points)

    def query(self, collection: str, vector: List[float], filters: Dict[str, Any], limit: int, with_vectors: Optional[bool] = False) -> List[RetrievalHit]:
        flt = self._build_filter(filters)
        res = self.client.search(
            collection_name=collection,
            query_vector=vector,
            query_filter=flt,
            limit=limit,
            with_vectors=with_vectors,
        )
        hits: List[RetrievalHit] = []
        now = datetime.utcnow()
        for scored in res:
            payload = scored.payload or {}
            ck = self._payload_to_chunk(payload, point_id=str(scored.id), vector=scored.vector if with_vectors else None)
            score = scored.score or 0.0
            if score < self.min_score:
                continue
            created_at_ts = payload.get("created_at_ts")
            if created_at_ts and created_at_ts <= 0:
                pass
            hits.append(
                RetrievalHit(
                    chunk=ck,
                    score=score,
                    reason=f"相似度={score:.2f}",
                    source_query=QueryCandidate(query_text=filters.get("_query_text", ""), source=filters.get("_source", "user")),
                )
            )
        return hits

    def clear(self, collection: str) -> None:
        self.client.delete(collection_name=collection, points_selector=qmodels.FilterSelector(filter=qmodels.Filter()))

    def _build_payload(self, ck: DocumentChunk) -> Dict[str, Any]:
        meta = ck.metadata
        return {
            "doc_id": ck.doc_id,
            "chunk_index": ck.chunk_index,
            "content": ck.content,
            "md5": ck.md5,
            "summary": meta.summary,
            "keywords": meta.keywords,
            "potential_questions": meta.potential_questions,
            "tags": meta.tags,
            "source_path": meta.source_path,
            "title": meta.title,
            "section": meta.section,
            "extra": meta.extra,
            "created_at": meta.created_at.isoformat(),
            "created_at_ts": meta.created_at.timestamp(),
        }

    def _payload_to_chunk(self, payload: Dict[str, Any], point_id: str = "", vector: Optional[List[float]] = None) -> DocumentChunk:
        meta = ChunkMetadata(
            summary=payload.get("summary"),
            keywords=payload.get("keywords") or [],
            potential_questions=payload.get("potential_questions") or [],
            tags=payload.get("tags") or [],
            source_path=payload.get("source_path"),
            title=payload.get("title"),
            section=payload.get("section"),
            extra=payload.get("extra") or {},
        )
        return DocumentChunk(
            id=point_id or str(payload.get("id") or payload.get("md5") or ""),
            doc_id=payload.get("doc_id", ""),
            chunk_index=payload.get("chunk_index", 0),
            content=payload.get("content", ""),
            metadata=meta,
            md5=payload.get("md5", ""),
            embedding=vector,
        )

    def _build_filter(self, filters: Dict[str, Any]):
        if not filters:
            return None
        must: List[Any] = []
        tags = filters.get("tags")
        if tags:
            must.append(qmodels.FieldCondition(key="tags", match=qmodels.MatchAny(any=tags)))
        created_after = filters.get("created_after")
        if created_after:
            must.append(qmodels.FieldCondition(key="created_at_ts", range=qmodels.Range(gte=created_after)))
        return qmodels.Filter(must=must) if must else None
