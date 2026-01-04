"""
混合长期记忆存储，支持 pluggable backend（默认 sqlite），向量与文本检索
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any, Dict, List, Optional
from ..base import MemoryRecord, MemoryQuery, MemoryHit, BaseMemoryStore
from ...core.exceptions import MemoryException
from ..embedder.base import BaseEmbedder
from ..backend.base import BaseBackend
from ..backend.sqlite_backend import SQLiteBackend


class HybridLongTermMemoryStore(BaseMemoryStore):
    """长期+语义混合记忆存储，后端可替换"""

    def __init__(
        self,
        backend: Optional[BaseBackend] = None,
        db_path: str = "data/memory/hybrid.db",
        embedder: Optional[BaseEmbedder] = None,
        top_k: int = 8,
        decay_lambda: float = 0.05,
        min_score: float = 0.3,
    ):
        self.backend = backend or SQLiteBackend(db_path)
        self.embedder = embedder
        self.top_k = top_k
        self.decay_lambda = decay_lambda
        self.min_score = min_score

    def add(self, record: MemoryRecord) -> str:
        emb = record.embedding
        if emb is None and self.embedder:
            emb = self.embedder.embed([record.content])[0]
            record.embedding = emb
        # 去重：同 user_id + type + content 不重复写入
        user_id = record.metadata.get("user_id") if record.metadata else None
        try:
            if self.backend.exists(user_id=user_id, mtype=record.type, content=record.content):
                return record.id
        except Exception:
            # 去重检查失败不阻塞正常写入
            pass
        self.backend.insert(record)
        return record.id

    def search(self, query: MemoryQuery) -> List[MemoryHit]:
        now = datetime.utcnow()
        rows = self.backend.query_all(query.type_scope, user_id=query.user_id, filters=query.filters)
        vector_hits: List[MemoryHit] = []
        text_hits: List[MemoryHit] = []
        q_emb: Optional[List[float]] = None
        if self.embedder and query.text:
            q_emb = self.embedder.embed([query.text])[0]
        for row in rows:
            rec = self._row_to_record(row)
            rec_score = self._decay_score(rec, now)
            if rec.expire_at and rec.expire_at <= now:
                continue
            if row.get("embedding") and q_emb:
                # 向量相似度+衰减得分计算
                emb = json.loads(row["embedding"])
                sim = self._cosine_sim(q_emb, emb)
                score = sim * 0.7 + rec_score * 0.3
                vector_hits.append(MemoryHit(record=rec, score=score, source_type=rec.type, reason=f"相似度={sim:.2f}"))
            else:
                # 兜底的文本匹配
                text_score = self._text_score(query.text, rec.content)
                score = text_score * 0.6 + rec_score * 0.4
                text_hits.append(MemoryHit(record=rec, score=score, source_type=rec.type, reason=f"文本匹配={text_score:.2f}"))
            rec.access_count += 1
            rec.last_accessed_at = now
            self.backend.update_access(rec.id, rec.access_count, rec.last_accessed_at)
        vector_hits.sort(key=lambda h: h.score, reverse=True)
        text_hits.sort(key=lambda h: h.score, reverse=True)
        return self._merge_hits(vector_hits, text_hits, query.top_k)

    def delete(self, record_id: str) -> None:
        self.backend.delete(record_id)

    def clear(self, user_id: Optional[str] = None) -> None:
        self.backend.clear(user_id)

    def forget(self, now: datetime) -> List[str]:
        removed: List[str] = []
        rows = self.backend.query_all()
        for row in rows:
            rec = self._row_to_record(row)
            if rec.expire_at and rec.expire_at <= now:
                self.delete(rec.id)
                removed.append(rec.id)
                continue
            decayed = self._decay_score(rec, now)
            if decayed < self.min_score:
                self.delete(rec.id)
                removed.append(rec.id)
        return removed

    def stats(self) -> Dict[str, Any]:
        rows = self.backend.query_all()
        return {"count": len(rows)}

    def _row_to_record(self, row: Dict[str, Any]) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            type=row["type"],
            content=row["content"],
            metadata=json.loads(row["metadata"] or "{}"),
            importance=row["importance"],
            score=row["score"],
            access_count=row["access_count"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_accessed_at=datetime.fromisoformat(row["last_accessed_at"]) if row.get("last_accessed_at") else None,
            expire_at=datetime.fromisoformat(row["expire_at"]) if row.get("expire_at") else None,
            embedding=json.loads(row["embedding"]) if row.get("embedding") else None,
            tags=json.loads(row["tags"] or "[]") if row.get("tags") is not None else [],
        )

    def _decay_score(self, rec: MemoryRecord, now: datetime) -> float:
        age_days = max((now - rec.created_at).total_seconds() / 86400.0, 0.0)
        return rec.score * math.exp(-self.decay_lambda * age_days) * (0.5 + rec.importance)

    def _cosine_sim(self, a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _text_score(self, query_text: str, target: str) -> float:
        if not query_text:
            return 0.0
        q_tokens = set(query_text.lower().split())
        t_tokens = set(target.lower().split())
        if not q_tokens or not t_tokens:
            return 0.0
        inter = len(q_tokens & t_tokens)
        return inter / len(q_tokens)

    def _merge_hits(self, vec_hits: List[MemoryHit], text_hits: List[MemoryHit], top_k: int) -> List[MemoryHit]:
        fused: Dict[str, MemoryHit] = {}
        rrf_k = 60
        for hits in (vec_hits, text_hits):
            for rank, hit in enumerate(hits):
                score = 1.0 / (rrf_k + rank + 1)
                if hit.record.id in fused:
                    fused[hit.record.id].score += score
                    fused[hit.record.id].reason += f";排名{rank}"
                else:
                    fused[hit.record.id] = MemoryHit(
                        record=hit.record,
                        score=score,
                        source_type=hit.source_type,
                        reason=hit.reason + f";排名{rank}",
                    )
        merged = list(fused.values())
        merged.sort(key=lambda h: h.score, reverse=True)
        return merged[:top_k]
