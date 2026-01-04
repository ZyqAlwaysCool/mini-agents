'''
Description: 短期记忆存储, 依靠内存和ttl控制
Author: zyq
Date: 2025-12-31 10:51:25
LastEditors: zyq
LastEditTime: 2025-12-31 17:09:05
'''

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from typing import Deque, Dict, List, Optional
from rank_bm25 import BM25Okapi
from ..base import MemoryRecord, MemoryQuery, MemoryHit, BaseMemoryStore
from ...core.exceptions import MemoryException


class SessionMemoryStore(BaseMemoryStore):
    def __init__(self, max_items: int = 200, ttl_seconds: int = 3600):
        self.max_items = max_items # 记忆容量
        self.ttl_seconds = ttl_seconds # 记忆时效ttl, <=0代表不过期
        self._records: Deque[MemoryRecord] = deque()

    def add(self, record: MemoryRecord) -> str:
        self._cleanup_expired(datetime.utcnow())
        if len(self._records) >= self.max_items:
            self._records.popleft()
        self._records.append(record)
        return record.id

    def search(self, query: MemoryQuery) -> List[MemoryHit]:
        now = datetime.utcnow()
        self._cleanup_expired(now)
        hits: List[MemoryHit] = []

        # BM25，用于计算权重得分
        corpus = [rec.content for rec in self._records]
        bm25 = BM25Okapi([c.split() for c in corpus]) if corpus else None
        bm25_scores: List[float] = bm25.get_scores(query.text.split()) if bm25 and query.text else [0.0] * len(self._records)
        query_tokens = query.text.split() if query.text else []

        for idx, rec in enumerate(list(self._records)):
            if query.type_scope and rec.type not in query.type_scope:
                continue
            if not self._match_filters(rec, query):
                continue
            sim_raw = bm25_scores[idx] if len(bm25_scores) > idx else 0.0
            score = self._calc_score(rec, now, sim_raw, query_tokens)
            rec.access_count += 1
            rec.last_accessed_at = now
            hits.append(MemoryHit(record=rec, score=score, source_type=rec.type, reason="短期记忆"))
        # 直接按得分排序
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[: query.top_k]

    def delete(self, record_id: str) -> None:
        for idx, rec in enumerate(self._records):
            if rec.id == record_id:
                del self._records[idx]
                return
        raise MemoryException(f"未找到记录 {record_id}")

    def clear(self, user_id: Optional[str] = None) -> None:
        if user_id is None:
            self._records.clear()
            return
        self._records = deque([rec for rec in self._records if rec.metadata.get("user_id") != user_id])

    def forget(self, now: datetime) -> List[str]:
        removed_ids: List[str] = []
        if self.ttl_seconds > 0:
            valid_window = now - timedelta(seconds=self.ttl_seconds)
            kept = deque()
            while self._records:
                rec = self._records.popleft()
                # 检查哪些记忆处于过期状态
                if rec.expire_at and rec.expire_at <= now:
                    removed_ids.append(rec.id)
                    continue
                if rec.created_at < valid_window:
                    removed_ids.append(rec.id)
                    continue
                kept.append(rec)
            self._records = kept
        else:
            # 仅处理 expire_at
            kept = deque()
            while self._records:
                rec = self._records.popleft()
                if rec.expire_at and rec.expire_at <= now:
                    removed_ids.append(rec.id)
                else:
                    kept.append(rec)
            self._records = kept
        return removed_ids

    def stats(self) -> Dict[str, any]:
        return {"count": len(self._records), "max_items": self.max_items, "ttl_seconds": self.ttl_seconds}

    def _cleanup_expired(self, now: datetime) -> None:
        if self.ttl_seconds <= 0:
            return
        valid_window = now - timedelta(seconds=self.ttl_seconds)
        self._records = deque([rec for rec in self._records if rec.created_at >= valid_window])

    def _calc_score(self, rec: MemoryRecord, now: datetime, sim_raw: float, query_tokens: List[str]) -> float:
        """
        复合得分：w = a*Sim + b*Recency + r*Importance
        - Sim(相似性): BM25 相关度，查询为空时为0
        - Recency(新鲜度): 基于时间衰减 0~1，越新越高
        - Importance(重要性): 直接使用记录 importance
        默认权重（短期偏重新鲜度）：a=0.3, b=0.5, r=0.2
        """
        a, b, r = 0.3, 0.5, 0.2
        # BM25 原始分数归一化
        sim_norm = sim_raw / (sim_raw + 1.0) if sim_raw > 0 else 0.0
        # 简单关键词覆盖度：提高中文场景可用性
        overlap = 0.0
        if query_tokens:
            matched = sum(1 for tok in query_tokens if tok and tok in rec.content)
            overlap = matched / len(query_tokens)
        sim_total = max(sim_norm, overlap)  # 取较高者作为相关度

        age_hours = max((now - rec.created_at).total_seconds() / 3600.0, 0.0)
        recency = 0.99 ** age_hours  # 越新越接近1

        importance = rec.importance
        return a * sim_total + b * recency + r * importance

    def _match_filters(self, rec: MemoryRecord, query: MemoryQuery) -> bool:
        filters = query.filters or {}
        tags = filters.get("tags")
        if tags:
            if not set(tags).issubset(set(rec.tags or [])):
                return False
        role = filters.get("role")
        if role and rec.metadata.get("role") != role:
            return False
        user_id = query.user_id
        if user_id and rec.metadata.get("user_id") not in (None, user_id):
            return False
        return True
