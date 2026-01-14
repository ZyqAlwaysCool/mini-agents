'''
Description: 记忆组件网关, 负责跨存储检索、融合排序、格式化与清理
Manager本身不做业务过滤/打分, 此部分内容由各store完成, 这里通过RRF融合跨类型记忆检索结果
Author: zyq
Date: 2025-12-31 10:53:14
LastEditors: zyq
LastEditTime: 2026-01-14 16:13:44
'''
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Sequence
from loguru import logger

from .base import MemoryRecord, MemoryQuery, MemoryHit, MemoryType, BaseMemoryStore
from ..core.exceptions import MemoryException
from .refiner import MemoryRefiner


class MemoryManager:
    def __init__(self, rrf_k: int = 60, default_top_k: int = 5, refiner: Optional[MemoryRefiner] = None):
        self.rrf_k = rrf_k # RRF融合排序平滑参数
        self.default_top_k = default_top_k # 检索默认返回条数
        self.refiner = refiner or MemoryRefiner() # 记忆精炼器
        self._type_store_map: Dict[MemoryType, BaseMemoryStore] = {} # session -> SessionMemoryStore，long_term/semantic -> HybridLongTermMemoryStore
        self._stores: List[BaseMemoryStore] = []

    def register_store(self, memory_types: Sequence[MemoryType], store: BaseMemoryStore) -> None:
        for mtype in memory_types:
            self._type_store_map[mtype] = store
        if store not in self._stores:
            self._stores.append(store)

    def add(self, record: MemoryRecord) -> str:
        store = self._type_store_map.get(record.type)
        if not store:
            raise MemoryException(f"未找到 {record.type} 对应的存储")
        return store.add(record)

    def batch_add(self, records: List[MemoryRecord]) -> List[str]:
        ids: List[str] = []
        for rec in records:
            ids.append(self.add(rec))
        return ids # 保存的记录id

    def search(self, query: MemoryQuery) -> List[MemoryHit]:
        stores = self._select_stores(query)
        all_hits: List[List[MemoryHit]] = []
        for store in stores:
            try:
                hits = store.search(query)
                all_hits.append(hits)
            except Exception as e:
                logger.warning(f"store 检索异常: {e}")
        fused = self._rrf_merge(all_hits, query.top_k or self.default_top_k)
        return fused

    def build_memory_context(self, query: MemoryQuery, top_k: Optional[int] = None) -> str:
        """构造长期记忆上下文结构"""
        query.top_k = top_k or query.top_k or self.default_top_k
        hits = self.search(query)
        if not hits:
            return ""
        lines = ["[MEMORY_START]"]
        for hit in hits:
            ts = hit.record.created_at.isoformat() if hit.record.created_at else ""
            lines.append(f"- {hit.record.content} (时间={ts}, 来源={hit.source_type}, 分值={hit.score:.2f})")
        lines.append("[MEMORY_END]")
        logger.debug(f"生成记忆上下文: {lines}")
        return "\n".join(lines)

    def forget_all(self, now: Optional[datetime] = None) -> Dict[str, List[str]]:
        now = now or datetime.utcnow()
        removed: Dict[str, List[str]] = {}
        for store in self._stores:
            try:
                ids = store.forget(now)
                if ids:
                    removed[store.__class__.__name__] = ids
            except Exception as e:
                logger.warning(f"store 清理异常: {e}")
        return removed

    def refine_async(
        self,
        records: List[MemoryRecord],
        target_type: MemoryType = "long_term",
        user_id: Optional[str] = None,
        llm_client=None,
    ):
        try:
            logger.info(f"refiner start, target={target_type}, user_id={user_id}, records={len(records)}")
            return self.refiner.run_async(records, self, target_type, user_id, llm_client)
        except Exception as e:
            logger.warning(f"refiner 调用失败: {e}")
            return None

    def _select_stores(self, query: MemoryQuery) -> List[BaseMemoryStore]:
        if query.type_scope:
            stores = []
            seen = set()
            for t in query.type_scope:
                store = self._type_store_map.get(t)
                if store and id(store) not in seen:
                    stores.append(store)
                    seen.add(id(store))
            return stores
        return self._stores

    def _rrf_merge(self, list_hits: List[List[MemoryHit]], top_k: int) -> List[MemoryHit]:
        """对每个命中按排名加分 1/(rrf_k+rank+1)，累加同 id 的分数与 reason，最后降序取 top_k"""
        fused: Dict[str, MemoryHit] = {}
        for hits in list_hits:
            for rank, hit in enumerate(hits):
                score = 1.0 / (self.rrf_k + rank + 1)
                if hit.record.id in fused:
                    fused_hit = fused[hit.record.id]
                    fused_hit.score += score
                    fused_hit.reason += f";排名{rank}"
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
