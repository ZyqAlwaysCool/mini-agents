'''
Description: 向量召回器: 对候选查询依次检索并合并结果
Author: zyq
Date: 2026-01-05 14:48:58
LastEditors: zyq
LastEditTime: 2026-01-06 16:37:24
'''

from __future__ import annotations

from typing import List, Optional
from loguru import logger

from ..models import Query, QueryCandidate, RetrievalHit
from ..store.base import BaseVectorStore
from ..data_process.embedder import RemoteEmbedder


class Retriever:
    def __init__(self, store: BaseVectorStore, embedder: RemoteEmbedder):
        self.store = store
        self.embedder = embedder

    def retrieve(self, collection: str, query: Query, candidates: List[QueryCandidate], with_vectors: Optional[bool] = False) -> List[RetrievalHit]:
        hits: List[RetrievalHit] = []
        for cand in candidates:
            vec = self.embedder.embed([cand.query_text])[0]
            filters = dict(query.filters or {})
            filters["_query_text"] = cand.query_text
            filters["_source"] = cand.source
            res = self.store.query(collection=collection, vector=vec, filters=filters, limit=query.fetch_k, with_vectors=with_vectors)
            hits.extend(res)
        logger.info(f"召回完成: 候选={len(candidates)}, 命中={len(hits)}")
        return hits
