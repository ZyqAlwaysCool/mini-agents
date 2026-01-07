'''
Description: 将数据块向量化后写入向量库
Author: zyq
Date: 2026-01-05 14:48:19
LastEditors: zyq
LastEditTime: 2026-01-06 09:51:51
'''
from __future__ import annotations

from typing import List
from loguru import logger

from ..models import DocumentChunk
from ..store.base import BaseVectorStore
from .embedder import RemoteEmbedder


class Indexer:
    def __init__(self, store: BaseVectorStore, embedder: RemoteEmbedder, vector_size: int):
        self.store = store
        self.embedder = embedder
        self.vector_size = vector_size

    def index(self, collection: str, chunks: List[DocumentChunk]) -> dict:
        if not chunks:
            return {"written": 0}
        self.store.ensure_collection(collection, self.vector_size)
        texts = [c.content for c in chunks]
        vectors = self.embedder.embed(texts)
        for ck, vec in zip(chunks, vectors):
            ck.embedding = vec
            ck.ensure_md5()  # 稳定内容哈希用于去重/覆盖
        self.store.upsert(collection, chunks)
        logger.info(f"索引完成: collection={collection}, 条数={len(chunks)}")
        return {"written": len(chunks)}
