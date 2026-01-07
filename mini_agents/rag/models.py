'''
Description: RAG数据模型定义
Author: zyq
Date: 2026-01-05 11:43:32
LastEditors: zyq
LastEditTime: 2026-01-07 09:29:19
'''

from __future__ import annotations

import hashlib
import uuid
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class ChunkMetadata(BaseModel):
    source_path: Optional[str] = None
    title: Optional[str] = None
    section: Optional[str] = None
    summary: Optional[str] = None
    keywords: List[str] = []
    potential_questions: List[str] = []
    tags: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    extra: Dict[str, str] = {}


class DocumentChunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str = ""
    chunk_index: int = 0
    content: str
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)
    embedding: Optional[List[float]] = None
    md5: str = ""

    def ensure_md5(self):
        if not self.md5:
            self.md5 = _md5(self.content)
        return self.md5


class Query(BaseModel):
    text: str
    biz_id: Optional[str] = None
    biz_name: Optional[str] = None
    collection: Optional[str] = None
    filters: Dict[str, str] = {}
    top_k: int = 5
    fetch_k: int = 20
    enable_mqe: bool = True
    enable_hyde: bool = False
    enable_rerank: bool = False
    rerank_type: str = "llm"  # llm|model

    def collection_name(self, prefix: str) -> str:
        if self.collection:
            return self.collection
        biz = self.biz_id or self.biz_name or "default"
        return f"{prefix}{biz}"


class QueryCandidate(BaseModel):
    query_text: str = ""
    source: str = "user"  # user|mqe|hyde
    score: float = 1.0


class RetrievalHit(BaseModel):
    chunk: DocumentChunk
    score: float
    reason: str = ""
    source_query: QueryCandidate = Field(default_factory=QueryCandidate)


class RAGResult(BaseModel):
    hits: List[RetrievalHit] = []
    trace: Dict[str, Any] = {}
