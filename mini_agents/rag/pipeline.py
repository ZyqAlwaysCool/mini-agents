'''
Description: RAG处理管道
Author: zyq
Date: 2026-01-05 14:51:03
LastEditors: zyq
LastEditTime: 2026-01-07 09:40:04
'''
from __future__ import annotations

from typing import Optional, Dict, Any, List
from loguru import logger
import uuid

from .models import Query, RAGResult
from .data_process.loader import DocumentLoader
from .data_process.chunker import MarkdownTextChunker
from .data_process.metadata import MetadataEnricher
from .data_process.embedder import RemoteEmbedder
from .data_process.indexer import Indexer
from .retrieval.expander import QueryExpander
from .retrieval.retriever import Retriever
from .retrieval.reranker import RemoteReranker, BaseReranker
from .retrieval.aggregator import Aggregator
from .store.base import BaseVectorStore
from .store.memory_store import InMemoryVectorStore
from ..core.config import RAGConfig
from ..core.llm import BaseLLMClient
from ..core.exceptions import RAGException


class RAGPipeline:
    def __init__(
        self,
        rag_config: Optional[RAGConfig] = None,
        store: BaseVectorStore | None = None,
        embedder: RemoteEmbedder | None = None,
        llm_client: BaseLLMClient | None = None,
        reranker: BaseReranker | None = None,
    ):
        self.cfg = rag_config or RAGConfig.from_env()
        self.loader = DocumentLoader()
        self.chunker = MarkdownTextChunker(
            chunk_size=self.cfg.chunk_size,
            chunk_overlap=self.cfg.chunk_overlap,
        )
        self.embedder = embedder or self._build_embedder()
        self.store = store or InMemoryVectorStore()
        self.indexer = Indexer(self.store, self.embedder, vector_size=self.embedder.dim)
        self.metadata_enricher = MetadataEnricher(self.cfg, llm_client if self.cfg.metadata_mode == "llm" else None)
        self.expander = QueryExpander(self.cfg, llm_client if llm_client else None) # 查询扩展
        self.retriever = Retriever(self.store, self.embedder)
        self.aggregator = Aggregator()
        self.reranker = reranker if reranker else None

    def ingest(self, path: str, biz_name: Optional[str] = None, doc_id: Optional[str] = None) -> Dict[str, Any]:
        text, base_meta = self.loader.load(path)
        col = self.cfg.collection_name(biz_name=biz_name)
        chunks = self.chunker.chunk(text, base_meta, self.cfg, doc_id=doc_id or str(uuid.uuid4()))
        chunks = self.metadata_enricher.enrich(chunks)
        stats = self.indexer.index(col, chunks)
        return {"collection": col, **stats}

    def retrieve(self, query: Query) -> RAGResult:
        col = query.collection_name(self.cfg.collection_prefix)
        candidates = self.expander.expand(query)
        hits = self.retriever.retrieve(col, query, candidates, with_vectors=False)
        trace = {"collection": col, "candidates": len(candidates), "raw_hits": len(hits)}

        # rerank
        if query.enable_rerank:
            if self.reranker is None:
                raise RAGException("RAG rerank 模型未注入")
            hits = self.reranker.rerank(query.text, hits)
            trace["rerank"] = query.rerank_type

        top = self.aggregator.aggregate(hits, top_k=query.top_k or self.cfg.top_k)
        trace["returned"] = len(top)
        return RAGResult(hits=top, trace=trace)

    def _build_embedder(self) -> RemoteEmbedder:
        if not self.cfg.embedding_model or not self.cfg.embedding_base_url:
            raise RAGException("RAG embedding模型配置缺失")
        return RemoteEmbedder(
            model=self.cfg.embedding_model,
            api_key=self.cfg.embedding_api_key,
            base_url=self.cfg.embedding_base_url,
            dim=self.cfg.embedding_dim,
        )
