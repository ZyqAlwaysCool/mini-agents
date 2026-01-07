'''
Description: 文本分块器, 默认使用 MarkdownTextSplitter, 预留扩展
Author: zyq
Date: 2026-01-05 14:45:33
LastEditors: zyq
LastEditTime: 2026-01-06 09:35:42
'''
from __future__ import annotations

from typing import List, Dict
from loguru import logger
from langchain_text_splitters import MarkdownTextSplitter

from ..models import DocumentChunk, ChunkMetadata
from ...core.config import RAGConfig


class BaseChunker:
    """分块基类，后续接入其他分块库时只需继承并实现 chunk"""

    def chunk(self, text: str, base_meta: Dict[str, str], cfg: RAGConfig, doc_id: str = "") -> List[DocumentChunk]:
        raise NotImplementedError


class MarkdownTextChunker(BaseChunker):
    """基于 MarkdownTextSplitter 的分块器，预期输入一定是markdown"""

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def chunk(self, text: str, base_meta: Dict[str, str], cfg: RAGConfig, doc_id: str = "") -> List[DocumentChunk]:
        pieces = self._splitter.split_text(text)
        total = len(pieces)
        logger.info(f"分块完成: {total} 块")
        results: List[DocumentChunk] = []
        for idx, content in enumerate(pieces):
            meta = self._build_metadata(base_meta)
            chunk = DocumentChunk(
                doc_id=doc_id or base_meta.get("source_path", ""),
                chunk_index=idx,
                content=content.strip(),
                metadata=meta,
            )
            chunk.ensure_md5() # chunk块内容的md5值, 方便后续去重或更新
            results.append(chunk)
        return results

    def _build_metadata(self, base_meta: Dict[str, str]) -> ChunkMetadata:
        extra_keys = {"source_path", "title", "section"}
        extra = {k: v for k, v in (base_meta or {}).items() if k not in extra_keys}
        return ChunkMetadata(
            source_path=base_meta.get("source_path"),
            title=base_meta.get("title"),
            section=base_meta.get("section"),
            extra=extra,
        )
