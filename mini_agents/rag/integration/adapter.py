'''
Description: agent注入适配器
Author: zyq
Date: 2026-01-05 14:51:28
LastEditors: zyq
LastEditTime: 2026-01-07 10:20:19
'''

from __future__ import annotations

from typing import List
from ..models import RetrievalHit, Query
from ..pipeline import RAGPipeline


class RAGPromptAdapter:
    @staticmethod
    def build_context(hits: List[RetrievalHit]) -> str:
        if not hits:
            return ""
        lines = ["[RAG_START]"]
        for h in hits:
            summary = h.chunk.metadata.summary or h.chunk.content
            lines.append(f"- 摘要: {summary} | 分数={h.score:.2f} | 来源={h.chunk.metadata.source_path or h.chunk.doc_id}")
        lines.append("[RAG_END]")
        return "\n".join(lines)


def build_rag_tool(pipeline: RAGPipeline, name: str = "rag_query", description: str = "基于业务知识库的检索工具"):
    """可选工具适配器：显式注册时才暴露给 Agent"""
    from ...tools.base import tool_register
    if not name.startswith("rag_query"):
        name = f"rag_query_{name}"

    @tool_register(name=name, description=description, override=True)
    def rag_query(query: str, biz_id: str = "default", top_k: int = 5):
        """
        :param query: 用户查询文本
        :param biz_id: 业务标识，用于选择 collection
        :param top_k: 返回条数
        """
        q = Query(text=query, biz_id=biz_id, top_k=top_k, fetch_k=max(top_k * 2, 10))
        res = pipeline.retrieve(q)
        rag_context = RAGPromptAdapter.build_context(res.hits)
        return {
            "trace": res.trace,
            "rag_context": rag_context,
            "results": [
                {
                    "content": h.chunk.content,
                    "summary": h.chunk.metadata.summary,
                    "score": h.score,
                    "source": h.chunk.metadata.source_path or h.chunk.doc_id,
                }
                for h in res.hits
            ],
        }

    return rag_query
