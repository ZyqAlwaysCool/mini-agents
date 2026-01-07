'''
Description: 元数据增强: 默认 LLM，失败或关闭时走规则降级
Author: zyq
Date: 2026-01-05 14:46:16
LastEditors: zyq
LastEditTime: 2026-01-06 10:58:10
'''

from __future__ import annotations

import json
import re
from typing import List
from loguru import logger

from ..models import DocumentChunk
from ...core.config import RAGConfig
from ...core.message import Message
from ...core.llm import BaseLLMClient
from ...core.exceptions import RAGException


class MetadataEnricher:
    def __init__(self, cfg: RAGConfig, llm_client: BaseLLMClient | None = None):
        self.cfg = cfg
        self.llm_client = llm_client

    def enrich(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        mode = self.cfg.metadata_mode
        enriched: List[DocumentChunk] = []

        if mode == "llm" and not self.llm_client:
            raise RAGException("LLM 模式下必须提供 LLM 客户端")
        
        for ck in chunks:
            try:
                if mode == "llm":
                    self._llm_enrich(ck)
                elif mode == "rule":
                    self._rule_enrich(ck)
            except Exception as e:
                logger.warning(f"元数据处理失败: {e} chunk: {ck}")
            enriched.append(ck)
        return enriched

    def _llm_enrich(self, chunk: DocumentChunk) -> None:
        prompt = (
            "你是一个专业的文本元数据提取助手。请仔细阅读以下文本，仅输出一个严格有效的JSON对象，"
            "必须包含且仅包含以下三个字段：\n"
            "- summary: string类型，中文总结文本核心内容，控制在150字以内，语言精炼、自然。\n"
            "- keywords: array类型，1-4个最能代表文本核心主题的中文关键词或短语（不要超过2个词的短语），用列表形式。\n"
            "- potential_questions: array类型，1-3个用户很可能基于这段文本想提问的、具体且有价值的中文问题（以问号结尾）。\n\n"
            "输出必须是合法的JSON格式（以{开头，以}结尾），字段名用双引号，值正确转义，不要有任何多余文字、说明、Markdown标记或换行注释。\n\n"
            "示例输出格式：\n"
            "{\n"
            '  "summary": "这是关于某技术的简要总结...",\n'
            '  "keywords": ["关键词1", "关键词2", "关键词3"],\n'
            '  "potential_questions": ["问题一？", "问题二？", "问题三？"]\n'
            "}\n\n"
            "如果文本内容空或无意义，则输出：\n"
            "{\n"
            '  "summary": "",\n'
            '  "keywords": [],\n'
            '  "potential_questions": []\n'
            "}\n\n"
            f"文本内容：\n{chunk.content.strip()}"
        )
        raw = self.llm_client.invoke([Message(role="user", content=prompt)])
        if not raw:
            raise ValueError("LLM 返回为空")
        data = json.loads(raw)
        meta = chunk.metadata
        meta.summary = str(data.get("summary", "")).strip()
        meta.keywords = [str(k) for k in data.get("keywords", [])][:4]
        meta.potential_questions = [str(q) for q in data.get("potential_questions", [])][:3]

    def _rule_enrich(self, chunk: DocumentChunk) -> None:
        text = chunk.content.strip()
        meta = chunk.metadata
        meta.summary = text[:80]
        meta.keywords = self._extract_keywords(text, limit=4)
        meta.potential_questions = []

    def _extract_keywords(self, text: str, limit: int = 4) -> List[str]:
        tokens = re.findall(r"[\w\u4e00-\u9fa5]{2,}", text)
        freq = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        sorted_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [t for t, _ in sorted_tokens[:limit]]
