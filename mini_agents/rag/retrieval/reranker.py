'''
Description: reranker
Author: zyq
Date: 2026-01-06 17:45:05
LastEditors: zyq
LastEditTime: 2026-01-07 09:22:24
'''

from __future__ import annotations
import json
from typing import List, Dict, Optional
from loguru import logger
import requests

from ..models import RetrievalHit
from ...core.llm import BaseLLMClient
from ...core.message import Message


class BaseReranker:
    def rerank(self, query_text: str, hits: List[RetrievalHit]) -> List[RetrievalHit]:
        raise NotImplementedError


class RemoteReranker(BaseReranker):
    """
    - mode=llm：使用 LLM 打分，提示词约束输出 JSON
    - mode=default：调用兼容 OpenAI /v1/rerank 模型（如 bge-reranker 系列）
    """

    def __init__(
        self,
        mode: Optional[str] = "default",
        llm_client: Optional[BaseLLMClient] = None,
        rerank_model: Optional[str] = None,
        rerank_base_url: Optional[str] = None,
        rerank_api_key: Optional[str] = None,
    ):
        self.mode = mode
        self.llm_client = llm_client
        self.rerank_model = rerank_model
        self.rerank_base_url = rerank_base_url
        self.rerank_api_key = rerank_api_key
        self._session: Optional[requests.Session] = None
        self._rerank_url: Optional[str] = None
        if mode == "default" and rerank_model and rerank_base_url:
            base = rerank_base_url.rstrip("/")
            self._rerank_url = base if base.endswith("/rerank") else f"{base}/rerank"
            self._session = requests.Session()

    def rerank(self, query_text: str, hits: List[RetrievalHit]) -> List[RetrievalHit]:
        if not hits:
            return []
        if self.mode == "llm":
            return self._rerank_by_llm(query_text, hits)
        if self.mode == "default":
            return self._rerank_by_model(query_text, hits)
        return hits

    def _rerank_by_llm(self, query_text: str, hits: List[RetrievalHit]) -> List[RetrievalHit]:
        if not self.llm_client:
            return hits
        payload_lines = []
        for h in hits:
            content = h.chunk.content
            payload_lines.append(f'{{"chunk_id": "{h.chunk.id}", "text": "{content}"}}')
        prompt = (
            "你是一个专业的语义相关性评分助手。任务是：根据用户问题，对提供的每个文本片段打一个语义相关性分数（0到1之间的小数）。\n\n"
            "评分标准（严格遵守）：\n"
            "- 1.0：片段直接、完整地回答或高度匹配用户问题核心意图，包含关键事实或解决方案。\n"
            "- 0.8-0.9：片段高度相关，提供大部分所需信息或强力支持答案，但可能缺少少量细节。\n"
            "- 0.5-0.7：片段部分相关，涉及同一主题或包含部分有用信息，但不能独立解决问题。\n"
            "- 0.2-0.4：片段仅表面相关（如共享关键词），但语义上偏离或信息价值低。\n"
            "- 0.0-0.1：片段完全无关或无有用信息。\n"
            "评分时优先考虑语义匹配而非单纯关键词重叠。\n\n"
            "要求：\n"
            "- 每个片段必须返回其原始的 chunk_id（严格保持不变）。\n"
            "- score 为浮点数，保留最多2位小数（如 0.85）。\n"
            "- 仅输出一个合法的JSON数组，数组中每个元素为对象，只包含 \"chunk_id\" 和 \"score\" 两个字段。\n"
            "- 输出必须以[开头，以]结尾，不包含任何多余文字、说明、Markdown代码块或换行注释。\n\n"
            "示例输出格式（仅供参考，不输出此示例）：\n"
            '[{"chunk_id": "doc_001", "score": 0.95}, {"chunk_id": "doc_002", "score": 0.72}, '
            '{"chunk_id": "doc_003", "score": 0.31}, {"chunk_id": "doc_004", "score": 0.08}]\n\n'
            f"用户问题：{query_text.strip()}\n\n"
            "片段列表：\n"
            f"{','.join(payload_lines)}\n\n"  # 假设payload_lines每个元素形如 "chunk_id: xxx, content: yyy"
            "现在直接输出JSON数组："
        )
        raw = self.llm_client.invoke([Message(role="user", content=prompt)])
        scores = self._parse_scores(raw)
        for h in hits:
            new_score = scores.get(h.chunk.id)
            if new_score is not None:
                h.score = float(new_score)
                h.reason += f";rerank_llm={new_score}"
        hits.sort(key=lambda x: x.score, reverse=True)
        return hits

    def _rerank_by_model(self, query_text: str, hits: List[RetrievalHit]) -> List[RetrievalHit]:
        if not self._session or not self._rerank_url or not self.rerank_model:
            return hits
        docs = [h.chunk.content for h in hits]
        headers = {}
        if self.rerank_api_key:
            headers["Authorization"] = f"Bearer {self.rerank_api_key}"
        try:
            # 调用兼容 OpenAI 的 /rerank 接口，按返回分数重排
            resp = self._session.post(
                self._rerank_url,
                json={
                    "model": self.rerank_model,
                    "query": query_text,
                    "documents": docs,
                    "top_n": len(hits),
                },
                headers=headers,
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
            # 兼容远端返回 results/data 两种字段
            scores = self._parse_model_scores(data, hits)
            for h in hits:
                new_score = scores.get(h.chunk.id)
                if new_score is not None:
                    h.score = float(new_score)
                    h.reason += f";rerank_model={new_score}"
            hits.sort(key=lambda x: x.score, reverse=True)
        except Exception as e:
            logger.warning(f"远程 reranker 调用失败: {e}")
        return hits

    def _parse_scores(self, raw: str) -> Dict[str, float]:
        try:
            data = json.loads(raw)
        except Exception as e:
            logger.warning(f"rerank 解析失败: {e}")
            return {}
        scores: Dict[str, float] = {}
        for item in data: # data形如: [{"chunk_id":xxx, "score":xxx}, {...}, {...}]
            cid = item.get("chunk_id")
            sc = item.get("score")
            if cid is None or sc is None:
                continue
            scores[str(cid)] = float(sc)
        return scores

    def _parse_model_scores(self, resp, hits: List[RetrievalHit]) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        if isinstance(resp, dict):
            data = resp.get("results") or resp.get("data") or []
        else:
            data = getattr(resp, "results", None) or getattr(resp, "data", None) or []
        for item in data:
            idx = item.get("index") if isinstance(item, dict) else getattr(item, "index", None)
            sc = item.get("relevance_score") if isinstance(item, dict) else getattr(item, "relevance_score", None)
            if idx is None or sc is None:
                continue
            try:
                hit = hits[int(idx)]
            except Exception:
                continue
            scores[hit.chunk.id] = float(sc)
        return scores
