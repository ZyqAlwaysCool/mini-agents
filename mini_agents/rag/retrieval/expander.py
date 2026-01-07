'''
Description: 查询扩展, 支持MQE/HyDE策略
Author: zyq
Date: 2026-01-05 14:48:41
LastEditors: zyq
LastEditTime: 2026-01-06 11:25:06
'''

from __future__ import annotations

import json
from typing import List
from loguru import logger

from ..models import Query, QueryCandidate
from ...core.llm import BaseLLMClient
from ...core.message import Message
from ...core.config import RAGConfig


class QueryExpander:
    def __init__(self, cfg: RAGConfig, llm_client: BaseLLMClient | None = None):
        self.cfg = cfg
        self.llm_client = llm_client

    def expand(self, query: Query) -> List[QueryCandidate]:
        candidates: List[QueryCandidate] = [QueryCandidate(query_text=query.text, source="user", score=1.0)]
        if not self.llm_client:
            return candidates
        if query.enable_mqe:
            candidates.extend(self._mqe(query.text))
        if query.enable_hyde:
            candidates.extend(self._hyde(query.text))
        return self._dedup(candidates)

    def _mqe(self, text: str) -> List[QueryCandidate]:
        prompt = (
            "你是一个专业的检索查询扩展助手。任务是：基于用户原始问题，生成2-4个语义等价或互补的查询改写，"
            "这些改写应有助于在向量数据库中检索到更多相关文档。\n\n"
            "要求：\n"
            "- 生成2-4个查询（不少于2个，不超过4个）。\n"
            "- 每个改写要用不同的表述方式，覆盖同义词、不同角度、潜在子意图或更具体/泛化的表达。\n"
            "- 查询语言保持自然、流畅，适合语义检索（避免纯关键词堆砌）。\n"
            "- 仅输出一个合法的JSON数组，数组中每个元素是一个对象，只包含\"query\"字段。\n"
            "- 输出必须以[开头，以]结尾，不包含任何多余文字、说明、Markdown代码块或换行注释。\n\n"
            "- 为每一个扩展查询打一个0-1的分数，分数越高，优先级越高，不得超过1.0，不得低于0.0，最多为小数点后两位。\n\n"
            "示例输出格式（仅供参考，不输出此示例）：\n"
            '[{"query": "Python装饰器如何实现单例模式", "score": 0.9}, {"query": "用Python decorator实现singleton的设计方法", "score": 0.8}, '
            '{"query": "Python中通过装饰器创建单例类的示例", "score": 0.7}, {"query": "单例模式在Python中的装饰器写法", "score": 0.75}]\n\n'
            f"用户原始问题：{text.strip()}\n"
            "现在直接输出JSON数组："
        )
        raw = self.llm_client.invoke([Message(role="user", content=prompt)])
        try:
            data = json.loads(raw)
        except Exception as e:
            logger.warning(f"MQE 解析失败: {e}")
            return []
        results: List[QueryCandidate] = []
        for item in data:
            q = str(item.get("query") or "").strip()
            s = float(item.get("score", 0.9))
            if q and 0.0 <= s <= 1.0:
                results.append(QueryCandidate(query_text=q, source="mqe", score=s))
        return results

    def _hyde(self, text: str) -> List[QueryCandidate]:
        prompt = (
            "你是一个专业的假设文档生成助手。任务是：基于用户问题，生成一段假设性的参考文档片段，作为可能的回答概要，用于HYDE策略下的向量检索。\n\n"
            "要求：\n"
            "- 文本风格：客观、信息密集，像真实知识库或文档中的事实描述，避免主观意见或对话语气。\n"
            "- 长度：控制在100-250字，确保覆盖问题核心，但不冗长。\n"
            "- 内容：聚焦于问题潜在答案的关键事实、步骤或解释，假设这是从可靠来源提取的。\n"
            "- 仅输出纯文本内容（无标题、无引号、无解释、无Markdown），直接开始正文。\n\n"
            "示例1：\n"
            "用户问题：Python中如何实现线程安全？\n"
            "输出：线程安全在Python中可以通过多种机制实现，包括使用锁（如threading.Lock）来保护共享资源。代码示例：import threading; lock = threading.Lock(); with lock: shared_var += 1; 此外，队列（queue.Queue）可用于线程间通信，避免直接访问共享数据。全局解释器锁（GIL）影响多线程性能，但不保证安全。最佳实践包括最小化共享状态和使用原子操作。\n\n"
            "示例2：\n"
            "用户问题：什么是量子计算的基本原理？\n"
            "输出：量子计算基于量子力学原理，利用量子比特（qubit）而非经典比特。qubit可处于叠加态，同时表示0和1，提高并行计算能力。纠缠允许qubit间即时关联，实现高效算法如Shor's分解大数算法。测量会坍缩状态，导致概率性输出。主要挑战包括退相干和错误校正。目前硬件包括超导、离子阱等技术。\n\n"
            f"用户问题：{text.strip()}\n"
            "现在直接输出纯文本："
        )
        raw = self.llm_client.invoke([Message(role="user", content=prompt)])
        if not raw:
            return []
        return [QueryCandidate(query_text=raw.strip(), source="hyde", score=0.8)]

    def _dedup(self, items: List[QueryCandidate]) -> List[QueryCandidate]:
        seen = set()
        unique: List[QueryCandidate] = []
        for it in items:
            key = it.query_text.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(it)
        return unique
