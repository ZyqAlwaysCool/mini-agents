'''
Description: 上下文候选信息压缩
Author: zyq
Date: 2026-01-09 09:31:36
LastEditors: zyq
LastEditTime: 2026-01-14 17:21:00
'''

from __future__ import annotations

from typing import Dict, List, Tuple

from mini_agents.core.config import ContextCompressConfig
from mini_agents.core.llm import BaseLLMClient
from ..models import ContextCandidateInfo
from ..token_counter import TokenCounter


class Compressor:
    def __init__(
        self,
        config: ContextCompressConfig,
        token_counter: TokenCounter | None = None,
        llm_client: BaseLLMClient | None = None,
    ):
        self.config = config
        self.token_counter = token_counter or TokenCounter()
        self.llm_client = llm_client

    def compress(self, sections: Dict[str, List[ContextCandidateInfo]]) -> Tuple[Dict[str, List[ContextCandidateInfo]], str]:
        merged = {k: list(v) for k, v in sections.items()}
        total_tokens = _calc_total_tokens(merged, self.token_counter)
        if total_tokens <= self.config.token_limit:
            return merged, _render_sections(merged)

        # 优先删减低优先级分区的低分内容
        priorities = list(self.config.partition_priority)
        for key in reversed(priorities):
            if total_tokens <= self.config.token_limit:
                break
            items = merged.get(key, [])
            if not items:
                continue
            # 按得分从低到高删除
            items = sorted(items, key=lambda c: c.relevance_score)
            while items and total_tokens > self.config.token_limit:
                removed = items.pop(0)
                total_tokens -= removed.token_count
            merged[key] = items

        # 兜底摘要：使用LLM做摘要总结压缩
        if total_tokens > self.config.token_limit and self.llm_client:
            for key in priorities:
                if key == "system":
                    continue
                new_items: List[ContextCandidateInfo] = []
                for item in merged.get(key, []):
                    if total_tokens <= self.config.token_limit:
                        new_items.append(item)
                        continue
                    if not self.llm_client:
                        break
                    summary_text = _summarize_with_llm(
                        self.llm_client, key, merged.get(key, []), self.config.summary_token, self.token_counter
                    )
                    new_items = [
                        ContextCandidateInfo(content=summary_text, token_count=self.token_counter.count(summary_text))
                    ]
                    merged[key] = new_items
                    total_tokens = _calc_total_tokens(merged, self.token_counter)
                    break
                if new_items:
                    merged[key] = new_items

        return merged, _render_sections(merged)


def _calc_total_tokens(sections: Dict[str, List[ContextCandidateInfo]], counter: TokenCounter) -> int:
    total = 0
    for items in sections.values():
        for item in items:
            if item.token_count <= 0:
                item.token_count = counter.count(item.content)
            total += item.token_count
    return total

def _render_sections(sections: Dict[str, List[ContextCandidateInfo]]) -> str:
    order = ["system", "constraints", "query_related", "recent_history", "long_term"]
    parts: List[str] = []
    for key in order:
        items = sections.get(key, [])
        if not items:
            continue
        title_map = {
            "system": "### 系统指令区",
            "constraints": "### 工具与环境约束区",
            "query_related": "### 查询关联候选区",
            "recent_history": "### 近期对话片段区",
            "long_term": "### 长期记忆区",
        }
        parts.append(f"{title_map.get(key, key)}:")
        for item in items:
            parts.append(f"- {item.content}")
    return "\n".join(parts)


def _summarize_with_llm(
    llm_client: BaseLLMClient,
    partition_key: str,
    items: List[ContextCandidateInfo],
    target_tokens: int,
    token_counter: TokenCounter,
) -> str:
    if not items:
        return ""
    prompt = (
        "你是上下文压缩助手，请对下列内容做摘要总结，保留关键约束/实体，不改变事实。"
        f"分区标识: {partition_key}，摘要请尽量控制在 {target_tokens} token 内，输出要点列表。"
    )
    content_lines = [f"- {c.content}" for c in items]
    user_text = "\n".join(content_lines)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_text},
    ]
    try:
        text = llm_client.invoke(messages)  # type: ignore[arg-type]
        return (text or "").strip()
    except Exception:
        return ""
