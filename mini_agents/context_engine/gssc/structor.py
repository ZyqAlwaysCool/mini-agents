'''
Description: 上下文候选信息结构化
Author: zyq
Date: 2026-01-09 09:31:04
LastEditors: zyq
LastEditTime: 2026-01-12 10:10:18
'''

from __future__ import annotations

from typing import Dict, List, Tuple

from mini_agents.core.config import ContextStructConfig
from ..models import ContextCandidateInfo, ContextSourceType


Section = Tuple[str, str, List[ContextCandidateInfo]]


class Structor:
    def __init__(self, config: ContextStructConfig):
        self.config = config
        self._template = [
            ("system", "系统指令区", [ContextSourceType.system_prompt]),
            ("constraints", "工具与环境约束区", [ContextSourceType.tool_result, ContextSourceType.others]),
            ("query_related", "查询关联候选区", [ContextSourceType.rag, ContextSourceType.short_memory]),
            ("recent_history", "近期对话片段区", [ContextSourceType.history]),
            ("long_term", "长期记忆区", [ContextSourceType.long_memory]),
        ]

    def structure(self, candidates: List[ContextCandidateInfo]) -> Tuple[str, Dict[str, List[ContextCandidateInfo]]]:
        buckets: Dict[str, List[ContextCandidateInfo]] = {key: [] for key, _, _ in self._template}
        for c in candidates:
            for key, _, types in self._template:
                if c.type in types:
                    buckets[key].append(c)
                    break

        # 近期对话片段只保留配置条数，防止过长
        history_bucket = buckets.get("recent_history", [])
        if len(history_bucket) > self.config.history_turns:
            buckets["recent_history"] = history_bucket[: self.config.history_turns]

        sections: List[Section] = []
        for key, title, _ in self._template:
            sections.append((key, title, buckets.get(key, [])))

        context_parts: List[str] = []
        for key, title, items in sections:
            if not items:
                continue
            context_parts.append(f"{title}:")
            for item in items:
                context_parts.append(f"- {item.content}")
        context_text = "\n".join(context_parts)
        return context_text, {k: v for k, _, v in sections}
