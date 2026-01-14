'''
Description: 上下文候选信息筛选器
Author: zyq
Date: 2026-01-09 09:30:42
LastEditors: zyq
LastEditTime: 2026-01-12 09:59:20
'''

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mini_agents.core.config import ContextSelectConfig
from ..models import ContextCandidateInfo, ContextSourceType
from ..token_counter import TokenCounter
from mini_agents.core.exceptions import ContextException


class Selector:
    def __init__(self, config: ContextSelectConfig, token_counter: TokenCounter | None = None):
        self.config = config
        self.token_counter = token_counter or TokenCounter()

    def select(self, query: str, candidates: List[ContextCandidateInfo]) -> List[ContextCandidateInfo]:
        # 去重：基于候选 id
        unique: Dict[str, ContextCandidateInfo] = {}
        for c in candidates:
            if c.id not in unique:
                unique[c.id] = c
        deduped = list(unique.values())

        # 计算综合得分：相似度 + 新近性
        scored: List[Tuple[float, ContextCandidateInfo]] = []
        for c in deduped:
            sim = _text_cosine(query, c.content)
            rec = _recency_score(c.timestamp, self.config.recency_window_hours, self.config.recency_fade_hours)
            score = self.config.weight_similarity * sim + (1 - self.config.weight_similarity) * rec
            scored.append((score, c))

        # 按优先级分桶，桶内按综合得分排序
        buckets: Dict[int, List[Tuple[float, ContextCandidateInfo]]] = defaultdict(list)
        for score, c in scored:
            buckets[c.priority].append((score, c))

        ordered_candidates: List[ContextCandidateInfo] = []
        for pri in sorted(buckets.keys(), reverse=True):
            bucket = sorted(buckets[pri], key=lambda x: x[0], reverse=True)
            ordered_candidates.extend([c for _, c in bucket])

        # 选取：系统指令强制保留；非系统指令受 top_k 和 token_limit 约束
        selected: List[ContextCandidateInfo] = []
        non_sys_count = 0
        token_used = 0
        for c in ordered_candidates:
            if c.token_count <= 0:
                c.token_count = self.token_counter.count(c.content)
            if c.type == ContextSourceType.system_prompt:
                selected.append(c)
                token_used += c.token_count
                continue
            if non_sys_count >= self.config.top_k:
                continue
            if token_used + c.token_count > self.config.token_limit:
                continue
            selected.append(c)
            non_sys_count += 1
            token_used += c.token_count
        return selected


def _text_cosine(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if TfidfVectorizer and cosine_similarity:
        try:
            vec = TfidfVectorizer(lowercase=True)
            mat = vec.fit_transform([a, b])
            sim = cosine_similarity(mat[0], mat[1])[0][0]
            return float(sim)
        except Exception:
            raise ContextException("cosine相似度计算失败, 请先检查sklearn库是否安装")


def _recency_score(ts, window_hours: float, fade_hours: float) -> float:
    if ts is None:
        return 0.0
    try:
        if isinstance(ts, (int, float)):
            base_ts = float(ts)
        elif isinstance(ts, datetime):
            base_ts = ts.timestamp()
        else:
            base_ts = datetime.fromisoformat(str(ts)).timestamp()
        delta_hours = max((datetime.utcnow().timestamp() - base_ts) / 3600, 0.0)
    except Exception:
        return 0.0
    if delta_hours <= window_hours:
        return 1.0
    if delta_hours >= window_hours + fade_hours:
        return 0.0
    remain = window_hours + fade_hours - delta_hours
    return remain / fade_hours
