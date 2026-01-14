'''
Description: 多源信息收集器, 转换成统一的输出格式
Author: zyq
Date: 2026-01-09 09:30:15
LastEditors: zyq
LastEditTime: 2026-01-13 09:23:29
'''

from __future__ import annotations

from datetime import datetime
from typing import Callable, Dict, List, Tuple, Any

from loguru import logger

from mini_agents.core.config import ContextGatherConfig
from ..models import ContextCandidateInfo, ContextSourceType
from ..token_counter import TokenCounter


# 单个来源的数据采集器：从 state 中取需要的信息，返回统一的候选列表, 需要自行实现
ContextSourceFetcher = Callable[[Dict[str, Any]], List[ContextCandidateInfo]]


class Gatherer:
    def __init__(self, config: ContextGatherConfig, token_counter: TokenCounter | None = None):
        self.config = config
        self.token_counter = token_counter or TokenCounter()
        self._providers: List[Tuple[str, ContextSourceFetcher]] = []

    def register(self, name: str, fetcher: ContextSourceFetcher) -> None:
        self._providers.append((name, fetcher))

    def gather(self, state: Dict[str, Any]) -> List[ContextCandidateInfo]:
        collected: List[ContextCandidateInfo] = []
        for name, fetcher in self._providers:
            try:
                res = fetcher(state) or []
                for item in res:
                    if item.type == ContextSourceType.system_prompt:
                        item.priority = max(item.priority, 100)
                    if item.token_count <= 0:
                        item.token_count = self.token_counter.count(item.content)
                    collected.append(item)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"收集源失败: {name}, err={e}")

        filtered: List[ContextCandidateInfo] = []
        for item in collected:
            # 过滤低阈值得分的内容
            if item.type != ContextSourceType.system_prompt and item.relevance_score < self.config.min_score:
                continue
            filtered.append(item)

        history_items = [c for c in filtered if c.type == ContextSourceType.history]
        others = [c for c in filtered if c.type != ContextSourceType.history]
        history_sorted = sorted(history_items, key=lambda c: _ts_value(c.timestamp), reverse=True)
        
        if self.config.history_limit > 0:
            history_sorted = history_sorted[: self.config.history_limit]
        return others + history_sorted


def _ts_value(ts) -> float:
    if ts is None:
        return 0.0
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, datetime):
        return ts.timestamp()
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts).timestamp()
        except Exception:
            try:
                return float(ts)
            except Exception:
                return 0.0
    return 0.0
