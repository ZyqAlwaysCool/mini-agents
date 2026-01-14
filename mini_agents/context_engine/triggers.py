'''
Description: 上下文压缩触发器
Author: zyq
Date: 2026-01-09 09:29:32
LastEditors: zyq
LastEditTime: 2026-01-09 09:45:37
'''

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .token_counter import TokenCounter
from mini_agents.core.exceptions import ContextException


class BaseTrigger(ABC):
    @abstractmethod
    def should_trigger(self, state: Dict[str, Any]) -> bool:
        raise NotImplementedError


class RoundLimitTrigger(BaseTrigger):
    """基于轮次限制触发"""
    def __init__(self, round_limit: int):
        self.round_limit = round_limit

    def should_trigger(self, state: Dict[str, Any]) -> bool:
        if "round" not in state and "round_index" not in state:
            raise ContextException("state中缺少round和round_index字段, 无法执行trigger限制判断")
        current = state.get("round", state.get("round_index", 0))
        try:
            return int(current) >= self.round_limit
        except Exception:
            return False


class TokenLimitTrigger(BaseTrigger):
    """基于 token 上限触发"""
    def __init__(self, token_limit: int, token_counter: Optional[TokenCounter] = None):
        self.token_limit = token_limit
        self.token_counter = token_counter or TokenCounter()

    def should_trigger(self, state: Dict[str, Any]) -> bool:
        if "history_text" not in state:
            raise ContextException("state中缺少history_text字段, 无法执行trigger限制判断")
        text = state.get("history_text") or ""
        if not isinstance(text, str):
            return False
        tokens = self.token_counter.count(text)
        return tokens >= self.token_limit
