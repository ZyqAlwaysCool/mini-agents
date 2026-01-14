'''
Description: Token计数器, 基于tiktoken
Author: zyq
Date: 2026-01-09 09:29:42
LastEditors: zyq
LastEditTime: 2026-01-09 11:01:59
'''

from __future__ import annotations

import math
import tiktoken
from abc import ABC, abstractmethod

class BaseTokenCounter(ABC):
    @abstractmethod
    def count(self, text: str) -> int:
        raise NotImplementedError

class TokenCounter(BaseTokenCounter):
    def __init__(self):
        self._encoder = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        return len(self._encoder.encode(text=text))
