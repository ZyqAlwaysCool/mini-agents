'''
Description: 上下文工程数据模型
Author: zyq
Date: 2026-01-09 09:28:59
LastEditors: zyq
LastEditTime: 2026-01-09 10:23:04
'''

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class ContextSourceType(str, Enum):
    """上下文信息源类型"""
    system_prompt = "system_prompt" # 系统提示词
    short_memory = "short_memory" # 短期记忆
    long_memory = "long_memory" # 长期记忆
    rag = "rag" # rag检索结果
    tool_result = "tool_result" # 工具调用结果
    history = "history" # 历史会话记录
    others = "others" # 其他


class ContextCandidateInfo(BaseModel):
    content: str # 信息内容
    timestamp: Optional[int | str | datetime] = None
    token_count: int = 0 # 由tiktoken计算得出
    relevance_score: float = 0.0 # 相关性评分0-1
    metadata: Dict[str, Any] = Field(default_factory=dict)
    type: ContextSourceType = ContextSourceType.others
    priority: int = 0
    id: str = ""

    @model_validator(mode="after") # 实例创建后填充逻辑
    def fill_id_and_score(self):
        if not self.id:
            self.id = hashlib.md5(self.content.encode("utf-8")).hexdigest()
        if self.relevance_score < 0:
            self.relevance_score = 0.0
        if self.relevance_score > 1:
            self.relevance_score = 1.0
        return self


class PipelineResult(BaseModel):
    triggered: bool
    context_text: str
    gathered: List[ContextCandidateInfo] = Field(default_factory=list)
    selected: List[ContextCandidateInfo] = Field(default_factory=list)
    sections: Dict[str, List[ContextCandidateInfo]] = Field(default_factory=dict)
