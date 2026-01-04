'''
Description: 记忆组件数据模型和接口定义
Author: zyq
Date: 2025-12-31 10:49:51
LastEditors: zyq
LastEditTime: 2025-12-31 14:55:39
'''
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field

# session->临时记忆(单次对话) long_term->长期记忆 semantic->语义记忆 custom->用户自定义预留
MemoryType = Literal["session", "long_term", "semantic", "custom"]

class MemoryRecord(BaseModel):
    """记忆记录"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: MemoryType
    content: str
    metadata: Dict[str, Any] = {} # 元数据会包含user_id, 用于区别记忆所属的用户
    importance: float = 0.5
    score: float = 1.0
    access_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed_at: Optional[datetime] = None
    expire_at: Optional[datetime] = None
    embedding: Optional[List[float]] = None
    tags: List[str] = []


class MemoryQuery(BaseModel):
    """记忆检索请求"""

    text: str
    user_id: Optional[str] = None # 检索特定用户的记忆数据
    top_k: int = 5
    type_scope: Optional[List[MemoryType]] = None
    filters: Dict[str, Any] = {}
    include_raw_history: bool = False


class MemoryHit(BaseModel):
    """检索命中结果"""

    record: MemoryRecord
    score: float
    source_type: MemoryType
    reason: str = ""


class BaseMemoryStore(ABC):
    """记忆存储"""

    @abstractmethod
    def add(self, record: MemoryRecord) -> str:
        raise NotImplementedError

    def batch_add(self, records: List[MemoryRecord]) -> List[str]:
        return [self.add(rec) for rec in records]

    @abstractmethod
    def search(self, query: MemoryQuery) -> List[MemoryHit]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, record_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def clear(self, user_id: Optional[str] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def forget(self, now: datetime) -> List[str]:
        """遗忘记忆: 衰减或过期清理，返回被删除的记录 id 列表"""
        raise NotImplementedError

    def promote(self, records: List[MemoryRecord]) -> List[str]:
        """子类可覆盖，默认为批量写入"""
        return self.batch_add(records)

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        raise NotImplementedError
