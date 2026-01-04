'''
Description: 供长期/语义记忆使用, 持久化存储引擎抽象
Author: zyq
Date: 2025-12-31 17:31:43
LastEditors: zyq
LastEditTime: 2025-12-31 17:41:33
'''

from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime
from ...memory.base import MemoryRecord


class BaseBackend:
    """持久化后端接口，封装 CRUD 与查询"""

    def insert(self, record: MemoryRecord) -> None:
        raise NotImplementedError

    def bulk_insert(self, records: List[MemoryRecord]) -> None:
        for rec in records:
            self.insert(rec)

    def query_all(
        self,
        type_scope: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def exists(self, user_id: Optional[str], mtype: str, content: str) -> bool:
        """检查是否存在同一用户、同类型、相同内容的记录"""
        raise NotImplementedError

    def delete(self, record_id: str) -> None:
        raise NotImplementedError

    def clear(self, user_id: Optional[str] = None) -> None:
        raise NotImplementedError

    def update_access(self, record_id: str, access_count: int, last_accessed_at: Optional[datetime]) -> None:
        raise NotImplementedError
