'''
Description: sqlite存储实现
Author: zyq
Date: 2025-12-31 17:32:04
LastEditors: zyq
LastEditTime: 2026-01-04 09:05:53
'''
from __future__ import annotations

import os
import json
import sqlite3
from typing import Any, Dict, List, Optional
from datetime import datetime
from .base import BaseBackend
from ...memory.base import MemoryRecord


class SQLiteBackend(BaseBackend):
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory (
                    id TEXT PRIMARY KEY,
                    type TEXT,
                    content TEXT,
                    metadata TEXT,
                    importance REAL,
                    score REAL,
                    access_count INTEGER,
                    created_at TEXT,
                    last_accessed_at TEXT,
                    expire_at TEXT,
                    embedding TEXT,
                    tags TEXT
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def insert(self, record: MemoryRecord) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory
                (id, type, content, metadata, importance, score, access_count, created_at, last_accessed_at, expire_at, embedding, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.type,
                    record.content,
                    json.dumps(record.metadata or {}),
                    record.importance,
                    record.score,
                    record.access_count,
                    record.created_at.isoformat(),
                    record.last_accessed_at.isoformat() if record.last_accessed_at else None,
                    record.expire_at.isoformat() if record.expire_at else None,
                    json.dumps(record.embedding) if record.embedding is not None else None,
                    json.dumps(record.tags or []),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def query_all(
        self,
        type_scope: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            clauses = []
            params: List[Any] = []
            if type_scope:
                placeholder = ",".join(["?"] * len(type_scope))
                clauses.append(f"type IN ({placeholder})")
                params.extend(type_scope)
            if user_id:
                clauses.append("json_extract(metadata, '$.user_id') = ?")
                params.append(user_id)
            if filters:
                tags = filters.get("tags")
                if tags:
                    for tag in tags:
                        clauses.append("EXISTS (SELECT 1 FROM json_each(memory.tags) je WHERE je.value = ?)")
                        params.append(tag)
                role = filters.get("role")
                if role:
                    clauses.append("json_extract(metadata, '$.role') = ?")
                    params.append(role)
                created_after = filters.get("created_after")
                if created_after:
                    clauses.append("created_at >= ?")
                    params.append(created_after.isoformat())

            where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            cur = conn.execute(f"SELECT * FROM memory {where_sql}", tuple(params))
            rows = cur.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def delete(self, record_id: str) -> None:
        conn = self._connect()
        try:
            conn.execute("DELETE FROM memory WHERE id=?", (record_id,))
            conn.commit()
        finally:
            conn.close()

    def clear(self, user_id: Optional[str] = None) -> None:
        conn = self._connect()
        try:
            if user_id:
                conn.execute(
                    "DELETE FROM memory WHERE json_extract(metadata, '$.user_id')=?",
                    (user_id,),
                )
            else:
                conn.execute("DELETE FROM memory")
            conn.commit()
        finally:
            conn.close()

    def update_access(self, record_id: str, access_count: int, last_accessed_at: Optional[datetime]) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE memory SET access_count=?, last_accessed_at=? WHERE id=?",
                (access_count, last_accessed_at.isoformat() if last_accessed_at else None, record_id),
            )
            conn.commit()
        finally:
            conn.close()

    def exists(self, user_id: Optional[str], mtype: str, content: str) -> bool:
        conn = self._connect()
        try:
            clauses = ["type = ?", "content = ?"]
            params: List[Any] = [mtype, content]
            if user_id:
                clauses.append("json_extract(metadata, '$.user_id') = ?")
                params.append(user_id)
            where_sql = " AND ".join(clauses)
            cur = conn.execute(f"SELECT 1 FROM memory WHERE {where_sql} LIMIT 1", tuple(params))
            row = cur.fetchone()
            return row is not None
        finally:
            conn.close()
