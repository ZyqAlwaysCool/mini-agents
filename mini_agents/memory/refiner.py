'''
Description: 记忆精炼: 使用LLM做摘要、去重、打标签并写入长期记忆
Author: zyq
Date: 2025-12-31 10:53:24
LastEditors: zyq
LastEditTime: 2026-01-04 11:08:28
'''
from __future__ import annotations

import threading
from typing import List, Optional, Set, TYPE_CHECKING
from loguru import logger
from .base import MemoryRecord, MemoryType


class MemoryRefiner:
    """记忆精炼器: LLM 摘要 + 去重 + 标签增强"""

    def __init__(self, summary_max_items: int = 30):
        self.summary_max_items = summary_max_items

    def run_async(
        self,
        records: List[MemoryRecord],
        memory_manager,
        target_type: MemoryType = "long_term",
        user_id: Optional[str] = None,
        llm_client=None,
    ):
        if not llm_client:
            logger.warning("refiner 缺少 llm_client，跳过精炼")
            return None
        thread = threading.Thread(
            target=self._run,
            args=(records, memory_manager, target_type, user_id, llm_client),
            daemon=False,
        )
        thread.start()
        return thread

    def _run(
        self,
        records: List[MemoryRecord],
        memory_manager,
        target_type: MemoryType,
        user_id: Optional[str],
        llm_client,
    ) -> None:
        try:
            selected = records[: self.summary_max_items]
            if not selected:
                return
            tags: Set[str] = set()
            lines = []
            for rec in selected:
                tags.update(rec.tags or [])
                role = rec.metadata.get("role") if rec.metadata else ""
                lines.append(f"{role}: {rec.content}")
            payload = "\n".join(lines)

            from mini_agents.core.message import Message  # 延迟导入，避免循环依赖

            prompt = (
                "请基于以下对话与工具观察，输出 JSON 数组，每个元素包含：\n"
                "- content: 精炼后的事实/画像（中文，简洁）\n"
                "- tags: 列表，中文短词\n"
                "- score: 0~1，表示与本轮对话的关联度/置信度\n"
                "- importance: 0~1，表示长期记忆的重要性\n"
                "要求：1) 合并相似信息，去重；2) 最多3条；3) 若没有任何值得长期存储的信息，请返回[]；4) 严格合法 JSON，不要额外说明。\n"
                "示例：\n"
                "[\n"
                "  {\"content\": \"用户偏好 Python asyncio 编写服务\", \"tags\": [\"python\", \"asyncio\"], \"score\": 0.92, \"importance\": 0.9},\n"
                "  {\"content\": \"常用数据库为PostgreSQL\", \"tags\": [\"postgres\", \"db\"], \"score\": 0.8, \"importance\": 0.7}\n"
                "]\n"
                f"原始内容：\n{payload}"
            )
            llm_res = llm_client.invoke([Message(role="user", content=prompt)])
            if not llm_res:
                return

            distilled_records, parse_err = self._parse_json_records(llm_res, target_type, user_id, tags)
            if not distilled_records:
                logger.info(f"refiner 未产出长期记忆，原因={parse_err}, user_id={user_id}, 原始输出={llm_res}")
                return
            # 去重：按 content 去重
            seen = set()
            unique_records: List[MemoryRecord] = []
            for rec in distilled_records:
                if rec.content in seen:
                    continue
                seen.add(rec.content)
                unique_records.append(rec)

            memory_manager.batch_add(unique_records)
            logger.info(f"refiner 完成写入，条数={len(unique_records)}")
        except Exception as e:
            logger.warning(f"refiner 处理异常: {e}")

    def _parse_json_records(
        self,
        llm_res: str,
        target_type: MemoryType,
        user_id: Optional[str],
        extra_tags: Set[str],
    ) -> (List[MemoryRecord], Optional[str]):
        """解析 LLM JSON 输出"""
        import json

        try:
            data = json.loads(llm_res)
            if not isinstance(data, list):
                return [], "llm_output_not_list"
            if len(data) == 0:
                return [], "llm_returned_empty"
        except Exception as e:
            return [], f"json_load_error:{e}"

        records: List[MemoryRecord] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            content = item.get("content") or ""
            if not content:
                continue
            tags_list = item.get("tags") or []
            score = item.get("score", 0.8)
            importance = item.get("importance", 0.8)
            records.append(
                MemoryRecord(
                    type=target_type,
                    content=str(content).strip(),
                    metadata={"user_id": user_id, "source": "refiner"},
                    importance=float(importance),
                    score=float(score),
                    tags=list(extra_tags | set(tags_list)),
                )
            )
        if not records:
            return [], "empty_records_after_parse"
        return records, None
