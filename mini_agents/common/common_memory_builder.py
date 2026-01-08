'''
Description: 封装记忆组件调用
Author: zyq
Date: 2026-01-07 17:04:09
LastEditors: zyq
LastEditTime: 2026-01-08 10:00:00
'''

from typing import List, Optional
import asyncio
from loguru import logger

from mini_agents.core.config import MemoryConfig
from mini_agents.core.message import Message
from mini_agents.memory import MemoryRecord, MemoryQuery, MemoryManager
from mini_agents.core.llm import BaseLLMClient


def inject_memory_context(memory_manager: MemoryManager, memory_config: MemoryConfig, user_message: Message) -> str:
    """根据用户消息检索记忆片段并返回上下文字符串"""
    if not memory_manager or not memory_config or not memory_config.enable_memory:
        return ""
    user_id = user_message.metadata.get("user_id") if user_message.metadata else None
    try:
        mq = MemoryQuery(text=user_message.content, user_id=user_id, top_k=memory_config.default_top_k)
        return memory_manager.inject_context(mq)
    except Exception as e:
        logger.warning(f"记忆检索失败: {e}")
        return ""


def add_session_message(memory_manager: MemoryManager, content: str, role: str, user_id: Optional[str], tags=None, score: float = 0.7, importance: float = 0.6) -> None:
    """写入单条短期记忆"""
    if not memory_manager:
        return
    try:
        rec = MemoryRecord(
            type="session",
            content=content,
            metadata={"role": role, "user_id": user_id},
            tags=tags or [],
            score=score,
            importance=importance,
        )
        memory_manager.add(rec)
    except Exception as e:
        logger.warning(f"写入记忆失败: {e}")


def save_history_messages(memory_manager: MemoryManager, history: List[Message], user_id: Optional[str]) -> None:
    """批量写入当前轮次的历史消息"""
    for msg in history:
        add_session_message(memory_manager, msg.content, msg.role, user_id)


def collect_records(user_msg: Message, history: List[Message], final_answer: str, user_id: Optional[str]) -> List[MemoryRecord]:
    """整理本轮对话记录, 用于精炼"""
    records: List[MemoryRecord] = []
    records.append(
        MemoryRecord(
            type="session",
            content=user_msg.content,
            metadata={"role": "user", "user_id": user_id},
            score=0.8,
            importance=0.8,
        )
    )
    for msg in history:
        records.append(
            MemoryRecord(
                type="session",
                content=msg.content,
                metadata={"role": msg.role, "user_id": user_id},
                score=0.7,
                importance=0.6,
            )
        )
    records.append(
        MemoryRecord(
            type="session",
            content=final_answer,
            metadata={"role": "assistant", "user_id": user_id},
            score=0.9,
            importance=0.9,
        )
    )
    return records


def run_memory_refiner_with_timeout(memory_manager: MemoryManager, 
                                    memory_config: MemoryConfig, 
                                    session_records: List[MemoryRecord], 
                                    user_id: Optional[str], 
                                    llm_client: BaseLLMClient) -> None:
    """启动记忆精炼为长期记忆, 并按配置超时等待"""
    if not memory_manager or not memory_config or not memory_config.refiner_enabled:
        logger.error("未配置记忆精炼能力")
        return
    th = memory_manager.refine_async(session_records, target_type="long_term", user_id=user_id, llm_client=llm_client)
    if not th:
        return
    timeout = getattr(memory_config, "refiner_timeout", 0) or 0
    if timeout <= 0:
        return
    th.join(timeout)
    if th.is_alive():
        logger.warning(f"refiner 在超时内未完成, user_id={user_id}, timeout={timeout}s")


async def run_memory_refiner_with_timeout_async(memory_manager: MemoryManager, 
                                                memory_config: MemoryConfig, 
                                                session_records: List[MemoryRecord], 
                                                user_id: Optional[str], 
                                                llm_client) -> None:
    """异步等待精炼线程, 避免阻塞事件循环"""
    if not memory_manager or not memory_config or not memory_config.refiner_enabled:
        return
    th = memory_manager.refine_async(session_records, target_type="long_term", user_id=user_id, llm_client=llm_client)
    if not th:
        return
    timeout = getattr(memory_config, "refiner_timeout", 0) or 0
    if timeout <= 0:
        return
    await asyncio.to_thread(th.join, timeout)
    if th.is_alive():
        logger.warning(f"refiner 在超时内未完成，user_id={user_id}, timeout={timeout}s")


def forget_if_needed(memory_manager: MemoryManager, memory_config: MemoryConfig) -> None:
    """按配置执行遗忘清理"""
    if not memory_manager or not memory_config:
        return
    if memory_config.forget_on_run_end:
        memory_manager.forget_all()
