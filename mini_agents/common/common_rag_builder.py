'''
Description: 封装RAG组件调用
Author: zyq
Date: 2026-01-07 17:04:21
LastEditors: zyq
LastEditTime: 2026-01-08 10:04:10
'''

from typing import Any, Tuple
from mini_agents.core.message import Message


def extract_rag_context(tool_exec_res: Any) -> str:
    """从工具返回值中提取 rag_context 字段"""
    if isinstance(tool_exec_res, dict) and tool_exec_res.get("rag_context"):
        return tool_exec_res.get("rag_context", "")
    return ""


def build_tool_observation(tool_exec_res: Any) -> Tuple[str, Message]:
    """生成标准的工具观测消息(对应于react agent中的observation过程), 并返回 rag 上下文"""
    rag_ctx = extract_rag_context(tool_exec_res)
    if rag_ctx:
        return rag_ctx, Message(role="tool", content=f"Observation: RAG上下文已更新\n{rag_ctx}")
    return "", Message(role="tool", content=f"Observation: {tool_exec_res}")
