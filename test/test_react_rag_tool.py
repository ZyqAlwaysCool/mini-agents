'''
Description: ReAct + RAG 工具端到端集成测试（真实 pipeline/LLM/向量库）
Author: zyq
Date: 2026-01-07 11:32:00
LastEditors: zyq
LastEditTime: 2026-01-07 11:33:16
'''
import os
import tempfile
from pathlib import Path

import pytest
from dotenv import load_dotenv

from mini_agents.agents.react_agent import ReActAgent
from mini_agents.core.config import AgentConfig, RAGConfig
from mini_agents.core.llm import BaseLLMClient
from mini_agents.core.message import Message
from mini_agents.rag.integration.adapter import build_rag_tool
from mini_agents.rag.pipeline import RAGPipeline
from mini_agents.rag.store.qdrant_store import QdrantVectorStore
from mini_agents.tools.base import ToolExecutor, ToolException

load_dotenv(override=True)


def _env_ready() -> bool:
    """检查真实依赖是否齐备"""
    required = [
        os.getenv("RAG_EMBEDDING_MODEL"),
        os.getenv("RAG_EMBEDDING_BASE_URL"),
        os.getenv("RAG_EMBEDDING_API_KEY"),
        os.getenv("RAG_QDRANT_URL"),
        os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
        os.getenv("LLM_BASE_URL") or os.getenv("DEFAULT_BASE_URL"),
        os.getenv("LLM_MODEL_ID") or os.getenv("DEFAULT_MODEL"),
    ]
    return all(required)


@pytest.mark.skipif(not _env_ready(), reason="缺少 RAG/LLM/Qdrant 配置，跳过真实集成测试")
def test_react_agent_with_real_rag_pipeline():
    # 1) 构建真实 RAG pipeline（使用环境变量配置）
    rag_cfg = RAGConfig.from_env()
    store = QdrantVectorStore(url=rag_cfg.qdrant_url, api_key=rag_cfg.qdrant_api_key)
    llm_client = BaseLLMClient()
    pipeline = RAGPipeline(rag_config=rag_cfg, store=store, llm_client=llm_client)

    # 2) 写入一批假数据到知识库
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "kb_demo.txt"
        p.write_text("北极星项目是公司的旗舰产品，核心亮点是实时数据同步与高可用架构。", encoding="utf-8")
        ingest_stats = pipeline.ingest(str(p), biz_name="kb_demo")
        assert ingest_stats["written"] > 0

    # 3) 注册专用 RAG 工具，默认固定 biz_id=kb_demo，避免 LLM 误填
    ToolExecutor.set_allowed_tools(None)
    try:
        build_rag_tool(pipeline, name="rag_query_kb_demo", description="kb_demo 业务知识库检索")
    except ToolException:
        build_rag_tool(pipeline, name="rag_query_kb_demo", description="kb_demo 业务知识库检索")

    # 4) 构造真实 LLM + ReActAgent，仅开放 RAG 工具，问题显式要求调用
    llm = BaseLLMClient()
    ToolExecutor.set_allowed_tools(["rag_query_kb_demo"])
    agent = ReActAgent(name="react", llm_client=llm, agent_config=AgentConfig, tool_executor=ToolExecutor)
    user_q = "北极星项目的核心亮点是什么?"
    resp = agent.run(Message(content=user_q, role="user"))
    
    print(resp)

    # 5) 断言：应完成回答且下一轮 prompt 带有 RAG 片段
    history = agent.get_run_history()
    assert any("rag_query_kb_demo" in m.content for m in history if m.role == "assistant"), "未调用 RAG 工具"
    assert any("RAG上下文已更新" in m.content for m in history if m.role == "tool"), "未注入 RAG 片段"
    assert resp.content and len(resp.content.strip()) >= 10, "最终回答过短或为空"


if __name__ == "__main__":
    test_react_agent_with_real_rag_pipeline()
