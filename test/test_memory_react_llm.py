'''
Description: 
- 需要环境变量提供 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL_ID（或 BaseLLMClient 默认配置）
- 先写入用户特征，再提问，让 Agent 回答时利用记忆
运行前请确保网络与鉴权可用，否则可跳过本测试
Author: zyq
Date: 2026-01-04 11:31:51
LastEditors: zyq
LastEditTime: 2026-01-04 16:39:43
'''
import os
import pytest
from pathlib import Path

from mini_agents.agents.react_agent import ReActAgent
from mini_agents.core.message import Message
from mini_agents.core.config import AgentConfig, MemoryConfig
from mini_agents.core.llm import BaseLLMClient
from mini_agents.memory import MemoryQuery, MemoryRecord, create_memory_manager
from mini_agents.tools.base import tool_register, ToolExecutor

@tool_register()
def get_current_date() -> str:
    """
    获取当前日期
    """
    return "今天是2025年12月29日"

@tool_register()
def get_weather(location: str) -> str:
    """
    获取指定地点的天气情况
    :param location: 地点
    :return 当地的天气情况
    """
    return f"{location}的天气晴朗, 温度为25度"

def _has_llm_env():
    return bool(os.environ.get("LLM_API_KEY") or os.environ.get("DEFAULT_API_KEY"))


def _has_embed_env():
    return bool(os.environ.get("MEMORY_REMOTE_EMBEDDING_MODEL") and os.environ.get("MEMORY_REMOTE_EMBEDDING_BASE_URL"))


@pytest.mark.skipif(not (_has_llm_env() and _has_embed_env()), reason="缺少 LLM 或 embedding 配置，跳过集成测试")
def test_react_agent_memory_with_real_llm(tmp_path: Path):
    # 配置记忆
    mem_cfg = MemoryConfig(
        enable_memory=True,
        hybrid_db_path=str(tmp_path / "hybrid_llm.db"),
        refiner_enabled=True,
        forget_on_run_end=False,
        remote_embedding_model=os.environ.get("MEMORY_REMOTE_EMBEDDING_MODEL", ""),
        remote_embedding_base_url=os.environ.get("MEMORY_REMOTE_EMBEDDING_BASE_URL", ""),
        remote_embedding_api_key=os.environ.get("MEMORY_REMOTE_EMBEDDING_API_KEY", ""),
        remote_embedding_dim=int(os.environ.get("MEMORY_REMOTE_EMBEDDING_DIM", 1024)),
    )
    manager = create_memory_manager(mem_cfg)
    llm = BaseLLMClient()

    agent_1 = ReActAgent(
        name="react-llm",
        llm_client=llm,
        agent_config=AgentConfig,
        tool_executor=None,  # 无工具，纯 LLM
        memory_manager=manager,
        memory_config=mem_cfg,
    )
    user_id = "user_test_v1"
    # 第一轮提问, 让agent记住特征
    test_user = Message(content="你好, 我叫李四, 来自北京, 喜欢python, 请记住我.", role="user", metadata={"user_id": user_id})
    _ = agent_1.run(test_user)

    # 第二轮提问，让 Agent 调用记忆
    agent_2 = ReActAgent(
        name="react-llm",
        llm_client=llm,
        agent_config=AgentConfig,
        tool_executor=None,  # 无工具，纯 LLM
        memory_manager=manager,
        memory_config=mem_cfg,
    )
    user2 = Message(content="我是谁？我喜欢什么？我的家乡在哪里?", role="user", metadata={"user_id": user_id})
    resp = agent_2.run(user2)

    # 确认回答是否包含用户特征
    print("第二轮回答:", resp.content)

def test_react_agent_memory_with_useless_message(tmp_path: Path):
    """增加无意义的消息, 测试精炼记忆过程是否做拦截, 不增加长期记忆."""
    user_id = "user_test_v1"
    
    # 配置记忆
    mem_cfg = MemoryConfig(
        enable_memory=True,
        hybrid_db_path=str(tmp_path / "hybrid_llm.db"),
        refiner_enabled=True,
        forget_on_run_end=False,
        remote_embedding_model=os.environ.get("MEMORY_REMOTE_EMBEDDING_MODEL", ""),
        remote_embedding_base_url=os.environ.get("MEMORY_REMOTE_EMBEDDING_BASE_URL", ""),
        remote_embedding_api_key=os.environ.get("MEMORY_REMOTE_EMBEDDING_API_KEY", ""),
        remote_embedding_dim=int(os.environ.get("MEMORY_REMOTE_EMBEDDING_DIM", 1024)),
    )
    manager = create_memory_manager(mem_cfg)
    llm = BaseLLMClient()

    agent_1 = ReActAgent(
        name="react-llm",
        llm_client=llm,
        agent_config=AgentConfig,
        tool_executor=ToolExecutor,
        memory_manager=manager,
        memory_config=mem_cfg,
    )
    
    resp_1 = agent_1.run(Message(content="你是谁?", role="user", metadata={"user_id": user_id}))
    print(resp_1.content)
    
    resp_2 = agent_1.run(Message(content="今天是几号?", role="user", metadata={"user_id": user_id}))
    print(resp_2.content)
    


if __name__ == "__main__":
    if not (_has_llm_env() and _has_embed_env()):
        print("缺少 LLM 或 embedding 配置，跳过")
    else:
        tmp_dir = Path("./tmp_db")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        test_react_agent_memory_with_real_llm(tmp_dir)
        #test_react_agent_memory_with_useless_message(tmp_dir)
