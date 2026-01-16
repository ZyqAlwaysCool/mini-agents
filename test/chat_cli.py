'''
Description: 验证长时程agent
Author: zyq
Date: 2026-01-13 17:34:27
LastEditors: zyq
LastEditTime: 2026-01-14 18:11:57
'''

import argparse
import sys

from dotenv import load_dotenv

from mini_agents.agents.chat_agent import ChatAgent
from mini_agents.core.config import AgentConfig, MemoryConfig, ContextPipelineConfig
from mini_agents.core.llm import BaseLLMClient
from mini_agents.core.message import Message
from mini_agents.memory import create_memory_manager
from mini_agents.tools.base import ToolExecutor
from mini_agents.tools.web_search import register_web_search_tool

load_dotenv(override=True)


def build_agent(user_id: str, enable_search: bool) -> ChatAgent:
    llm = BaseLLMClient()
    mem_cfg = MemoryConfig.from_env()
    mem_mgr = create_memory_manager(mem_cfg)
    if enable_search:
        register_web_search_tool()
    agent = ChatAgent(
        name="chat",
        llm_client=llm,
        agent_config=AgentConfig.from_env(),
        tool_executor=ToolExecutor,
        memory_manager=mem_mgr,
        memory_config=mem_cfg,
        context_config=ContextPipelineConfig.from_env(),
        auto_register_search=False,  # 手动按开关注册
    )
    # 将 user_id 绑定在 agent 实例级别
    agent._default_user_id = user_id  # 简单存储，交互时用
    return agent


def main():
    parser = argparse.ArgumentParser(description="终端长时程对话")
    parser.add_argument("--user-id", required=True, help="用户ID，绑定记忆空间")
    parser.add_argument("--enable-search", action="store_true", help="启用联网搜索(web_search)")
    args = parser.parse_args()

    agent = build_agent(args.user_id, args.enable_search)

    print("输入内容开始对话，输入 quit 结束")
    while True:
        try:
            text = input("你：").strip()
        except EOFError:
            break
        if not text:
            continue
        msg = Message(content=text, role="user", metadata={"user_id": args.user_id})
        resp = agent.run(msg)
        print(f"助手：{resp.content}")
        if text.lower() == "quit":
            break


if __name__ == "__main__":
    main()
