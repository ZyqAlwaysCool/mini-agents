'''
Description: 
Author: zyq
Date: 2025-12-26 16:41:27
LastEditors: zyq
LastEditTime: 2026-01-08 10:38:26
'''
from dotenv import load_dotenv
load_dotenv(override=True)

from mini_agents.agents.simple_agent import SimpleAgent
from mini_agents.core.llm import BaseLLMClient
from mini_agents.core.message import Message
from mini_agents.core.config import LLMConfig, AgentConfig
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


def test_simple_agent():
    llm_cfg = LLMConfig.from_env()
    llm_client = BaseLLMClient(model=llm_cfg.default_model, provider=llm_cfg.default_provider, api_key=llm_cfg.default_apikey, base_url=llm_cfg.default_base_url)
    agent = SimpleAgent("test_agent", llm_client)
    test_msg = Message(content="你是谁", role="user")
    answer = agent.run(test_msg, stream=False)
    print(answer)

def test_simple_agent_with_tools():
    llm_client = BaseLLMClient()
    agent = SimpleAgent("test_agent_2", llm_client, agent_config=AgentConfig, tool_executor=ToolExecutor)
    test_msg = Message(content="今天桂林的天气情况如何?", role="user")
    answer = agent.run(test_msg, stream=False)
    print(answer)
    
    print("==========")

    test_msg = Message(content="今天的日期是什么?", role="user")
    answer = agent.run(test_msg, stream=False)
    print(answer)
    


if __name__ == "__main__":
    #test_simple_agent()
    test_simple_agent_with_tools()

