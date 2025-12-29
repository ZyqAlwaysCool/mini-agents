'''
Description:
Author: zyq
Date: 2025-12-29 15:11:41
LastEditors: zyq
LastEditTime: 2025-12-29 17:09:03
'''
from mini_agents.agents.react_agent import ReActAgent
from mini_agents.core.llm import BaseLLMClient
from mini_agents.core.config import AgentConfig
from mini_agents.tools.base import ToolExecutor, tool_register
from mini_agents.core.message import Message

from dotenv import load_dotenv
load_dotenv(override=True)

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

def test_react_agent():
    llm = BaseLLMClient()
    agent = ReActAgent(name="react", llm_client=llm, agent_config=AgentConfig, tool_executor=ToolExecutor)
    resp = agent.run(Message(content="今天是几号? 桂林的天气如何?", role="user"))
    print(resp)
    
if __name__ == "__main__":
    test_react_agent()