'''
Description:
Author: zyq
Date: 2025-12-30 16:10:37
LastEditors: zyq
LastEditTime: 2025-12-30 16:20:21
'''
from dotenv import load_dotenv

from mini_agents.agents.plan_act_agent import PlanActAgent
from mini_agents.core.llm import BaseLLMClient
from mini_agents.core.config import AgentConfig, MCPConfig
from mini_agents.tools.base import ToolExecutor, tool_register
from mini_agents.core.message import Message
from mini_agents.tools.mcp import MCPToolManager

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


def test_plan_act_agent():
    llm = BaseLLMClient()
    agent = PlanActAgent(name="planner", llm_client=llm, agent_config=AgentConfig, tool_executor=ToolExecutor)
    resp = agent.run(Message(content="今天是几号? 北京的天气如何?", role="user"))
    print(resp)

    print("="* 20)
    print(agent.get_run_history())


def test_plan_act_agent_with_mcp():
    MCPToolManager(mcp_config=MCPConfig().from_env()).register(ToolExecutor)
    ToolExecutor.set_allowed_tools(["mcp::mcp_server_1::getCurrentDateTime", 
                                    "mcp::mcp_server_1::getTables", 
                                    "mcp::mcp_server_1::getTableColumns",
                                    "mcp::mcp_server_1::getTableRowsData",])
    llm = BaseLLMClient()
    agent = PlanActAgent(name="planner", llm_client=llm, agent_config=AgentConfig, tool_executor=ToolExecutor)
    resp = agent.run(Message(content="去年珠海市人民政府发布了哪些政策", role="user"))
    print(resp)
    
    print("="* 20)
    print(agent.get_run_history())


if __name__ == "__main__":
    # test_plan_act_agent()
    test_plan_act_agent_with_mcp()
