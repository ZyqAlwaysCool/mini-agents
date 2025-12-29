'''
Description:
Author: zyq
Date: 2025-12-29 15:11:41
LastEditors: zyq
LastEditTime: 2025-12-30 16:14:32
'''
from mini_agents.agents.react_agent import ReActAgent
from mini_agents.core.llm import BaseLLMClient
from mini_agents.core.config import AgentConfig, MCPConfig, LLMConfig
from mini_agents.tools.base import ToolExecutor, tool_register
from mini_agents.core.message import Message
from mini_agents.tools.mcp import MCPToolManager


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

    print(agent.get_run_history())


def test_react_agent_async():
    """异步调用示例"""
    import asyncio

    async def _run():
        llm = BaseLLMClient()
        agent = ReActAgent(name="react", llm_client=llm, agent_config=AgentConfig, tool_executor=ToolExecutor)
        resp = await agent.run_async(Message(content="今天是几号? 桂林的天气如何?", role="user"))
        print(resp)

    asyncio.run(_run())

def test_react_agent_with_mcp():
    llm_cfg = LLMConfig.from_env()
    MCPToolManager(mcp_config=MCPConfig().from_env()).register(ToolExecutor)
    ToolExecutor.set_allowed_tools(["mcp::mcp_server_1::getCurrentDateTime", 
                                    "mcp::mcp_server_1::getTables", 
                                    "mcp::mcp_server_1::getTableColumns",
                                    "mcp::mcp_server_1::getTableRowsData",])
    # llm = BaseLLMClient(model=llm_cfg.default_model, provider=llm_cfg.default_provider, api_key=llm_cfg.default_apikey, base_url=llm_cfg.default_base_url)
    llm = BaseLLMClient()
    agent = ReActAgent(name="react", llm_client=llm, agent_config=AgentConfig, tool_executor=ToolExecutor)
    resp = agent.run(Message(content="去年珠海市人民政府发布了哪些政策", role="user"))
    print(resp)

    print('='* 20)
    for history in agent.get_run_history():
        print(history)
    
    
if __name__ == "__main__":
    # test_react_agent()
    
    # 运行异步示例
    # test_react_agent_async()

    test_react_agent_with_mcp()
    
    
