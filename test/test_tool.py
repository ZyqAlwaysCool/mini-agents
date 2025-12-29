'''
Description: 
Author: zyq
Date: 2025-12-26 11:46:31
LastEditors: zyq
LastEditTime: 2025-12-26 15:15:48
'''
from mini_agents.tools.base import ToolExecutor, tool_register
from typing import Optional

@tool_register(name="test_tool", description="A simple test tool")
def test_tool(name: str, age: Optional[int] = None) -> str:
    return f"Hello from test_tool, {name}!"

@tool_register(name="test_tool2")
def test_tool2(name: str, age: Optional[int] = None) -> str:
    """测试工具2
    :param name: 名称
    :param age: 年龄
    """
    return f"Hello from test_tool2, {name}!"


if __name__ == "__main__":    
    print(ToolExecutor.get_tool_specs())
    print(ToolExecutor.run("test_tool", "wwwwworld!"))