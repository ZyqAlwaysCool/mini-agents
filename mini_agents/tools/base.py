'''
Description: 工具注册的基础模块。通过 @tool_register 装饰器将函数收集到全局注册表，Agent 可在运行时查找并调用这些工具。
Author: zyq
Date: 2025-12-26 11:08:17
LastEditors: zyq
LastEditTime: 2025-12-30 14:42:37
'''

from typing import Any, Callable, Dict, List, Optional, Set, get_type_hints
import inspect
import re

from mini_agents.core.exceptions import ToolException
from pydantic import BaseModel
from loguru import logger


class ToolParam(BaseModel):
    """工具参数的基本信息"""

    name: str
    type: str
    description: str


class RegisteredTool(BaseModel):
    """已注册工具的完整信息"""

    name: str
    description: str
    params: List[ToolParam]
    func: Callable

class ToolExecutor:
    """
    工具注册与调用的基础类。Agent 可以通过它获取工具清单或直接触发调用。
    """
    _registry: Dict[str, RegisteredTool] = {}
    _allowed_tools: Optional[Set[str]] = None
    
    @classmethod
    def register(cls, tool:RegisteredTool, override: bool = False) -> None:
        if not override and tool.name in cls._registry:
            raise ToolException(f"工具 {tool.name} 已注册，如需覆盖请传入 override=True")
        cls._registry[tool.name] = tool

    @classmethod
    def get(cls, name: str) -> RegisteredTool:
        cls._ensure_allowed(name)
        try:
            return cls._registry[name]
        except KeyError as exc:
            raise ToolException(f"未找到名称为 {name} 的工具") from exc

    @classmethod
    def list(cls) -> List[RegisteredTool]:
        if cls._allowed_tools is None:
            return list(cls._registry.values())
        return [tool for name, tool in cls._registry.items() if name in cls._allowed_tools]

    @classmethod
    def get_tool_desc(cls) -> List[Dict[str, Any]]:
        """以通用格式返回所有工具的元数据，便于喂给 LLM 作为工具列表"""
        specs: List[Dict[str, Any]] = []
        for tool in ToolExecutor.list():
            specs.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": [param.model_dump() for param in tool.params],
                }
            )
        return specs
    
    @classmethod
    def get_tool_info_by_name(cls, name: str) -> Dict[str, Any]:
        """获取指定工具元数据"""
        cls._ensure_allowed(name)
        return cls._registry[name].model_dump()

    @classmethod
    def run(cls, name: str, *args, **kwargs):
        """调用指定名称的工具"""
        cls._ensure_allowed(name)
        tool = cls._registry.get(name)
        if tool is None:
            raise ToolException(f"未找到名称为 {name} 的工具")
        return tool.func(*args, **kwargs)

    @classmethod
    def set_allowed_tools(cls, tool_names: Optional[List[str]] = None) -> None:
        """设置允许使用的工具名单，传入 None 表示全量开放"""
        cls._allowed_tools = set(tool_names) if tool_names else None

    @classmethod
    def _ensure_allowed(cls, name: str) -> None:
        """校验工具是否允许被使用"""
        if cls._allowed_tools is not None and name not in cls._allowed_tools:
            raise ToolException(f"工具 {name} 未被允许使用")


def _type_to_str(annotation: Any) -> str:
    """将类型注解转换为可读字符串"""
    if annotation is inspect._empty:
        return "Any"
    if isinstance(annotation, str):
        return annotation
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation)


def _parse_param_docs(docstring: Optional[str]) -> Dict[str, str]:
    """从 docstring 中提取 :param param: 描述"""
    if not docstring:
        return {}
    pattern = re.compile(r":param\s+(?P<name>\w+):\s*(?P<desc>.+)")
    return {m.group("name"): m.group("desc").strip() for m in pattern.finditer(docstring)}


def _build_tool_params(func: Callable) -> List[ToolParam]:
    """根据函数签名与文档构建参数列表"""
    signature = inspect.signature(func)
    type_hints = get_type_hints(func)
    doc_params = _parse_param_docs(inspect.getdoc(func))

    params: List[ToolParam] = []
    for name, param in signature.parameters.items():
        annotated_type = type_hints.get(name, param.annotation)
        param_type = _type_to_str(annotated_type)
        desc = doc_params.get(name)
        fallback_desc = "required" if param.default is inspect._empty else f"default={param.default!r}"
        params.append(ToolParam(name=name, type=param_type, description=desc or fallback_desc))
    return params


def tool_register(
    name: Optional[str] = None,
    description: Optional[str] = None,
    override: bool = False,
) -> Callable:
    """
    装饰器：将函数注册为 Agent 可调用的工具。

    Args:
        name: 对外暴露的工具名，默认使用函数名。
        description: 工具描述，默认取 docstring 的第一行。
        override: 是否允许覆盖同名已注册工具。
    """

    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__

        doc = inspect.getdoc(func) or ""
        tool_description = description or (doc.split("\n")[0] if doc else "No description provided.")

        registered = RegisteredTool(
            name=tool_name,
            description=tool_description,
            params=_build_tool_params(func),
            func=func,
        )

        ToolExecutor.register(registered, override=override)
        setattr(func, "__tool_registered__", True)
        setattr(func, "__tool_name__", tool_name)
        return func

    return decorator
