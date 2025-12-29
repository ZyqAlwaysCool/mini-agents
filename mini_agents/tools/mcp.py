'''
Description: MCP 工具封装，将远程 MCP 服务的工具统一注册到本地 ToolExecutor（基于 mcp 官方客户端）
Author: zyq
Date: 2025-12-30 11:00:00
'''
from typing import Any, Dict, List, Optional
import asyncio
from loguru import logger
import httpx
import mcp.types as mcp_types
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from exceptiongroup import BaseExceptionGroup

from .base import ToolExecutor, RegisteredTool, ToolParam
from ..core.config import MCPConfig, MCPServerConfig
from ..core.exceptions import ToolException


def format_mcp_tool_name(server_name: str, tool_name: str) -> str:
    return f"mcp::{server_name}::{tool_name}"


def parse_mcp_tool_name(full_name: str) -> tuple[str, str]:
    parts = full_name.split("::")
    if len(parts) != 3 or parts[0] != "mcp":
        raise ToolException("MCP 工具名必须符合 mcp::<server>::<tool> 格式")
    return parts[1], parts[2]


def _build_params(raw_params: Any) -> List[ToolParam]:
    params: List[ToolParam] = []
    if not raw_params:
        return params

    if isinstance(raw_params, list):
        for item in raw_params:
            if not isinstance(item, dict):
                continue
            params.append(
                ToolParam(
                    name=str(item.get("name", "")),
                    type=str(item.get("type", "Any")),
                    description=str(item.get("description", "")),
                )
            )
    elif isinstance(raw_params, dict):
        target_schema = raw_params.get("parameters") if "parameters" in raw_params else raw_params
        if isinstance(target_schema, dict) and "properties" in target_schema:
            for key, schema in target_schema.get("properties", {}).items():
                params.append(
                    ToolParam(
                        name=str(key),
                        type=str(schema.get("type", "Any")),
                        description=str(schema.get("description", "")),
                    )
                )
    return params


def _run_sync(coro, err_msg: str):
    try:
        asyncio.get_running_loop()
        raise ToolException(err_msg)
    except RuntimeError:
        return asyncio.run(coro)


class MCPAsyncClient:
    """对 mcp 官方客户端的简化封装，负责列举工具与调用工具"""

    def __init__(self, server: MCPServerConfig):
        self._server = server

    async def list_tools(self) -> List[Dict[str, Any]]:
        headers = {}
        if self._server.api_key:
            headers[self._server.api_key_header] = self._server.api_key

        # 预检连通性，避免进入 SSE 时抛出异常组
        try:
            with httpx.Client(timeout=self._server.timeout) as client:
                with client.stream("GET", self._server.base_url, headers=headers) as resp:
                    resp.raise_for_status()
        except Exception as exc:
            logger.exception(f"MCP 服务 {self._server.name} 预检连接失败: {exc}")
            return []

        try:
            async with sse_client(self._server.base_url, headers=headers) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    init_result = await session.initialize()
                    logger.info(f"MCP服务器连接成功 server=({init_result.serverInfo.name}) version=({init_result.serverInfo.version})")

                    tools: List[Dict[str, Any]] = []
                    cursor: Optional[str] = None
                    while True:
                        result: mcp_types.ListToolsResult = await session.list_tools(cursor=cursor)
                        for tool in result.tools:
                            input_schema = getattr(tool, "inputSchema", None)
                            if input_schema is None:
                                schema_dict: Dict[str, Any] = {}
                            elif hasattr(input_schema, "model_dump"):
                                schema_dict = input_schema.model_dump()
                            elif isinstance(input_schema, dict):
                                schema_dict = input_schema
                            else:
                                schema_dict = {}

                            tools.append(
                                {
                                    "name": tool.name,
                                    "description": tool.description or "",
                                    "parameters": schema_dict,
                                }
                            )
                        cursor = result.nextCursor
                        if cursor is None:
                            break
                    return tools
        except BaseExceptionGroup as beg:
            logger.exception(f"MCP 服务 {self._server.name} list_tools 异常: {beg}")
            return []
        except Exception as exc:
            logger.exception(f"MCP 服务 {self._server.name} list_tools 异常: {exc}")
            return []

    async def call_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        headers = {}
        if self._server.api_key:
            headers[self._server.api_key_header] = self._server.api_key

        try:
            async with sse_client(self._server.base_url, headers=headers) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, args)
                    return result.model_dump()
        except BaseExceptionGroup as beg:
            logger.exception(f"MCP 服务 {self._server.name} 工具 {tool_name} 调用异常: {beg}")
            return {"error": str(beg)}
        except Exception as exc:
            logger.exception(f"MCP 服务 {self._server.name} 工具 {tool_name} 调用异常: {exc}")
            return {"error": str(exc)}


class MCPToolManager:
    """负责加载 MCP 工具并注册到 ToolExecutor"""

    def __init__(self, mcp_config: Optional[MCPConfig] = None):
        self._cfg = mcp_config or MCPConfig.from_env()

    def _build_registered_tool(self, server: MCPServerConfig, tool_meta: Dict[str, Any], client: MCPAsyncClient) -> RegisteredTool:
        raw_name = str(tool_meta.get("name") or tool_meta.get("tool_name") or "").strip()
        if not raw_name:
            raise ToolException(f"MCP 服务 {server.name} 返回了缺少名称的工具定义: {tool_meta}")

        merged_name = format_mcp_tool_name(server.name, raw_name)
        description = str(tool_meta.get("description", "")) or f"MCP 工具 {raw_name}"
        params = _build_params(tool_meta.get("parameters") or tool_meta.get("input_schema"))

        def _caller(**kwargs):
            return _run_sync(client.call_tool(raw_name, kwargs), "检测到正在运行的事件循环，请在异步上下文中直接 await MCPAsyncClient.call_tool")

        return RegisteredTool(
            name=merged_name,
            description=description,
            params=params,
            func=_caller,
        )

    def register(self, tool_executor: ToolExecutor, allowed_tools: Optional[List[str]] = None) -> List[str]:
        registered: List[str] = []

        validated_allow: Optional[List[str]] = None
        if allowed_tools is not None:
            validated_allow = []
            for name in allowed_tools:
                if name.startswith("mcp::"):
                    parse_mcp_tool_name(name)
                validated_allow.append(name)

        tools = _run_sync(self.register_async(tool_executor, allowed_tools), "检测到正在运行的事件循环，请在异步上下文中调用 register_async")

        if validated_allow is not None:
            existing = set(tool_executor._registry.keys())
            filtered = [name for name in validated_allow if name in existing]
            tool_executor.set_allowed_tools(filtered if filtered else None)

        return tools

    async def register_async(self, tool_executor: ToolExecutor, allowed_tools: Optional[List[str]] = None) -> List[str]:
        registered: List[str] = []

        validated_allow: Optional[List[str]] = None
        if allowed_tools is not None:
            validated_allow = []
            for name in allowed_tools:
                if name.startswith("mcp::"):
                    parse_mcp_tool_name(name)
                validated_allow.append(name)

        for server in self._cfg.servers:
            if not server.enabled:
                logger.info(f"MCP 服务 {server.name} 未启用，跳过")
                continue

            client = MCPAsyncClient(server)

            try:
                tools = await client.list_tools()
            except BaseExceptionGroup as beg:
                logger.exception(f"MCP 服务 {server.name} 拉取工具失败: {beg}")
                continue
            except BaseException as exc:
                logger.exception(f"MCP 服务 {server.name} 拉取工具失败: {exc}")
                continue

            for tool_meta in tools:
                try:
                    reg_tool = self._build_registered_tool(server, tool_meta, client)
                    tool_executor.register(reg_tool, override=True)
                    registered.append(reg_tool.name)
                except Exception as exc:
                    logger.exception(f"MCP 服务 {server.name} 工具注册失败: {exc}")

        if validated_allow is not None:
            existing = set(tool_executor._registry.keys())
            filtered = [name for name in validated_allow if name in existing]
            tool_executor.set_allowed_tools(filtered if filtered else None)

        return registered
