"""
联网搜索工具，默认使用 Tavily 引擎
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from loguru import logger
from tavily import TavilyClient

from mini_agents.core.config import WebSearchConfig
from mini_agents.core.exceptions import ToolException
from .base import tool_register


class SearchEngine:
    """搜索引擎抽象"""

    name: str = "base"

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        raise NotImplementedError


class TavilySearchEngine(SearchEngine):
    """Tavily 引擎实现"""

    name: str = "tavily"

    def __init__(self, client: TavilyClient):
        self.client = client

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if not query:
            raise ToolException("查询词不能为空")
        resp = self.client.search(query=query, max_results=top_k)
        results = resp.get("results") or []
        items: List[Dict[str, Any]] = []
        for item in results:
            content = item.get("content") or item.get("snippet") or item.get("text") or ""
            items.append(
                {
                    "title": item.get("title") or "",
                    "url": item.get("url") or "",
                    "content": content,
                }
            )
        return items


class WebSearchService:
    """统一封装格式化输出"""

    def __init__(self, engine: SearchEngine, default_top_k: int = 5, max_content_len: int = 1000):
        self.engine = engine
        self.default_top_k = default_top_k
        self.max_content_len = max_content_len

    def search(self, query: str, top_k: Optional[int] = None) -> str:
        limit = top_k or self.default_top_k
        items = self.engine.search(query, limit)
        if not items:
            return "未检索到有效结果"
        lines: List[str] = []
        for idx, item in enumerate(items, start=1):
            title = item.get("title") or ""
            url = item.get("url") or ""
            content = (item.get("content") or "").strip()
            if len(content) > self.max_content_len:
                if self.max_content_len > 0:
                    content = content[:self.max_content_len] + "..."
            lines.append(f"[{idx}] {title}\n链接: {url}\n摘要: {content}")
        return "\n".join(lines)


def register_web_search_tool(
    config: Optional[WebSearchConfig] = None,
    engine: Optional[SearchEngine] = None,
) -> bool:
    """
    注册 web_search 工具
    无 API Key 时跳过注册并记录 warning
    """
    cfg = config or WebSearchConfig.from_env()
    if engine is None:
        if not cfg.enabled:
            logger.warning("未检测到 WEB_SEARCH_API_KEY，web_search 工具已禁用")
            return False
        client = TavilyClient(api_key=cfg.api_key, api_base_url=cfg.api_base_url)
        engine = TavilySearchEngine(client)

    service = WebSearchService(engine, cfg.top_k, cfg.max_content_len)

    @tool_register(name="web_search", description="基于{}引擎的联网搜索工具，可按照搜索关键词检索互联网，获取相关的信息".format(service.engine.name), override=True)
    def _web_search(query: str, top_k: Optional[int] = None) -> str:
        """
        联网搜索
        :param query: 搜索关键词
        :param top_k: 返回条数
        """
        return service.search(query, top_k)

    return True


__all__ = [
    "WebSearchService",
    "SearchEngine",
    "TavilySearchEngine",
    "register_web_search_tool",
]
