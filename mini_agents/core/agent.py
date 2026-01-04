'''
Description: 
Author: zyq
Date: 2025-12-25 16:23:04
LastEditors: zyq
LastEditTime: 2026-01-04 15:24:45
'''
from abc import ABC, abstractmethod
from typing import Optional, List
from loguru import logger

from .message import Message
from .llm import BaseLLMClient
from .config import AgentConfig, MemoryConfig
from ..tools.base import ToolExecutor
from ..memory import MemoryManager, MemoryRecord, MemoryQuery

class BaseAgent(ABC):
    def __init__(self, 
                 name: str, 
                 llm_client: BaseLLMClient, 
                 system_prompt: Optional[str] = None, 
                 agent_config: Optional[AgentConfig] = None,
                 tool_executor: Optional[ToolExecutor] = None,
                 memory_manager: Optional[MemoryManager] = None,
                 memory_config: Optional[MemoryConfig] = None,
                 ):

        self._name = name
        self._llm_client = llm_client
        self._system_prompt = system_prompt
        # 兼容传入配置实例、配置类或 None 三种场景
        self._agent_config = self._load_config(agent_config, AgentConfig)
        self._history: List[Message] = [] # 对话历史记录
        if tool_executor != None:
            self.tool_executor = tool_executor
            self._enabled_tool_calling = True
        else:
            self._enabled_tool_calling = False
        self._memory_config = self._load_config(memory_config, MemoryConfig)
        self._memory_manager = memory_manager
        self._memory_enabled = self._memory_config.enable_memory and self._memory_manager is not None

    def _load_config(self, cfg_obj, cfg_cls):
        """优先使用传入实例，其次调用传入类的 from_env，最后兜底到默认类 from_env"""
        if cfg_obj is None:
            return cfg_cls.from_env()
        if isinstance(cfg_obj, cfg_cls):
            return cfg_obj
        # 允许直接传入类
        if hasattr(cfg_obj, "from_env") and callable(getattr(cfg_obj, "from_env")):
            return cfg_obj.from_env()
        # 不符合预期时仍回退默认，避免 None 造成后续属性访问错误
        return cfg_cls.from_env()
    
    @abstractmethod
    def run(self, user_message: Message, **kwargs) -> Message:
        pass
    
    def add_history(self, message: Message) -> None:
        logger.info(f"Agent: {self._name} add history message: {message}")
        self._history.append(message)
    
    def clear_history(self) -> None:
        logger.info(f"Agent: {self._name} clearn history")
        self._history.clear()
    
    def get_history(self) -> List[Message]:
        return self._history
    
    def __str__(self) -> str:
        return f"Agent: {self._name} Provider: {self._llm_client.provider} Model: {self._llm_client.model}"
