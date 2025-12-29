'''
Description: 
Author: zyq
Date: 2025-12-25 16:23:04
LastEditors: zyq
LastEditTime: 2025-12-29 11:21:51
'''
from abc import ABC, abstractmethod
from typing import Optional, List
from loguru import logger

from .message import Message
from .llm import BaseLLMClient
from .config import AgentConfig
from ..tools.base import ToolExecutor

class BaseAgent(ABC):
    def __init__(self, 
                 name: str, 
                 llm_client: BaseLLMClient, 
                 system_prompt: Optional[str] = None, 
                 agent_config: Optional[AgentConfig] = None,
                 tool_executor: Optional[ToolExecutor] = None,
                 ):

        self._name = name
        self._llm_client = llm_client
        self._system_prompt = system_prompt
        self._agent_config = agent_config.from_env() if agent_config != None else None
        self._history: List[Message] = [] # 对话历史记录
        if tool_executor != None:
            self.tool_executor = tool_executor
            self._enabled_tool_calling = True
        else:
            self._enabled_tool_calling = False
    
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