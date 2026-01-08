'''
Description: 简单agent, 基于OpenAI原生API
Author: zyq
Date: 2025-12-26 15:18:58
LastEditors: zyq
LastEditTime: 2025-12-29 11:23:24
'''

from typing import Optional, get_type_hints, get_origin, get_args, List, Union, Iterator
import inspect
import re
from loguru import logger

from ..core.agent import BaseAgent
from ..core.llm import BaseLLMClient
from ..core.config import AgentConfig
from ..core.message import Message
from ..tools.base import ToolExecutor
from ..core.exceptions import BaseAgentsException
from ..common import build_enhanced_tool_section

class SimpleAgent(BaseAgent):
    def __init__(self,
                 name: str,
                 llm_client: BaseLLMClient,
                 agent_config: Optional[AgentConfig] = None,
                 system_prompt: Optional[str] = None,
                 tool_executor: Optional[ToolExecutor] = None,
                 ):
        self.messages: List[Message] = []
        super().__init__(name, llm_client, system_prompt, agent_config, tool_executor)
        
    def _get_enhanced_system_prompt(self) -> str:
        """构建增强的系统提示词，包含工具信息"""
        base_prompt = self._system_prompt or "你是一个有用的AI助手。"
        
        if not self._enabled_tool_calling:
            return base_prompt
        
        # 获取工具描述
        tool_desc_list = self.tool_executor.get_tool_desc()
        if len(tool_desc_list) == 0:
            return base_prompt
        
        tools_section = build_enhanced_tool_section(tool_desc_list)
        return base_prompt + tools_section

    def _parse_tool_calls(self, text: str) -> list:
        """解析文本中的工具调用，容忍无参/缺少右括号等轻微格式问题"""
        # 允许可选的 [] 包裹，允许参数为空，匹配到行尾或右括号
        pattern = r"\[?TOOL_CALL\s*:\s*(?P<tool>[^:\]\s]+)\s*:(?P<params>[^\]\n\r]*)\]?"
        tool_calls = []
        for match in re.finditer(pattern, text):
            tool_name = match.group("tool").strip()
            parameters = (match.group("params") or "").strip()
            tool_calls.append(
                {
                    "tool_name": tool_name,
                    "parameters": parameters,
                    "original": match.group(0),
                }
            )
        return tool_calls

    def _execute_tool_call(self, tool_name: str, parameters: str) -> str:
        """执行工具调用"""
        if self.tool_executor is None:
            raise BaseAgentsException("工具调用失败：未设置工具执行器")

        try:
            # 获取工具定义
            tool = self.tool_executor.get(tool_name)
            if not tool:
                raise BaseAgentsException(f"未找到工具：{tool_name}")

            # 解析参数
            param_dict = self._parse_tool_parameters(tool_name, parameters, tool)

            # 调用工具
            result = self.tool_executor.run(tool_name, **param_dict)
            return f"工具 {tool_name} 执行结果：\n{result}"

        except Exception as e:
            logger.error(f"工具调用异常: {e}")
            return f"工具调用失败：{str(e)}"

    def _parse_tool_parameters(self, tool_name: str, parameters: str, tool=None) -> dict:
        """
        将 LLM 生成的参数字符串解析为字典。
        支持格式：
          - key=value 逗号分隔  如 a=1,b=2
          - 单参数直接文本      如 search:Python编程(工具仅一个参数时)
        """
        tool_def = tool or self.tool_executor.get(tool_name)
        param_defs = {p.name: p for p in tool_def.params}
        signature = inspect.signature(tool_def.func)
        type_hints = get_type_hints(tool_def.func)

        raw = parameters.strip()
        if not raw:
            return {}

        def _base_type(annotation):
            """获取 Optional/Union 等展开后的基础类型"""
            origin = get_origin(annotation)
            args = get_args(annotation)
            if origin is None:
                return annotation
            if origin in (list, dict, tuple, set):
                return origin
            if type(None) in args:
                return next((arg for arg in args if arg is not type(None)), annotation)
            return origin or annotation

        def convert_value(key: str, raw_val: str):
            """根据函数注解做基础转换，注解缺失则尽量保持原样"""
            ann = type_hints.get(key)
            base = _base_type(ann) if ann else None
            val = raw_val.strip().strip('"').strip("'")
            lower = val.lower()
            if lower in ("none", "null", ""):
                return None
            try:
                if base is int:
                    return int(val)
                if base is float:
                    return float(val)
                if base is bool:
                    return lower in ("true", "1", "yes", "y", "t")
            except Exception:
                pass

            # 如果注解缺失，尝试简单数字解析
            if base is None:
                if re.fullmatch(r"-?\d+", val):
                    return int(val)
                if re.fullmatch(r"-?\d+\.\d+", val):
                    return float(val)
                if lower in ("true", "false"):
                    return lower == "true"

            return val

        parsed: dict = {}
        if "=" in raw:
            # 逗号分隔的 key=value
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            for part in parts:
                if "=" not in part:
                    raise BaseAgentsException(f"参数格式错误：{part}，请使用 key=value")
                key, val = part.split("=", 1)
                key = key.strip()
                if key not in param_defs:
                    raise BaseAgentsException(f"工具 {tool_name} 不存在参数：{key}")
                parsed[key] = convert_value(key, val)
        else:
            # 无 key=value，推断为单参数
            real_params = [p for p in signature.parameters.keys() if p != "self"]
            if len(real_params) != 1:
                raise BaseAgentsException("参数格式缺少键名，且工具参数不止一个，请使用 key=value")
            only_key = real_params[0]
            parsed[only_key] = convert_value(only_key, raw)

        # 基础缺失校验（根据函数签名判定必填项）
        missing = [
            name for name, param in signature.parameters.items()
            if name != "self" and param.default is inspect._empty and name not in parsed
        ]
        if missing:
            raise BaseAgentsException(f"缺少必要参数: {', '.join(missing)}")

        return parsed
    
    def run(self, user_message: Message, **kwargs) -> Message:
        """运行agent

        Args:
            user_message: 用户输入的消息
            **kwargs: 其他参数

        Returns:
            agent响应
        """
        # 添加system prompt
        system_prompt = self._get_enhanced_system_prompt()
        logger.debug(f"system prompt: {system_prompt}")
        self.messages.append(Message(role="system", content=system_prompt))

        # 添加历史对话记录
        if len(self._history) > 0:
            for history_message in self._history:
                self.messages.append(history_message)
        
        # 添加当前用户消息
        if not isinstance(user_message, Message):
            raise BaseAgentsException("用户输入的消息必须是Message类型")
        self.messages.append(user_message)

        if not self._enabled_tool_calling:
            response = self._llm_client.invoke(self.messages)
            assistant_message = Message(role="assistant", content=response)
            self.add_history(user_message)
            self.add_history(assistant_message)
            return assistant_message
        
        # 迭代处理多轮工具
        current_iteration = 0
        final_response = None
        
        while current_iteration < self._agent_config.max_round:
            # 调用llm
            response = self._llm_client.invoke(self.messages)
            
            # 检查是否有工具调用
            tool_calls = self._parse_tool_calls(response)
            logger.debug(f"parsed tool calls: {tool_calls}")
            
            if tool_calls:
                # 执行工具调用
                tool_results = []
                for call in tool_calls:
                    result = self._execute_tool_call(call["tool_name"], call["parameters"])
                    tool_results.append(result)
                

                self.messages.append(Message(role="assistant", content=response))
                
                # 添加工具结果
                tool_results_text = "\n\n".join(tool_results)
                self.messages.append({"role": "user", "content": f"工具执行结果：\n{tool_results_text}\n\n请基于这些结果给出完整的回答。"})

                current_iteration += 1
                continue
            
            # 无工具调用时, 返回最终响应
            final_response = response
            break

        # 超过最大迭代轮次无final resp, 再调用一次模型
        if current_iteration >= self._agent_config.max_round and final_response is None:
            final_response = self._llm_client.invoke(self.messages)
        
        # 保存历史记录
        final_assistant_message = Message(role="assistant", content=final_response)
        self.add_history(user_message)           
        self.add_history(final_assistant_message)
        
        return final_assistant_message


                
        
        
        
        


        

        
        
