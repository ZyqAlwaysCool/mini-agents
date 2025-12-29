'''
Description: ReAct Agent
Author: zyq
Date: 2025-12-29 10:35:26
LastEditors: zyq
LastEditTime: 2025-12-30 14:49:03
'''
import re
import asyncio
from typing import Optional, List,Tuple, Any, Dict, Literal
from pocketflow import AsyncFlow, AsyncNode, Flow, Node
from pydantic import BaseModel
from loguru import logger

from ..core.agent import BaseAgent
from ..core.llm import BaseLLMClient
from ..core.config import AgentConfig
from ..core.message import Message
from ..tools.base import ToolExecutor
from ..core.exceptions import BaseAgentsException

# 默认ReAct提示词模板
DEFAULT_REACT_PROMPT = """你是一个具备推理和行动能力的AI助手。你可以通过思考分析问题, 然后调用合适的工具来获取信息, 最终给出准确的答案。

## 可用工具列表
{tools}

## 工作流程
根据上下文决定下一步要调用的工具及参数, 或者返回FINISH结束。在调用工具时, 务必严格遵守工具的入参schema定义, 字段名称必须完全一致, 不要使用入参schema中未声明的字段名。
请严格按照以下输出JSON格式进行回应, 每次只能执行一个步骤: 
输出JSON定义: {{\"action\": \"CALL\"|\"FINISH\", \"tool_name\": \"...\", \"args\": {{...}}, \"reason\": \"...\", \"answer\": \"...\"}}
action: 下一步动作, 格式为:
- CALL: 调用工具获取相关信息.
- FINISH: 结束任务, 当你有足够信息得出结论时.
reason: 下一步动作的具体理由.
answer: 仅当action=FINISH时有效, 表示预期输出的最终回答.

## 重要提醒
1. 每次回应必须严格遵循输出JSON格式定义
2. 工具调用的入参必须严格遵循工具定义的入参schema
3. 只有当你确信有足够信息回答问题时,才使用FINISH
4. 如果工具返回的信息不够，继续使用其他工具或相同工具的不同参数
5. 你需要在有限的轮次内回答问题

## 当前任务
**Question:** {question}

## 剩余步数
{rounds}

## 执行历史
{history}

现在开始你的推理和行动："""

DecideNodeActionStat = Literal["init", "act", "abort"]
ExecuteNodeActionStat = Literal["init", "act", "abort"]

class DecideNodePrepModel(BaseModel):
    prep_action: DecideNodeActionStat = "init"
    prep_res: Dict[str, Any] = None

class DecideNodeExecModel(BaseModel):
    exec_action: DecideNodeActionStat = "init"
    exec_res: Dict[str, Any] = None

class ToolCallInfo(BaseModel):
    tool_name: str
    tool_args: Dict[str, Any]
    tool_exec_res: str = None

class RunStateRecord(BaseModel):
    current_step: int
    action: str
    reason: str
    node_type: str

class ReActSharedState(BaseModel):
    query: str # 用户问题
    history: List[Message]# 执行历史信息
    left_steps: int # 剩余可用步数
    tools_desc: str # 可用工具描述
    final_answer: str # 最终回答
    tool_call: List[ToolCallInfo] # 工具调用信息
    # run_record: List[RunStateRecord] # 运行记录

class DecideNode(Node):
    def __init__(self, llm_client: BaseLLMClient):
        super().__init__()
        self._llm_client = llm_client
    
    def _safe_json_loads(self, text: str) -> Optional[Dict[str, Any]]:
        import json
        try:
            if "<think>" in text:
                text = text.split("</think>")[1].strip()
                return json.loads(text)
            return json.loads(text)
        except Exception:
            return None
    
    def prep(self, shared: ReActSharedState) -> DecideNodePrepModel:
        self._shared = shared
        # 剩余步数耗尽则提前结束
        if self._shared.left_steps <= 0:
            if not self._shared.final_answer:
                self._shared.final_answer = "已达到最大轮次数，结束对话"
            return DecideNodePrepModel(prep_action="abort")

        history_msg_desc = ""
        if len(self._shared.history) > 0:
            for history_msg in self._shared.history:
                history_msg_desc += f"{history_msg.role}: {history_msg.content}\n"
        self._prompt = DEFAULT_REACT_PROMPT.format(
            tools=shared.tools_desc,
            question=shared.query,
            rounds=shared.left_steps,
            history=history_msg_desc
        )
        return DecideNodePrepModel(prep_action="act")
    
    def exec(self, prep_model: DecideNodePrepModel) -> DecideNodeExecModel:
        if not isinstance(prep_model, DecideNodePrepModel):
            raise BaseAgentsException("prep_res必须是DecideNodePrepModel type")
        if prep_model.prep_action == "abort":
            return DecideNodeExecModel(exec_action="abort")
        query = self._shared.query
        if query == None:
            raise BaseAgentsException("从共享状态shared中未找到用户的问题")

        all_messages: List[Message] = []
        all_messages.append(Message(role="system", content=self._prompt))
        raw_llm_answer = self._llm_client.invoke(all_messages)
        return DecideNodeExecModel(exec_action="act", exec_res={"llm_answer": raw_llm_answer})
    
    def post(self, shared: ReActSharedState, _: DecideNodePrepModel, exec_model: DecideNodeExecModel) -> str:
        if not isinstance(exec_model, DecideNodeExecModel):
            raise BaseAgentsException("exec_res必须是DecideNodeExecModel type")
        if exec_model.exec_action == "abort":
            return "abort"
        raw_llm_answer = exec_model.exec_res.get("llm_answer", None)
        parsed_llm_answer = self._safe_json_loads(raw_llm_answer)
        if not parsed_llm_answer or "action" not in parsed_llm_answer:
            raise BaseAgentsException("LLM回答解析失败, 无法感知下一步动作. llm_answer: " + raw_llm_answer)

        # 消耗一次决策步
        self._shared.left_steps = max(self._shared.left_steps - 1, 0)

        if parsed_llm_answer["action"].upper() == "CALL":
            # 如果步数耗尽则直接结束
            if self._shared.left_steps <= 0:
                self._shared.final_answer = parsed_llm_answer.get("answer") or parsed_llm_answer.get("reason") or "已达到最大轮次数，结束对话"
                return "finish"
            tool_call_info = ToolCallInfo(tool_name=parsed_llm_answer["tool_name"], tool_args=parsed_llm_answer["args"])
            self._shared.history.append(Message(role="assistant", content=f"Action: {tool_call_info.tool_name}, Args: {tool_call_info.tool_args}"))
            self._shared.tool_call.append(tool_call_info)
            shared = self._shared
            return "call"
        elif parsed_llm_answer["action"].upper() == "FINISH":
            self._shared.final_answer = parsed_llm_answer.get("answer") or parsed_llm_answer.get("reason", "")
            self._shared.history.append(Message(role="assistant", content=f"Finish: {self._shared.final_answer}"))
            shared = self._shared
            return "finish"
        else:
            self._shared.final_answer = parsed_llm_answer.get("answer", "")
            return "finish"

class ExecuteNodePrepModel(BaseModel):
    prep_action: ExecuteNodeActionStat = "init"
    prep_res: Dict[str, Any] = None

class ExecuteNodeExecModel(BaseModel):
    exec_action: ExecuteNodeActionStat = "init"
    exec_res: Dict[str, Any] = None

class ExecuteNode(Node):
    def __init__(self, tool_executor: ToolExecutor):
        super().__init__()
        self._tool_executor = tool_executor
    
    def prep(self, shared: ReActSharedState) -> ExecuteNodePrepModel:
        if not shared.tool_call:
            return ExecuteNodePrepModel(prep_action="abort")
        return ExecuteNodePrepModel(prep_action="act", prep_res={"tool_name": shared.tool_call[-1].tool_name, "tool_args": shared.tool_call[-1].tool_args})
    
    def exec(self, prep_model: ExecuteNodePrepModel) -> ExecuteNodeExecModel:
        if not isinstance(prep_model, ExecuteNodePrepModel):
            raise BaseAgentsException("prep_res必须是ExecuteNodePrepModel type")
        tool_name = prep_model.prep_res["tool_name"]
        tool_args = prep_model.prep_res["tool_args"]
        tool_call_res = self._tool_executor.run(tool_name, **tool_args)
        return ExecuteNodeExecModel(exec_action="act", exec_res={"tool_call_result": tool_call_res})
    
    def post(self, shared: ReActSharedState, _: ExecuteNodePrepModel, exec_model: ExecuteNodeExecModel) -> str:
        if not isinstance(exec_model, ExecuteNodeExecModel):
            raise BaseAgentsException("exec_res必须是ExecuteNodeExecModel type")
        if exec_model.exec_action == "abort":
            return "abort"
        
        # 获取tool执行结果
        tool_exec_res = exec_model.exec_res.get("tool_call_result", None)
        shared.tool_call[-1].tool_exec_res = tool_exec_res 
        shared.history.append(Message(role="tool", content=f"Observation: {tool_exec_res}"))
        return "decide"      

# 异步节点版本
class AsyncDecideNode(AsyncNode):
    def __init__(self, llm_client: BaseLLMClient):
        super().__init__()
        self._llm_client = llm_client

    def _safe_json_loads(self, text: str) -> Optional[Dict[str, Any]]:
        import json
        try:
            return json.loads(text)
        except Exception:
            return None

    async def prep_async(self, shared: ReActSharedState) -> DecideNodePrepModel:
        self._shared = shared
        if self._shared.left_steps <= 0:
            if not self._shared.final_answer:
                self._shared.final_answer = "已达到最大轮次数，结束对话"
            return DecideNodePrepModel(prep_action="abort")

        history_msg_desc = ""
        if len(self._shared.history) > 0:
            for history_msg in self._shared.history:
                history_msg_desc += f"{history_msg.role}: {history_msg.content}\n"
        self._prompt = DEFAULT_REACT_PROMPT.format(
            tools=shared.tools_desc,
            question=shared.query,
            rounds=shared.left_steps,
            history=history_msg_desc
        )
        return DecideNodePrepModel(prep_action="act")

    async def exec_async(self, prep_model: DecideNodePrepModel) -> DecideNodeExecModel:
        if not isinstance(prep_model, DecideNodePrepModel):
            raise BaseAgentsException("prep_res必须是DecideNodePrepModel type")
        if prep_model.prep_action == "abort":
            return DecideNodeExecModel(exec_action="abort")
        query = self._shared.query
        if query == None:
            raise BaseAgentsException("从共享状态shared中未找到用户的问题")

        all_messages: List[Message] = []
        all_messages.append(Message(role="system", content=self._prompt))
        raw_llm_answer = await asyncio.to_thread(self._llm_client.invoke, all_messages)
        return DecideNodeExecModel(exec_action="act", exec_res={"llm_answer": raw_llm_answer})

    async def post_async(self, shared: ReActSharedState, _: DecideNodePrepModel, exec_model: DecideNodeExecModel) -> str:
        if not isinstance(exec_model, DecideNodeExecModel):
            raise BaseAgentsException("exec_res必须是DecideNodeExecModel type")
        if exec_model.exec_action == "abort":
            return "abort"
        raw_llm_answer = exec_model.exec_res.get("llm_answer", None)
        parsed_llm_answer = self._safe_json_loads(raw_llm_answer)
        if not parsed_llm_answer or "action" not in parsed_llm_answer:
            raise BaseAgentsException("LLM回答解析失败, 无法感知下一步动作. llm_answer: " + raw_llm_answer)

        self._shared.left_steps = max(self._shared.left_steps - 1, 0)

        if parsed_llm_answer["action"].upper() == "CALL":
            if self._shared.left_steps <= 0:
                self._shared.final_answer = parsed_llm_answer.get("answer") or parsed_llm_answer.get("reason") or "已达到最大轮次数，结束对话"
                return "finish"
            tool_call_info = ToolCallInfo(tool_name=parsed_llm_answer["tool_name"], tool_args=parsed_llm_answer["args"])
            self._shared.history.append(Message(role="assistant", content=f"Action: {tool_call_info.tool_name}, Args: {tool_call_info.tool_args}"))
            self._shared.tool_call.append(tool_call_info)
            return "call"
        elif parsed_llm_answer["action"].upper() == "FINISH":
            self._shared.final_answer = parsed_llm_answer.get("answer") or parsed_llm_answer.get("reason", "")
            self._shared.history.append(Message(role="assistant", content=f"Finish: {self._shared.final_answer}"))
            return "finish"
        else:
            self._shared.final_answer = parsed_llm_answer.get("answer", "")
            return "finish"

class AsyncExecuteNode(AsyncNode):
    def __init__(self, tool_executor: ToolExecutor):
        super().__init__()
        self._tool_executor = tool_executor

    async def prep_async(self, shared: ReActSharedState) -> ExecuteNodePrepModel:
        if not shared.tool_call:
            return ExecuteNodePrepModel(prep_action="abort")
        return ExecuteNodePrepModel(prep_action="act", prep_res={"tool_name": shared.tool_call[-1].tool_name, "tool_args": shared.tool_call[-1].tool_args})

    async def exec_async(self, prep_model: ExecuteNodePrepModel) -> ExecuteNodeExecModel:
        if not isinstance(prep_model, ExecuteNodePrepModel):
            raise BaseAgentsException("prep_res必须是ExecuteNodePrepModel type")
        tool_name = prep_model.prep_res["tool_name"]
        tool_args = prep_model.prep_res["tool_args"]
        tool_call_res = await asyncio.to_thread(self._tool_executor.run, tool_name, **tool_args) # 交出事件循环, 避免同步阻塞
        return ExecuteNodeExecModel(exec_action="act", exec_res={"tool_call_result": tool_call_res})

    async def post_async(self, shared: ReActSharedState, _: ExecuteNodePrepModel, exec_model: ExecuteNodeExecModel) -> str:
        if not isinstance(exec_model, ExecuteNodeExecModel):
            raise BaseAgentsException("exec_res必须是ExecuteNodeExecModel type")
        if exec_model.exec_action == "abort":
            return "abort"

        tool_exec_res = exec_model.exec_res.get("tool_call_result", None)
        shared.tool_call[-1].tool_exec_res = tool_exec_res
        shared.history.append(Message(role="tool", content=f"Observation: {tool_exec_res}"))
        return "decide"
class EndNode(Node):
    def post(self, shared, prep_res, exec_res) -> str:
        # 用于做结束标志
        return "done"
        

class ReActAgent(BaseAgent):
    def __init__(self, 
                 name: str,
                 llm_client: BaseLLMClient,
                 system_prompt: Optional[str] = None,
                 agent_config: Optional[AgentConfig] = None,
                 tool_executor: Optional[ToolExecutor] = None,
                 ):
        if system_prompt == None:
            super().__init__(name, llm_client, DEFAULT_REACT_PROMPT, agent_config, tool_executor)
        else:
            super().__init__(name, llm_client, system_prompt, agent_config, tool_executor)

    
    def run(self, user_message: Message, **kwargs) -> Message:
        decide_node = DecideNode(self._llm_client)
        execute_node = ExecuteNode(self.tool_executor)
        end_node = EndNode()
        decide_node - "call" >> execute_node
        execute_node - "decide" >> decide_node
        decide_node - "finish" >> end_node
        
        total_step = self._agent_config.max_round
        tools_desc = ""
        if self._enabled_tool_calling:
            tools_desc = "\n".join(
                f"-{tool_info.get('name', '')}: {tool_info.get('description', '')} |paramters={tool_info.get('parameters', '')}"
                for tool_info in self.tool_executor.get_tool_desc()
            )
        
        shared = ReActSharedState(
            query=user_message.content,
            history=[],
            left_steps=total_step,
            tools_desc=tools_desc,
            final_answer="",
            tool_call=[]
        )
        Flow(start=decide_node).run(shared)

        final_answer = shared.final_answer or "未获取到最终回答"
        assistant_msg = Message(content=final_answer, role="assistant")
        # 记录对话历史
        self.add_history(user_message)
        self.add_history(assistant_msg)
        
        # 记录shared
        self._shared = shared
        return assistant_msg

    async def run_async(self, user_message: Message, **kwargs) -> Message:
        decide_node = AsyncDecideNode(self._llm_client)
        execute_node = AsyncExecuteNode(self.tool_executor)
        end_node = EndNode()
        decide_node - "call" >> execute_node
        execute_node - "decide" >> decide_node
        decide_node - "finish" >> end_node

        total_step = self._agent_config.max_round
        tools_desc = ""
        if self._enabled_tool_calling:
            tools_desc = "\n".join(
                f"-{tool_info.get('name', '')}: {tool_info.get('description', '')} |paramters={tool_info.get('parameters', '')}"
                for tool_info in self.tool_executor.get_tool_desc()
            )

        shared = ReActSharedState(
            query=user_message.content,
            history=[],
            left_steps=total_step,
            tools_desc=tools_desc,
            final_answer="",
            tool_call=[]
        )
        await AsyncFlow(start=decide_node)._run_async(shared)

        final_answer = shared.final_answer or "未获取到最终回答"
        assistant_msg = Message(content=final_answer, role="assistant")
        self.add_history(user_message)
        self.add_history(assistant_msg)
        
        # 记录shared
        self._shared = shared
        return assistant_msg
    
    def get_run_history(self) -> List[Message]:
        return self._shared.history
        
