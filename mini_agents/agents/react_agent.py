'''
Description: ReAct Agent
Author: zyq
Date: 2025-12-29 10:35:26
LastEditors: zyq
LastEditTime: 2026-01-04 14:51:35
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
from ..memory import MemoryQuery, MemoryRecord

# 默认ReAct提示词模板
DEFAULT_REACT_PROMPT = """你是一个具备推理和行动能力的AI助手。你可以通过思考分析问题, 然后调用合适的工具来获取信息, 最终给出准确的答案。

## 可用工具列表
{tools}

## 相关记忆
{memory}

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
    memory_context: str # 记忆上下文片段
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
            memory=shared.memory_context,
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
            memory=shared.memory_context,
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
                 memory_manager=None,
                 memory_config=None,
                 ):
        if system_prompt == None:
            super().__init__(name, llm_client, DEFAULT_REACT_PROMPT, agent_config, tool_executor, memory_manager, memory_config)
        else:
            super().__init__(name, llm_client, system_prompt, agent_config, tool_executor, memory_manager, memory_config)

    
    def run(self, user_message: Message, **kwargs) -> Message:
        decide_node = DecideNode(self._llm_client)
        end_node = EndNode()
        if self._enabled_tool_calling:
            execute_node = ExecuteNode(self.tool_executor)
            decide_node - "call" >> execute_node
            execute_node - "decide" >> decide_node
            decide_node - "finish" >> end_node
        else:
            decide_node - "finish" >> end_node
        
        total_step = self._agent_config.max_round
        # 记忆上下文构造
        memory_context = ""
        user_id = user_message.metadata.get("user_id") if user_message.metadata else None
        if self._memory_enabled:
            try:
                mq = MemoryQuery(text=user_message.content, user_id=user_id, top_k=self._memory_config.default_top_k)
                memory_context = self._memory_manager.inject_context(mq) # 检索记忆片段, 构造记忆片段的上下文信息
            except Exception as e:
                logger.warning(f"记忆检索失败: {e}")
                memory_context = ""
            try:
                user_rec = MemoryRecord(type="session", content=user_message.content, metadata={"role": "user", "user_id": user_id}, tags=user_message.metadata.get("tags", []) if user_message.metadata else [])
                self._memory_manager.add(user_rec)
            except Exception as e:
                logger.warning(f"写入记忆失败: {e}")
        
        # tool调用
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
            memory_context=memory_context,
            final_answer="",
            tool_call=[]
        )
        Flow(start=decide_node).run(shared)

        final_answer = shared.final_answer or "未获取到最终回答"
        assistant_msg = Message(content=final_answer, role="assistant")
        # 记录对话历史
        self.add_history(user_message)
        self.add_history(assistant_msg)
        # 写入记忆与清理
        if self._memory_enabled:
            try:
                assistant_rec = MemoryRecord(type="session", content=final_answer, metadata={"role": "assistant", "user_id": user_id})
                self._memory_manager.add(assistant_rec)
                # 同步写入当前轮对话过程
                self._save_history_to_memory(shared, user_id)
                if self._memory_config.forget_on_run_end:
                    self._memory_manager.forget_all()
                if self._memory_config.refiner_enabled:
                    session_records = self._collect_session_records(shared, user_message, final_answer, user_id)
                    th = self._memory_manager.refine_async(session_records, target_type="long_term", user_id=user_id, llm_client=self._llm_client)
                    self._wait_refiner_thread(th, user_id)
            except Exception as e:
                logger.warning(f"记忆写入或清理失败: {e}")
        
        # 记录shared
        self._shared = shared
        return assistant_msg

    async def run_async(self, user_message: Message, **kwargs) -> Message:
        decide_node = AsyncDecideNode(self._llm_client)
        end_node = EndNode()
        if self._enabled_tool_calling:
            execute_node = AsyncExecuteNode(self.tool_executor)
            decide_node - "call" >> execute_node
            execute_node - "decide" >> decide_node
            decide_node - "finish" >> end_node
        else:
            decide_node - "finish" >> end_node

        total_step = self._agent_config.max_round
        memory_context = ""
        user_id = user_message.metadata.get("user_id") if user_message.metadata else None
        if self._memory_enabled:
            try:
                mq = MemoryQuery(text=user_message.content, user_id=user_id, top_k=self._memory_config.default_top_k)
                memory_context = self._memory_manager.inject_context(mq)
            except Exception as e:
                logger.warning(f"记忆检索失败: {e}")
                memory_context = ""
            try:
                user_rec = MemoryRecord(type="session", content=user_message.content, metadata={"role": "user", "user_id": user_id}, tags=user_message.metadata.get("tags", []) if user_message.metadata else [])
                self._memory_manager.add(user_rec)
            except Exception as e:
                logger.warning(f"写入记忆失败: {e}")
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
            memory_context=memory_context,
            final_answer="",
            tool_call=[]
        )
        await AsyncFlow(start=decide_node)._run_async(shared)

        final_answer = shared.final_answer or "未获取到最终回答"
        assistant_msg = Message(content=final_answer, role="assistant")
        self.add_history(user_message)
        self.add_history(assistant_msg)
        if self._memory_enabled:
            try:
                assistant_rec = MemoryRecord(type="session", content=final_answer, metadata={"role": "assistant", "user_id": user_id})
                self._memory_manager.add(assistant_rec)
                self._save_history_to_memory(shared, user_id)
                if self._memory_config.forget_on_run_end:
                    self._memory_manager.forget_all()
                if self._memory_config.refiner_enabled:
                    session_records = self._collect_session_records(shared, user_message, final_answer, user_id)
                    logger.info(f"启动精炼: records={len(session_records)}, user_id={user_id}")
                    th = self._memory_manager.refine_async(session_records, target_type="long_term", user_id=user_id, llm_client=self._llm_client)
                    await self._wait_refiner_thread_async(th, user_id)
            except Exception as e:
                logger.warning(f"记忆写入或清理失败: {e}")
        
        # 记录shared
        self._shared = shared
        return assistant_msg
    
    def get_run_history(self) -> List[Message]:
        return self._shared.history

    def _collect_session_records(self, shared: ReActSharedState, user_msg: Message, final_answer: str, user_id: Optional[str]) -> List[MemoryRecord]:
        """汇总本轮对话记录（含用户输入、工具观察、最终答案）供精炼"""
        session_records: List[MemoryRecord] = []
        session_records.append(
            MemoryRecord(
                type="session",
                content=user_msg.content,
                metadata={"role": "user", "user_id": user_id},
                score=0.8,
                importance=0.8,
            )
        )
        for msg in shared.history:
            session_records.append(
                MemoryRecord(
                    type="session",
                    content=msg.content,
                    metadata={"role": msg.role, "user_id": user_id},
                    score=0.7,
                    importance=0.6,
                )
            )
        session_records.append(
            MemoryRecord(
                type="session",
                content=final_answer,
                metadata={"role": "assistant", "user_id": user_id},
                score=0.9,
                importance=0.9,
            )
        )
        return session_records

    def _save_history_to_memory(self, shared: ReActSharedState, user_id: Optional[str]) -> None:
        """将当前轮的历史（工具观察等）同步写入短期记忆"""
        for msg in shared.history:
            try:
                rec = MemoryRecord(
                    type="session",
                    content=msg.content,
                    metadata={"role": msg.role, "user_id": user_id},
                    score=0.7,
                    importance=0.6,
                )
                self._memory_manager.add(rec)
            except Exception as e:
                logger.warning(f"写入历史记忆失败: {e}")

    def _wait_refiner_thread(self, thread, user_id: Optional[str]):
        """等待精炼线程，超时则记录告警"""
        if not thread:
            return
        timeout = getattr(self._memory_config, "refiner_timeout", 0) or 0
        if timeout <= 0:
            return
        thread.join(timeout)
        if thread.is_alive():
            logger.warning(f"refiner 在超时内未完成，user_id={user_id}, timeout={timeout}s")

    async def _wait_refiner_thread_async(self, thread, user_id: Optional[str]):
        """异步等待精炼线程"""
        if not thread:
            return
        timeout = getattr(self._memory_config, "refiner_timeout", 0) or 0
        if timeout <= 0:
            return
        await asyncio.to_thread(thread.join, timeout)
        if thread.is_alive():
            logger.warning(f"refiner 在超时内未完成，user_id={user_id}, timeout={timeout}s")
