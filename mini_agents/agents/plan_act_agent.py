'''
Description: plan-act agent, 先规划后执行
Author: zyq
Date: 2025-12-30 15:54:58
LastEditors: zyq
LastEditTime: 2026-01-09 16:59:54
'''

import json
from typing import Optional, List, Dict, Any, Literal
from loguru import logger
from pocketflow import Flow, AsyncFlow, Node, AsyncNode
from pydantic import BaseModel

from ..core.agent import BaseAgent
from ..core.llm import BaseLLMClient
from ..core.config import AgentConfig
from ..core.message import Message
from ..tools.base import ToolExecutor
from ..core.exceptions import BaseAgentsException, ToolException
from ..common import build_tool_lines


DEFAULT_PLAN_PROMPT = """你是一个善于规划并执行的智能体，请先生成可执行的步骤计划，然后再逐步行动。

## 任务
- 用户问题：{question}
- 允许的最大步骤数：{max_steps}

## 可用工具
{tools_desc}

## 输出要求
- 必须输出 JSON，格式：{{"plan": ["步骤1", "步骤2", ...], "reason": "规划理由"}}。
- plan 必须是字符串列表，步骤数量不超过 {max_steps}，每一步应可执行、具体、避免空话。
- 不要输出其他文本。
"""

DEFAULT_ACT_PROMPT = """你正在按计划执行子任务。

## 当前子任务
{current_step}

## 执行历史
{history}

## 可用工具
{tools_desc}

## 输出要求
- 必须输出 JSON：{{"action": "CALL"|"SKIP"|"FINISH", "tool_name": "...", "args": {{}}, "reason": "...", "answer": "..." }}
- action=CALL 时必须提供 tool_name 和 args（字典）；action=SKIP 说明原因；action=FINISH 给出最终 answer。
- 只有当信息足够给出最终答案或所有计划步骤已完成时，才使用 FINISH；否则继续 CALL/ SKIP 到下一步。
- 严格遵守工具入参定义，字段名精确匹配。
"""

PlanActionStat = Literal["init", "act", "abort"]
ExecuteActionStat = Literal["init", "act", "abort"]


class PlanSharedState(BaseModel):
    query: str
    tools_desc: str
    plan: List[str]
    current_idx: int
    history: List[Message]
    final_answer: str
    max_steps: int


class PlanNode(Node):
    def __init__(self, llm_client: BaseLLMClient):
        super().__init__()
        self._llm_client = llm_client

    def _safe_load_plan(self, text: str) -> Optional[List[str]]:
        try:
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`").strip()
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].strip()
            data = json.loads(cleaned)
            plan = data.get("plan", [])
            if isinstance(plan, list):
                return [str(p) for p in plan if str(p).strip()]
        except Exception:
            return None
        return None

    def prep(self, shared: PlanSharedState) -> PlanActionStat:
        self._shared = shared
        if shared.plan:
            return "abort"
        prompt = DEFAULT_PLAN_PROMPT.format(
            question=shared.query,
            max_steps=shared.max_steps,
            tools_desc=shared.tools_desc or "无可用工具"
        )
        self._prompt = prompt
        return "act"

    def exec(self, prep_res: PlanActionStat) -> Dict[str, Any]:
        if prep_res == "abort":
            return {"exec_action": "abort"}
        messages = [Message(role="system", content=self._prompt)]
        raw = self._llm_client.invoke(messages)
        return {"exec_action": "act", "llm_answer": raw}

    def post(self, shared: PlanSharedState, prep_res: PlanActionStat, exec_res: Dict[str, Any]) -> str:
        if exec_res.get("exec_action") == "abort":
            return "done"
        plan = self._safe_load_plan(exec_res.get("llm_answer", ""))
        if not plan:
            raise BaseAgentsException("规划阶段解析失败，未获取到 plan")
        shared.plan = plan
        shared.current_idx = 0
        # 记录规划结果到历史，便于后续节点和调试
        plan_text = "\n".join(f"{idx+1}. {step}" for idx, step in enumerate(plan))
        shared.history.append(Message(role="assistant", content=f"规划结果：\n{plan_text}"))
        return "done"


class ActNode(Node):
    def __init__(self, llm_client: BaseLLMClient, tool_executor: ToolExecutor):
        super().__init__()
        self._llm_client = llm_client
        self._tool_executor = tool_executor

    def _safe_json(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            cleaned = text.strip()
            if "<think>" in cleaned:
                cleaned = cleaned.split("</think>")[-1].strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`").strip()
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].strip()
            return json.loads(cleaned)
        except Exception:
            return None

    def prep(self, shared: PlanSharedState) -> ExecuteActionStat:
        self._shared = shared
        if shared.current_idx >= len(shared.plan):
            return "abort"
        return "act"

    def exec(self, prep_res: ExecuteActionStat) -> Dict[str, Any]:
        if prep_res == "abort":
            return {"exec_action": "abort"}
        step_text = self._shared.plan[self._shared.current_idx]
        history_txt = "\n".join(f"{m.role}: {m.content}" for m in self._shared.history)
        prompt = DEFAULT_ACT_PROMPT.format(
            current_step=step_text,
            history=history_txt or "无",
            tools_desc=self._shared.tools_desc or "无可用工具"
        )
        #logger.info(f"act prompt: {prompt}")
        messages = [Message(role="system", content=prompt)]
        raw = self._llm_client.invoke(messages)
        return {"exec_action": "act", "llm_answer": raw, "step": step_text}

    def _run_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        try:
            res = self._tool_executor.run(tool_name, **args)
            return str(res)
        except ToolException as exc:
            logger.error(f"工具执行失败: {exc}")
            return f"工具执行失败: {exc}"
        except Exception as exc:
            logger.error(f"工具执行异常: {exc}")
            return f"工具执行异常: {exc}"

    def post(self, shared: PlanSharedState, prep_res: ExecuteActionStat, exec_res: Dict[str, Any]) -> str:
        if exec_res.get("exec_action") == "abort":
            shared.final_answer = shared.final_answer or "已完成全部步骤"
            return "finish"

        parsed = self._safe_json(exec_res.get("llm_answer", ""))
        if not parsed or "action" not in parsed:
            raise BaseAgentsException("执行阶段解析失败，未获取 action")

        action = parsed.get("action", "").upper()
        tool_name = parsed.get("tool_name")
        args = parsed.get("args", {}) if isinstance(parsed.get("args"), dict) else {}

        if action == "FINISH":
            if not parsed.get("answer") and shared.current_idx < len(shared.plan) - 1:
                shared.history.append(Message(role="assistant", content=f"尝试提前结束，但未给出答案，继续下一步。原因: {parsed.get('reason','')}"))
                shared.current_idx += 1
                return "next"
            shared.final_answer = parsed.get("answer") or parsed.get("reason") or shared.final_answer
            return "finish"
        if action == "SKIP":
            shared.history.append(Message(role="assistant", content=f"跳过步骤: {shared.plan[shared.current_idx]}，原因: {parsed.get('reason','')}"))
            shared.current_idx += 1
            if shared.current_idx >= min(len(shared.plan), shared.max_steps):
                shared.final_answer = shared.final_answer or parsed.get("answer") or "已完成计划步骤"
                return "finish"
            return "next"
        if action == "CALL":
            if not tool_name:
                raise BaseAgentsException("action=CALL 但缺少 tool_name")
            res = self._run_tool(tool_name, args)
            shared.history.append(Message(role="assistant", content=f"Action: {tool_name}, Args: {args}"))
            shared.history.append(Message(role="tool", content=f"Observation: {res}"))
            shared.current_idx += 1
            if parsed.get("answer"):
                shared.final_answer = parsed.get("answer")
                return "finish"
            if shared.current_idx >= min(len(shared.plan), shared.max_steps):
                shared.final_answer = shared.final_answer or res
                return "finish"
            return "next"
        shared.final_answer = parsed.get("answer") or shared.final_answer
        return "finish"


class PlanActAgent(BaseAgent):
    def __init__(self,
                 name: str,
                 llm_client: BaseLLMClient,
                 agent_config: AgentConfig,
                 system_prompt: Optional[str] = None,
                 tool_executor: Optional[ToolExecutor] = None):
        prompt = system_prompt or DEFAULT_PLAN_PROMPT
        super().__init__(name, llm_client, prompt, agent_config, tool_executor)
        self._shared: Optional[PlanSharedState] = None

    def _build_tools_desc(self) -> str:
        if not self._enabled_tool_calling:
            return ""
        tool_list = self.tool_executor.get_tool_desc()
        return build_tool_lines(tool_list, bullet="- ")

    def run(self, user_message: Message, **kwargs) -> Message:
        plan_node = PlanNode(self._llm_client)
        act_node = ActNode(self._llm_client, self.tool_executor)
        end_node = Node()

        plan_node - "done" >> act_node
        act_node - "next" >> act_node
        act_node - "finish" >> end_node

        shared = PlanSharedState(
            query=user_message.content,
            tools_desc=self._build_tools_desc(),
            plan=[],
            current_idx=0,
            history=[],
            final_answer="",
            max_steps=self._agent_config.max_round,
        )

        Flow(start=plan_node).run(shared)

        final_answer = shared.final_answer or "未获取到最终回答"
        assistant_msg = Message(role="assistant", content=final_answer)
        self.add_history(user_message)
        self.add_history(assistant_msg)
        self._shared = shared
        return assistant_msg

    async def run_async(self, user_message: Message, **kwargs) -> Message:
        plan_node = PlanNode(self._llm_client)
        act_node = ActNode(self._llm_client, self.tool_executor)
        end_node = AsyncNode()

        plan_node - "done" >> act_node
        act_node - "next" >> act_node
        act_node - "finish" >> end_node

        shared = PlanSharedState(
            query=user_message.content,
            tools_desc=self._build_tools_desc(),
            plan=[],
            current_idx=0,
            history=[],
            final_answer="",
            max_steps=self._agent_config.max_round,
        )

        await AsyncFlow(start=plan_node)._run_async(shared)

        final_answer = shared.final_answer or "未获取到最终回答"
        assistant_msg = Message(role="assistant", content=final_answer)
        self.add_history(user_message)
        self.add_history(assistant_msg)
        self._shared = shared
        return assistant_msg

    def get_run_history(self) -> Optional[List[Message]]:
        return self._shared.history if self._shared else []
