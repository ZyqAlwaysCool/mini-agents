'''
Description: 长时程对话agent, 集成联网搜索、记忆与上下文压缩
Author: zyq
Date: 2026-01-13 16:38:11
LastEditors: zyq
LastEditTime: 2026-01-14 18:24:11
'''

from __future__ import annotations

import json
from typing import List, Optional, Dict, Any
from loguru import logger

from mini_agents.core.agent import BaseAgent
from mini_agents.core.llm import BaseLLMClient
from mini_agents.core.config import AgentConfig, MemoryConfig, ContextPipelineConfig
from mini_agents.core.message import Message
from mini_agents.core.exceptions import BaseAgentsException, ToolException
from mini_agents.tools.base import ToolExecutor
from mini_agents.tools.web_search import register_web_search_tool
from mini_agents.context_engine import (
    ContextPipeline,
    ContextPipelineConfig,
    RoundLimitTrigger,
    TokenLimitTrigger,
    Gatherer,
    Selector,
    Structor,
    Compressor,
    ContextCandidateInfo,
    ContextSourceType,
)
from mini_agents.common import (
    build_tool_lines,
    query_memory_context,
    add_session_message,
    save_history_messages,
    collect_records,
    run_memory_refiner_with_timeout,
    forget_if_needed,
)


DEFAULT_CHAT_PROMPT = """你是一个具备长期记忆的长对话智能助手。
你的核心任务是：基于现有记忆与上下文，在必要时合理使用工具，最终给用户提供准确、完整、有帮助的中文回答。

当前对话状态：
## 历史上下文记录
{context_text}

## 相关记忆片段（按重要性排序）
long_term: 长期记忆知识
{memory_context}

## 可用工具列表
{tools}

## 必须严格遵守的输出格式（任何格式错误都会被系统拒绝）
请严格按照以下JSON格式回复，**不要输出任何多余文字**：

```json
{{
  "action": "CALL" | "FINISH",
  "tool_name": "工具名称（仅当action=CALL时必填）",
  "args": {{参数键值对（仅当action=CALL时必填）}},
  "reason": "你此刻做此决定的主要理由，简洁清晰",
  "answer": "当action=FINISH时的完整最终回答（必须是流畅自然的中文）"
}}
```

## 决策规则（重要，请严格遵守）：

信息足够给出高质量完整回答 → action="FINISH" + 写出answer
需要更多信息/不清楚用户意图 → action="FINISH" + 在answer里礼貌地询问
需要调用工具获取信息 → action="CALL"（一次只建议调用一个最必要的工具）
连续对同一工具调用超过2次仍无实质进展 → 强制转为FINISH并说明情况
本轮对话已累计执行动作达到 {max_iter} 次 → 强制FINISH，给出当前最佳可能回答或说明受限原因

当前用户问题：{query}
"""

FALLBACK_PROMPT = """
你是一个聪明、知识渊博、态度极好、耐心又真诚的AI助手。

你的核心目标是：
1. 尽可能给用户最有帮助、最清晰、最实用的回答
2. 让用户感觉「被认真对待」且「很舒服」
3. 在准确的前提下，尽量用自然、亲切、像真人对话的语气

请遵循以下沟通原则（重要程度由高到低排列）：

• 真实第一：不会就诚实说不知道，不要硬编
• 先理解，再回答：先搞清楚用户真正想知道什么
• 结构清晰：复杂答案尽量分段、编号、使用小标题
• 避免过度谦虚/自贬，也不要过于自大
• 当用户明显情绪化/发泄时，先共情再解决问题
• 能用表格、列表、步骤分解时尽量用，方便阅读
• 专业内容尽量说人话，避免连续大量术语轰炸
• 超长回答时，先给重点结论，再展开细节
• 被问到争议/敏感话题时：尽量客观、中立、呈现多方主要观点，不轻易站队

### 用户问题
{query}

### 对话历史
{history}
"""



class ChatAgent(BaseAgent):
    """长时程对话 Agent"""
    _refine_interval: int = 5  # 精炼触发间隔（轮）

    def __init__(
        self,
        name: str,
        llm_client: BaseLLMClient,
        system_prompt: Optional[str] = None,
        agent_config: Optional[AgentConfig] = None,
        tool_executor: Optional[ToolExecutor] = ToolExecutor,
        memory_manager=None,
        memory_config: Optional[MemoryConfig] = None,
        context_config: Optional[ContextPipelineConfig] = None,
        auto_register_search: bool = True,
    ):
        self._context_config = context_config or ContextPipelineConfig.from_env()
        if self._context_config.round_limit is None:
            self._context_config.round_limit = 6
        if self._context_config.token_limit is None:
            self._context_config.token_limit = 10000
        self._context_pipeline = self._build_context_pipeline(llm_client) # 配置上下文组件
        self._run_count = 0

        mem_cfg = memory_config or MemoryConfig.from_env()

        super().__init__(
            name,
            llm_client,
            system_prompt or DEFAULT_CHAT_PROMPT,
            agent_config,
            tool_executor,
            memory_manager,
            mem_cfg,
        )

        if self._memory_manager is None:
            raise BaseAgentsException("ChatAgent 需要配置 memory_manager")

        self._tool_executor = tool_executor or ToolExecutor
        self._search_enabled = False
        if auto_register_search:
            self._search_enabled = register_web_search_tool()

    def _get_user_id(self, user_message: Message) -> str:
        user_id = (user_message.metadata or {}).get("user_id")
        if not user_id:
            raise BaseAgentsException("用户ID缺失，无法绑定记忆空间")
        return user_id

    def _build_context_pipeline(self, llm_client: BaseLLMClient) -> ContextPipeline:
        triggers = []
        if self._context_config.round_limit is not None:
            triggers.append(RoundLimitTrigger(self._context_config.round_limit))
        if self._context_config.token_limit is not None:
            triggers.append(TokenLimitTrigger(self._context_config.token_limit))

        gatherer = Gatherer(self._context_config.gather)

        def _fetch_memory(state: Dict[str, Any]) -> List[ContextCandidateInfo]:
            text = state.get("session_context") or ""
            if not text:
                return []
            return [
                ContextCandidateInfo(
                    content=text,
                    type=ContextSourceType.short_memory,
                    priority=70,
                    relevance_score=0.7,
                )
            ]

        def _fetch_history(state: Dict[str, Any]) -> List[ContextCandidateInfo]:
            msgs: List[Message] = state.get("history_messages") or []
            res: List[ContextCandidateInfo] = []
            for msg in msgs:
                res.append(
                    ContextCandidateInfo(
                        content=f"{msg.role}: {msg.content}",
                        type=ContextSourceType.history,
                        timestamp=msg.timestamp,
                        relevance_score=0.6,
                    )
                )
            return res

        gatherer.register("memory", _fetch_memory)
        gatherer.register("history", _fetch_history)

        selector = Selector(self._context_config.select)
        structor = Structor(self._context_config.struct)
        compressor = Compressor(self._context_config.compress, llm_client=llm_client)

        return ContextPipeline(triggers, gatherer, selector, structor, compressor)

    def _run_context_pipeline(
        self,
        query: str,
        session_context: str,
        history_messages: List[Message],
    ) -> str:
        state = {
            "session_context": session_context,
            "history_messages": history_messages,
            "history_text": "\n".join(msg.content for msg in history_messages),
            "round": len(history_messages),
        }
        result = self._context_pipeline.run(query, state)
        if result.triggered:
            return result.context_text
        # 未触发时回退到最近历史
        recent = history_messages[-5:] if history_messages else []
        return "\n".join(f"{m.role}: {m.content}" for m in recent)

    def _safe_json(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`").strip()
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].strip()
            if "<think>" in cleaned:
                cleaned = cleaned.split("</think>")[-1].strip()
            return json.loads(cleaned)
        except Exception:
            return None

    def run(self, user_message: Message, **kwargs) -> Message:
        self._run_count += 1
        user_id = self._get_user_id(user_message) # 要求输入用户id, 以划分不同的用户空间存储各自的长期记忆.
        if user_message.content.strip().lower() == "quit":
            return self._finish_session(user_message, quitting=True)

        long_memory_context = ""
        session_context = ""
        if self._memory_enabled:
            # 构建记忆上下文
            long_memory_context = query_memory_context(
                self._memory_manager, self._memory_config, user_message, type_scope=["long_term"]
            )
            session_context = query_memory_context(
                self._memory_manager, self._memory_config, user_message, type_scope=["session"]
            )
            add_session_message(self._memory_manager, user_message.content, "user", user_id)

        tools_desc = ""
        if self._enabled_tool_calling:
            # 构建可选tools描述
            tools_desc = build_tool_lines(self.tool_executor.get_tool_desc(), bullet="-")

        local_history: List[Message] = []
        conversation_history: List[Message] = list(self._history)
        conversation_history.append(user_message)

        base_context_text = self._run_context_pipeline(
            user_message.content, session_context, conversation_history
        )

        final_answer = ""
        no_progress = 0
        max_no_progress = 3
        iteration_guard = 0
        max_iteration_guard = 5  # 兜底防护，避免极端情况下死循环
        repeat_call_guard = {}

        while True:
            iteration_guard += 1
            if iteration_guard >= max_iteration_guard:
                final_answer = self._fallback_response(user_message, context_text)
                break
            extra_context = ""
            if local_history:
                extra_context = "\n[本轮中间结果]\n" + "\n".join(f"{m.role}: {m.content}" for m in local_history)
            context_text = (base_context_text or "无") + extra_context
            prompt = (self._system_prompt or DEFAULT_CHAT_PROMPT).format(
                context_text=context_text,
                memory_context=long_memory_context or "无",
                tools=tools_desc or "无",
                query=user_message.content,
                max_iter=max_iteration_guard,
            )
            #logger.debug(f"system prompt: {prompt}")
            llm_messages = [Message(role="system", content=prompt)]
            raw = self._llm_client.invoke(llm_messages)
            parsed = self._safe_json(raw)
            if not parsed or "action" not in parsed:
                # 出错保护
                no_progress += 1
                local_history.append(
                    Message(
                        role="assistant",
                        content="系统提醒：输出未按 JSON 格式，请返回 {\"action\":\"CALL|FINISH\",\"tool_name\":\"...\",\"args\":{},\"reason\":\"...\",\"answer\":\"...\"}",
                    )
                )
                if no_progress >= max_no_progress:
                    err_desc = "多次解析失败，当前信息不足，请提供更多细节"
                    final_answer = self._fallback_response(user_message, context_text, err_desc)
                    break
                continue

            action = str(parsed.get("action", "")).upper()
            if action == "CALL":
                no_progress = 0
                tool_name = parsed.get("tool_name")
                if not tool_name:
                    no_progress += 1
                    local_history.append(Message(role="assistant", content="系统提醒：缺少 tool_name，请补充工具名称"))
                    if no_progress >= max_no_progress:
                        err_desc = "缺少工具名称，无法继续，请补充更多信息"
                        final_answer = self._fallback_response(user_message, context_text, err_desc)
                        break
                    continue
                args = parsed.get("args", {})
                if args is None:
                    args = {}
                if not isinstance(args, dict):
                    no_progress += 1
                    local_history.append(Message(role="assistant", content="系统提醒：工具参数必须是字典，请按 schema 返回"))
                    if no_progress >= max_no_progress:
                        err_desc = "工具参数格式异常，需要用户补充更多信息或换一种描述"
                        final_answer = self._fallback_response(user_message, context_text, err_desc)
                        break
                    continue
                try:
                    res = self.tool_executor.run(tool_name, **args)
                except ToolException as exc:
                    res = f"工具调用失败: {exc}"
                obs_msg = Message(role="tool", content=f"{tool_name}结果: {res}")
                local_history.append(obs_msg)
                conversation_history.append(obs_msg)

                repeat_key = f"{tool_name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
                repeat_call_guard[repeat_key] = repeat_call_guard.get(repeat_key, 0) + 1
                if repeat_call_guard[repeat_key] >= 3:
                    err_desc = "已多次调用同一工具无新增信息，请补充更明确的需求"
                    final_answer = self._fallback_response(user_message, context_text, err_desc)
                    break
                continue

            if action == "FINISH":
                final_answer = parsed.get("answer") or parsed.get("reason") or "已完成回答"
                break

            no_progress += 1
            local_history.append(Message(role="assistant", content="系统提醒：action 无效，请使用 CALL 或 FINISH"))
            if no_progress >= max_no_progress:
                final_answer = parsed.get("reason") or "当前信息不足，请提供更多细节"
                break

        if not final_answer:
            final_answer = "当前信息不足，请提供更多细节"

        assistant_msg = Message(content=final_answer, role="assistant")
        self.add_history(user_message)
        self.add_history(assistant_msg)
        for msg in local_history:
            self.add_history(msg)

        if self._memory_enabled:
            try:
                add_session_message(self._memory_manager, final_answer, "assistant", user_id, score=0.9, importance=0.9)
                save_history_messages(self._memory_manager, local_history, user_id)
                forget_if_needed(self._memory_manager, self._memory_config)
                session_records = collect_records(user_message, local_history, final_answer, user_id)
                if (
                    self._memory_config.refiner_enabled
                    and self._run_count % self._refine_interval == 0
                    and session_records
                ):
                    run_memory_refiner_with_timeout(
                        self._memory_manager, self._memory_config, session_records, user_id, self._llm_client
                    )
            except Exception as exc:
                logger.warning(f"记忆写入或清理失败: {exc}")

        return assistant_msg

    def _finish_session(self, user_message: Message, quitting: bool) -> Message:
        """处理退出指令，提炼记忆"""
        user_id = self._get_user_id(user_message)
        final_answer = "已结束本次对话，长期记忆已更新" if quitting else "本次对话结束"
        assistant_msg = Message(content=final_answer, role="assistant")
        self.add_history(user_message)
        self.add_history(assistant_msg)

        if self._memory_enabled:
            try:
                add_session_message(self._memory_manager, user_message.content, "user", user_id)
                add_session_message(self._memory_manager, final_answer, "assistant", user_id, score=0.9, importance=0.9)
                session_records = collect_records(user_message, list(self._history), final_answer, user_id)
                if self._memory_config.refiner_enabled:
                    run_memory_refiner_with_timeout(
                        self._memory_manager, self._memory_config, session_records, user_id, self._llm_client
                    )
                forget_if_needed(self._memory_manager, self._memory_config)
            except Exception as exc:
                logger.warning(f"退出时记忆处理失败: {exc}")

        return assistant_msg
    
    def _fallback_response(self, user_message: Message, context_text: str, err_desc: Optional[str] = None) -> str:
        """兜底回复"""
        logger.warning("触发兜底回复逻辑")
        user_query = f"用户问题: {user_message.content}\n"
        if err_desc:
            user_query = f"用户问题: {user_message.content}\n\n 对话过程中遇到的异常情况: {err_desc}\n\n"
        self._llm_client.invoke(
                            [Message(role="system", content=FALLBACK_PROMPT.format(history=context_text or "无",
                                                                                   query=user_query))],
                        )


__all__ = ["ChatAgent"]
