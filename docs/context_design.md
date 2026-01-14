# 上下文工程组件设计（GSSC流水线）

## 1. 目标与原则
- 目标：提供可插拔的上下文工程组件，持续压缩高密度信息，避免长对话上下文腐蚀，确保少量 token 取得高质量输出。
- 原则：接口统一、实现简洁、默认可用；上下文模板分区结构固定以便压缩后仍可使用；容错单数据源失败不中断；可扩展（替换打分/压缩策略）；不侵入现有 Agent，提供框架层通用能力。

## 1.1 记忆分层与分区原则
- 长期记忆（long_term）：由精炼产生的摘要，作为背景知识直接填充在系统 prompt 的“相关记忆片段”分区，默认不进入压缩流水线，避免被重复截断。
- 短期记忆（session）：对话过程中累积的原始信息，进入压缩流水线并作为 `ContextSourceType.short_memory` 出现在“查询关联候选区”，可被压缩/裁剪。
- 记忆格式保持 `[MEMORY_START] ... [MEMORY_END]` 包裹，压缩后的短期记忆也沿用此格式，保证分区结构一致。

## 2. 核心数据模型
- ContextSourceType
  - system_prompt
  - short_memory
  - long_memory
  - rag
  - tool_result
  - history
  - others

- ContextCandidateInfo
  - content: str，候选内容
  - timestamp: int | str，时间戳或 ISO 字符串
  - token_count: int，预估或精确 token 数
  - relevance_score: float 0~1，来源侧的重要性/相关度
  - metadata: dict，含 user_id、来源标识等
  - type: ContextSourceType，来源类别
  - priority: int，系统指令固定高优先级，其余可默认 0
  - id: str，唯一键用于去重追踪, 使用content的Md5值作为去重id

## 3. 触发策略（Trigger）
- RoundLimitTrigger：对话轮次 >= N 触发, 轮次n可以通过形参注入配置.
- TokenLimitTrigger：历史 token 估算 >= 预算触发。
- Trigger 接口：`should_trigger(state) -> bool`，由业务在循环中注入 state，自由组合触发器。

## 4. Token 计数
- TokenCounter 接口：`count(text) -> int`。
- 默认实现：tiktoken，本地安装即用，无需模型或远程依赖；若缺失需安装依赖后再使用。

## 5. GSSC 流水线
### 5.1 Gather（收集）
- 多源适配器：系统指令、对话历史、记忆（短期/长期）、RAG、工具输出等。
- 容错：单源异常 try-except 记录日志，不阻断整体。
- 优先级：系统指令强制高优先级，必须保留。
- 历史限制：对话历史仅取最近 n 条。
- 阈值：低于最低得分阈值的候选丢弃。
- 记忆来源策略：
  - 短期记忆：仅 session 类型进入 Gather 阶段，标记为 `short_memory`，供后续压缩。
  - 长期记忆：不进入 Gather，由外层 prompt 直接填充到“相关记忆片段”分区。
- 输出统一的 ContextCandidateInfo 列表。

### 5.2 Select（筛选打分）
- 相似度：cosine(query, info) ∈ [0,1]（使用现有 embedding）。
- 新近性：分段线性衰减，24 小时内得分 1，之后按线性衰减至 0。
- 综合得分：`score = w1 * 相似度 + (1 - w1) * 新近性`，w1 默认 0.7，配置可调。
- 排序：先按优先级分桶（系统指令固定保留），桶内按综合得分排序，按 top-k 或 token 上限截断。
- 去重：相似内容可合并，避免重复 token。

### 5.3 Struct（结构化）
- 按 type 分区固定模板，示例：
  1) 系统指令区（必填）
  2) 工具/环境约束区
  3) 当前查询相关候选区（RAG/记忆/重要历史摘要）
  4) 近期对话片段区（未摘要的最近 n 轮）
  5) 长期摘要区
- 分区内可带简短提示语，模板固定，便于压缩保持结构。

### 5.4 Compress（兜底压缩）
- 若结构化结果超 token 上限：
  1) 先按分区优先级/得分删减最低分候选（系统指令区不可删）。
  2) 若仍超限，对超限分区调用 LLM 摘要，保持分区标题与顺序不变，限制输出长度。
- 压缩需确保分区结构完整，不跨区合并内容；摘要指令需明确“保持分区、保留关键实体/约束”。

## 6. 组件与接口草案
- ContextPipeline：组合 Trigger/Gather/Select/Struct/Compress，提供 `run(query, state)` 返回结构化上下文字符串及中间产物。
- Gatherer：多源注册与容错；可扩展新来源适配器。
- Selector：实现打分、分桶、top-k/token 截断、去重。
- Structor：根据模板与 type 映射生成分区上下文。
- Compressor：分区优先级删减 + 分区摘要压缩。
- TokenCounter：使用 tiktoken。
- 配置：w1、新近性阈值、历史条数、最低分阈值、top-k、token 上限、分区优先级等，通过统一配置对象暴露，减少调用侧模板代码。

## 7. 测试要点
- 触发器：轮次/预算达到与未达到的判定。
- Gather：单源异常不阻断、阈值过滤、系统指令必留、历史截断。
- Select：分段新近性、综合得分排序、优先级分桶、top-k/token 截断、去重。
- Struct：type 分区映射正确、结构稳定。
- Compress：超限时分区优先级删减与分区内摘要后结构仍完整；未超限不触发压缩。

## 8. 集成原则
- 先独立模块，不改现有 Agent；未来在 Agent 循环中以 Trigger 驱动调用 Pipeline。
- 接口保持统一与简洁，减少调用侧样板代码，所有策略通过配置与替换组件完成扩展。
