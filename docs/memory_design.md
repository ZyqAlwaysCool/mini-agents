# 记忆系统设计方案

## 1. 目标与原则
- 目标：提供可插拔记忆子系统，支持多类型记忆、混合检索、RRF 融合排序、异步精炼与动态衰减，易于嵌入现有 Agent。
- 原则：低耦合（接口组合）、可扩展（类型/嵌入/策略可插拔）、可落地（默认 sqlite + 远程 embedding），可观测（检索/清理/精炼有结构化日志），可维护（代码直观，初中级开发者易读）。

## 2. 架构总览
- MemoryManager 作为统一网关，负责：
  - 聚合多类型存储的读写与检索。
  - 融合排序（RRF）与结果格式化。
  - 触发遗忘/衰减与异步精炼。
- 存储分层：
  - SessionMemoryStore：短期/工作记忆，驻内存。
  - HybridLongTermMemoryStore：长期+语义混合存储，基于 sqlite，支持向量与元数据过滤（无向量时退化为文本检索）。
- 其他组件：
  - Embedder：远程 embedding 提供者。
  - MemoryRefiner：异步精炼、去重、摘要、打标与提升。
  - Decay 逻辑：基于时间/访问计数/重要性动态衰减。

## 3. 核心数据模型
- MemoryType：`session` | `long_term` | `semantic` | `custom`（长期与语义由 HybridLongTermMemoryStore 承载，类型字段保留便于过滤）。
- MemoryRecord：
  - 基础：`id`(uuid)、`type`、`content`、`metadata`（含 role/session_id/tags/source 等）
  - 评分/衰减：`importance`(0~1)、`score`(置信度)、`access_count`、`created_at`、`last_accessed_at`、`expire_at`
  - 检索：`embedding`(可空)、`tags`
- MemoryQuery：
  - `text`、`user_id`、`top_k`、`type_scope`、`filters`（tags/时间范围/role 等）、`include_raw_history`
- MemoryHit：
  - `record`、`score`（融合后得分）、`source_type`、`reason`（命中原因说明，便于观测）

## 4. 模块与目录
- `mini_agents/memory/`
  - `base.py`：数据模型、接口、异常、通用打分工具。
  - `manager.py`：MemoryManager，检索聚合、RRF 融合、格式化、衰减、调度精炼。
  - `stores/`
    - `session_store.py`：短期记忆。
    - `hybrid_store.py`：长期+语义混合存储（sqlite，含文本与向量列）。
    - `__init__.py`：工厂/注册。
  - `embedder/`
    - `base.py`：BaseEmbedder。
    - `remote.py`：RemoteEmbedder（OpenAI 兼容接口）。
  - `refiner.py`：MemoryRefiner（异步精炼、摘要、去重、打标与类型提升）。
  - `config.py`：MemoryConfig（容量、TTL、权重、路径、嵌入配置、衰减参数）。

## 5. 记忆类型实现
- SessionMemoryStore
  - 数据结构：`deque`/list，配置 `max_items`、`ttl_seconds`。
  - 淘汰：超容量 FIFO；`ttl` 过期惰性清理；记录 role/source/step。
  - 场景：维持对话连续性，执行后可清空或沉淀到长期。
- HybridLongTermMemoryStore
  - 存储：sqlite 表包含文本、metadata(json)、score、importance、access_count、created_at/last_accessed_at/expire_at、tags、embedding(json)。
  - 写入：可直接写文本/结构化摘要；若有 embedder 则生成 embedding。
  - 检索（混合）：
    - 向量检索：余弦相似度（无 embedding 时跳过）。
    - 文本检索：关键词/BM25 占位实现。
    - 元数据过滤：tags/时间范围/user_id/type_scope。
    - store 内部先对向量、文本结果分别排序，再合并。
  - 作用：长期事实、用户画像、知识片段。

## 6. 检索与排序
- 单 store 得分：
  - 向量：`sim_score`，时间加成 `fresh_bonus`，重要性/置信度加成。
  - 文本：`text_score`（BM25/关键词），同样叠加新鲜度与重要性。
- 融合（跨 store）：使用 RRF（Reciprocal Rank Fusion）按排名融合，避免不同量纲的分值冲突。默认参数 `k=60`。
- Prompt 注入：
- MemoryManager 提供 `build_memory_context`，输出结构化片段：
  ```
  [MEMORY_START]
  - 用户偏好: ... (时间=2026-01-13T10:00:00, source=session, score=0.82)
  - 历史事件: ... (时间=2026-01-12T09:30:00, source=long_term, score=0.75)
  [MEMORY_END]
  ```
  - 注入时附带写入时间，便于 LLM 判断时序与冲突；保留 `reason` 便于日志观测。

## 7. 遗忘与衰减
- 衰减函数示例：`decay_score = score * exp(-lambda * age_days) * (1 + log1p(access_count)) * (0.5 + importance)`。
- Session：ttl + 容量淘汰。
- Hybrid：到期或衰减后分数低于阈值的记录在 `forget_all` 中删除/归档；`access_count` 每次命中自增，`last_accessed_at` 更新。
- MemoryManager 提供 `forget_all(now)`，在 run 后或定期触发。

## 8. 异步精炼 (MemoryRefiner)
- 触发：Agent 响应返回后异步执行，或定期批处理。
- 任务：
  - 去重/冲突检测：相似内容合并或版本化。
  - 压缩摘要：长对话/观测生成 1-2 句事实。
  - 打标：抽取 tags、来源、置信度。
  - 类型提升：Session -> Hybrid（long_term/semantic）。
- 实现：`refiner.run_async(records, embedder, llm_client=None)`，可选使用 LLM 做摘要/标签；若无长期价值信息，LLM 允许返回空数组，框架会记录原因日志但不落库；失败时记录日志不影响主流程。
- 写库前去重：
  - 先对内容做精确去重。
  - 再使用 TF-IDF 余弦相似度去重（阈值默认 0.8），仅保留代表性记录，避免长列表出现大量近似表述（例如“用户是Jay”的多条变体）。

## 9. Agent 接入
- BaseAgent 增加可选 `memory_manager`。
- 运行前：构造 MemoryQuery（含 user_id/type_scope/top_k/filters），调用 `build_memory_context` 将记忆片段插入系统提示的专属段落。
- 运行中：用户消息、模型思考、工具 observation 写入 Session；必要时即时写入 Hybrid（重要事件/知识）。
- 运行后：调用 `refiner` 执行提升；调用 `forget_all` 做清理；需要时 `clear_session` 释放短期。
- ReActAgent：在提示构造阶段插入 `[MEMORY_START...END]`，工具 observation 追加为 Session 记忆，run 结束后触发精炼与遗忘。

## 10. 嵌入配置（远程）
- 仅支持 RemoteEmbedder（OpenAI 兼容接口），需要提供模型、Base URL、API Key（若服务无需鉴权可填占位值）。
- MemoryManager/HybridStore 强制注入远程 embedder，缺失配置将直接报错提醒部署 embedding 服务。

## 11. 配置建议 (MemoryConfig)
- 开关与通用：`enable_memory=True`，`default_top_k=5`，`forget_on_run_end=True`。
- Session：`session_max_items=200`，`session_ttl_seconds=3600`。
- Hybrid：`hybrid_db_path="data/memory/hybrid.db"`，`hybrid_min_score=0.3`，`hybrid_decay_lambda=0.05`，`hybrid_top_k=8`。
- RRF：`rrf_k=60`。
- Embedder（必填）：`MEMORY_REMOTE_EMBEDDING_MODEL`，`MEMORY_REMOTE_EMBEDDING_BASE_URL`，`MEMORY_REMOTE_EMBEDDING_API_KEY`（可为占位值），`MEMORY_REMOTE_EMBEDDING_DIM`。
- Refiner：`refiner_enabled=True`（开启记忆且启用精炼时，run 结束自动触发精炼），`refiner_batch_size`，`refiner_timeout`（精炼线程 join 超时，>0 时阻塞等待，超时记录 warning）。
- 去重：HybridLongTermMemoryStore 写入前按 `(user_id, type, content)` 检查重复，重复内容不再写入。

## 12. 测试与验收
- Session：写入/检索/TTL 淘汰。
- Hybrid：sqlite 持久化，tag/时间过滤，向量检索，衰减删除。
- 融合：构造多 store 命中，验证 RRF 排序稳定性。
- Refiner：长对话摘要/打标/去重后写入 Hybrid。
- Agent 集成：ReActAgent 在“无记忆/仅 Session/混合记忆”三种配置下，Prompt 注入与输出可用。

## 13. 后续扩展
- 异步存储接口（适配高并发）。
- 外部向量库适配（faiss/chroma/pinecone）通过 HybridStore 接口替换。
- 访问计数与排序 A/B 实验，动态调整衰减系数。
- 导出/同步：将 Hybrid 数据同步到外部知识库或搜索引擎。 
