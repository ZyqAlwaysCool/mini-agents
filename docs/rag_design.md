# RAG 设计方案

## 1. 目标与原则
- 提供独立、可插拔的 RAG 组件，与 agents/memory/tools 平级，可单独调用或由 Agent 挂载。
- 分层解耦：用户层/应用层/处理层/存储层/基础层，各层可替换。
- 默认优先质量：MarkItDown 统一抽取、规则+滑窗分块、LLM 元数据（可降级）、基础语义检索 + 可选 MQE/HyDE + 可选 rerank。
- 配置驱动：全部远程调用（OpenAI 兼容三元组：base_url/model/api_key），禁止本地模型硬编码。
- 可扩展：Chunker/Store/Reranker/QueryExpander 均用接口抽象，新增实现只需继承基类注册。

## 2. 分层架构
1) 用户层：统一 API（同步/异步）  
2) 应用层：问答/搜索/知识管理  
3) 处理层：文档加载 → 分块 → 元数据增强 → 向量化 → 写库  
4) 检索层：查询扩展 → 召回 → 去重 → rerank（可选） → top-k  
5) 存储层：Qdrant（默认，按业务 collection），预留其他向量库适配  
6) 基础层：远程嵌入、远程 rerank/LLM（OpenAI 兼容），MarkItDown

## 3. 目录规划（新增）
```
mini_agents/rag/
  models.py           # 数据模型：DocumentChunk/Metadata/Query/Hit/RAGResult 等
  pipeline.py         # RAGPipeline：data_process/retrieval 统一编排
  data_process/
    loader.py         # DocumentLoader：MarkItDown 转 markdown
    chunker.py        # BaseChunker + DefaultRecursiveChunker（langchain-text-splitters）
    metadata.py       # MetadataEnricher：LLM JSON + 规则降级
    embedder.py       # RemoteEmbedder（RAG 独立实现）
    indexer.py        # Indexer：向量化后写入 store
  retrieval/
    expander.py       # QueryExpander：MQE/HyDE 插件化
    retriever.py      # Retriever：主召回，含过滤与去重
    reranker.py       # BaseReranker + LightReranker + RemoteReranker
    aggregator.py     # 去重/合并/裁剪
  store/
    base.py           # BaseVectorStore 接口
    qdrant_store.py   # 默认 Qdrant 实现，按业务 collection
  agent_integration/
    adapter.py        # Agent 注入段落封装（[RAG_START...END]）
```

## 4. 数据模型（models.py）
- DocumentChunk：id、doc_id、chunk_index、content、md5、metadata（source_path/title/section/tags/keywords/summary/potential_questions/created_at/extra）、embedding（可空）。
- Query：text、biz_id/biz_name（用于 collection）、filters（tags/时间范围/来源）、top_k、fetch_k、enable_mqe/enable_hyde、enable_rerank、rerank_mode。
- QueryCandidate：query_text、source（user/mqe/hyde）、score。
- RetrievalHit：chunk、score、reason、source_query。
- RAGResult：hits（top-k）、trace（策略开关/耗时/降级信息）。

## 5. 配置（core/config.py 中的 RAGConfig，env 前缀示例）
- collection：`rag_{biz_id}` 默认，允许传入自定义名。
- 嵌入：`RAG_EMBEDDING_MODEL`、`RAG_EMBEDDING_BASE_URL`、`RAG_EMBEDDING_API_KEY`、`RAG_EMBEDDING_DIM`。
- Qdrant：`RAG_QDRANT_URL`、`RAG_QDRANT_API_KEY`、`RAG_QDRANT_MIN_SCORE`、`RAG_QDRANT_COLLECTION_PREFIX`（默认 `rag_`）。
- 分块：`RAG_CHUNK_SIZE`（默认 800 字符）、`RAG_CHUNK_OVERLAP`（默认 200）、`RAG_WINDOW_SIZE`（滑窗 0/关闭）、`RAG_CHUNKER_TYPE`（默认 recursive，可扩展）。
- 元数据：`RAG_METADATA_MODE`（llm|rule），`RAG_METADATA_ENABLE_FALLBACK`（默认 true）。
- 检索：`RAG_TOP_K`（默认 5）、`RAG_FETCH_K`（默认 20）、`RAG_ENABLE_MQE`（默认 true）、`RAG_ENABLE_HYDE`（默认 false）。
- rerank：`RAG_ENABLE_RERANK`（默认 false）、`RAG_RERANK_MODE`（light|llm）、`RAG_RERANK_MODEL/BASE_URL/API_KEY`。
- 观测：日志级别复用 GeneralConfig；Pipeline 输出 trace。

## 6. 组件设计
### 6.1 Data Process
- DocumentLoader：基于 MarkItDown，将多格式文档转 markdown；输出统一文本与结构化块（标题/代码块保留）。
- BaseChunker：接口 `chunk(text, metadata) -> List[DocumentChunk]`，方便后续新增实现。
- DefaultRecursiveChunker：
  - 底层使用 `langchain-text-splitters` 的 `RecursiveCharacterTextSplitter`。
  - 规则：标题优先切分、代码块整块保留、正文按长度切分，支持滑窗补上下文。
  - 参数：chunk_size/chunk_overlap/window_size，可通过 config 调整。
- MetadataEnricher：
  - 默认 LLM 模式：提示词要求严格 JSON，字段：`summary`、`keywords`(1-4)、`potential_questions`(1-3)。
  - 失败或关闭时走规则模式：首段截取摘要 + TF-IDF 关键词（简易实现）+ 可选问句模板。
  - 可配置是否启用降级。
- RemoteEmbedder（RAG 专属）：OpenAI 兼容 `embeddings` 接口，使用 RAG 前缀配置，不依赖 memory 的实现。
- Indexer：负责调用 embedder、构造 payload、写入 store，捕获失败重试/跳过，并返回写入统计。

### 6.2 Store
- BaseVectorStore：接口 `upsert(chunks)`、`query(query_text, top_k, filters)`、`delete/clear`、`ensure_collection`。
- QdrantStore：
  - 使用独立 RAG 配置，按 `collection_prefix + biz_id/biz_name` 命名。
  - payload 含 chunk 元数据、来源、版本、md5。
  - 支持 tags/时间范围过滤，min_score 兜底。

### 6.3 Retrieval
- QueryExpander：
  - MQE：LLM 生成多路改写，严格 JSON 数组；默认开启，数量可配。
  - HyDE：LLM 生成虚拟文档再 embed，默认关闭，可配数量与模板。
  - 扩展结果去重（文本 hash）后进入检索。
- Retriever：
  - 对候选查询逐个向量检索，合并结果；支持 filters、fetch_k。
  - 去重：按 chunk_id/md5，近似重复可用向量阈值（可配）。
- Reranker：
  - LightReranker（默认）：基于召回相似度 + 关键词重合度，统一归一化。
  - RemoteReranker：OpenAI 兼容接口打分，输出 JSON 分数；开关控制。
- Aggregator：融合各策略结果，按 rerank/召回得分排序，输出 top_k 与原因。

### 6.4 Pipeline
- `ingest(doc_path, biz_id, options)`：加载 → 分块 → 元数据 → 向量化 → 写库 → 返回统计。
- `retrieve(query, biz_id, options)`：扩展查询 → 召回 → 去重 → rerank（可选） → 返回 top-k + trace。
- `generate`（预留）：检索后将片段交给上层 Agent/LLM 生成，不在本轮实现。

### 6.5 Agent 集成
- 工具化接入：通过 `build_rag_tool(pipeline, name, description)` 将 RAGPipeline 暴露为通用工具，工具内部负责检索并返回 `{"trace": {...}, "rag_context": "[RAG_START..END]", "results": [...]}`，`rag_context` 使用 `RAGPromptAdapter` 构造。
- 通用注入：ReActAgent 的 ExecuteNode 在工具返回中发现 `rag_context` 即更新共享态并写入 history，DecideNode 在下一轮 Prompt 中注入最新片段；Agent 本身不预检索、不耦合具体知识库名称。
- 多知识库：可注册多个 RAG 工具（通过不同 name/description 标示领域），工具内固定或传参 `biz_id/collection`，无须修改 Agent 逻辑即可复用。
- Prompt 注入：`[RAG_START]` 每条命中含来源/分数/摘要片段 `[..] [RAG_END]`，与 Memory 注入隔离；提示语明确“需要业务知识时调用对应 RAG 工具”。

## 7. 依赖
- 新增：`markitdown`、`langchain-text-splitters`、`qdrant-client`（已存在版本可复用）、`rank_bm25`（用于规则关键词/轻量 rerank）。全部通过 uv 安装到 .venv。

## 8. 观测与健壮性
- Trace 记录：每步耗时、开关状态、降级/重试信息、命中理由。
- LLM/Embedding/Rerank 接口：严格 JSON 解析，失败自动降级或跳过并写日志。
- 去重策略：hash + chunk_id + 可选向量阈值，避免重复片段挤占 top-k。

## 9. 测试清单
- Ingestion：多格式转 md、分块数量/边界、元数据 LLM/降级路径、Qdrant 写入。
- Retrieval：MQE 开关、HyDE 开关、filters 生效、去重、rerank light/llm、top_k 截断。
- Agent 集成：开启/关闭 RAG、与 Memory 同时存在的提示注入正确。
- 回归：配置缺失时的报错与降级、远程接口超时重试。

## 10. 实施步骤
1) 落地目录与配置/模型定义；实现 RemoteEmbedder、Store 接口、Chunker 基类 + 默认实现、MetadataEnricher、Indexer、QueryExpander、Retriever、Reranker、Pipeline、Agent 适配。
2) 补充 uv 依赖与示例配置（.env 样例）。
3) 编写单元测试（test/test_rag_xxx.py），覆盖 ingestion、retrieval、开关组合与 Agent 注入。
4) 更新 README/usage 简要指引。***
