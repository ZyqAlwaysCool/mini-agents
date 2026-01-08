## 简介
mini-agents 是一款基于 OpenAI 兼容接口的轻量级智能体框架，支持本地工具与 MCP 远程工具的统一注册调用，内置 ReAct、simple、plan_act 等经典智能体示例，开箱即用. 后续将持续迭代，增强智能体核心能力. 

市面上的 Agent 框架封装度高，上手快但理解成本大（如 langgraph 的 create_react_agent）；本项目以最小可用实现呈现核心设计与链路，减少黑箱依赖，帮助快速理解 Agent 模式与 LLM 相关技术（RAG / Memory / MCP）. 

项目思路参考：HelloAgents（https://github.com/jjyaoao/HelloAgents）

编码辅助：codex

## 目录结构
mini_agents:
* core: 定义基类组件, 实现参考HelloAgents.
* tools: 定义工具管理基类, 包括本地工具调用和远端mcp工具调用实现, 交由统一的ToolExecutor管理工具注册、调用.
* test：包含多个不同的调用示例.
* agents: 包含多个具体的agent实现，主要通过pocketflow来做同步/异步版本的实现. pocketflow可视为一个工作流编排器，源码仅有100行，可作为workflow/agent设计的最小载体. 在它的基础上做能力扩展. pocketflow见：https://github.com/The-Pocket/PocketFlow
* memory: 包含短期/长期记忆功能的实现, 可配置在agent中, 目前已在react agent中适配记忆能力.
* rag: 实现基础rag pipeline, 支持以独立组件的方式提供数据处理+向量检索服务, 也可通过工具注册的方式, 注入到agent中. 参考`test_rag_pipeline.py`和`test_react_rag_tool.py`实现. 需提前配置embedding模型、向量存储数据库地址，推荐均以base_url、apikey的方式做远端调用，不在本地直接拉起模型.


## 依赖安装
1) 创建虚拟环境（推荐使用uv / python venv），并安装依赖：
```bash
uv pip install -r requirements.txt
```

2) 如需使用rag和memory组件, 需要额外安装以下依赖组件:
    * qdrant: 向量存储使用, 建议以docker方式部署
    * xinference: 部署embedding模型、rerank模型

## 配置
在项目根目录准备 `.env`，根目录下提供的`.env.example`为配置模板, 所有配置项详情见mini_agents/core/config.py, 常用变量如下：
```ini
# [default]
DEFAULT_PROVIDER="openai"
DEFAULT_MODEL="qwen3-max"
DEFAULT_API_KEY="your-api-key"
DEFAULT_BASE_URL="http://{your_openai_ip}/v1"

# [llm] if no default set
LLM_MODEL_ID="qwen3-max"
LLM_API_KEY="sk-..."
LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

# MCP多实例（从1开始递增, 不连续序号会被截断）
MCP_1_NAME=...
MCP_1_BASE_URL=...
MCP_1_API_KEY=...
MCP_1_API_KEY_HEADER=X-API-Key
MCP_1_TIMEOUT=10
MCP_1_ENABLED=true

# MCP单实例
MCP_NAME="..."
MCP_BASE_URL="..."
MCP_API_KEY="..."
MCP_ENABLED=true


# 日志
LOG_LEVEL=INFO
```
说明：
- MCP实例按编号递增 (`MCP_1_`、`MCP_2_`...). 若只需单实例，可使用 `MCP_BASE_URL` 等单组变量. 
- MCP工具注册后标准名称格式为 `mcp::<server>::<tool>`，`server` 来自 `MCP_1_NAME` 等配置. 
- default 组变量用于 `LLMConfig.from_env()` 的全局默认模型/Provider/地址；`LLM_*` 是底层 `BaseLLMClient` 的兜底配置（兼容已有环境变量）. 如果两者同时存在，`BaseLLMClient` 优先取 `LLM_*`，否则回落到 default 组. 
- memory、rag配置项见.env.example示例文件

## 快速开始
同步示例：
```python
from mini_agents.tools.mcp import MCPToolManager
from mini_agents.tools.base import ToolExecutor
from mini_agents.core.config import AgentConfig
from mini_agents.core.message import Message
from mini_agents.core.llm import BaseLLMClient
from mini_agents.agents.react_agent import ReActAgent

# 测试用, 实际替换成请求天气的API接口
@tool_register()
def get_weather(location: str) -> str:
    """
    获取指定地点的天气情况
    :param location: 地点
    :return 当地的天气情况
    """
    return f"{location}的天气晴朗, 温度为25度"

MCPToolManager().register(ToolExecutor)
llm = BaseLLMClient()
agent = ReActAgent(name="react", llm_client=llm, agent_config=AgentConfig, tool_executor=ToolExecutor)
resp = agent.run(Message(role="user", content="北京的天气情况如何?"))
print(resp)
```

异步示例：
```python
from mini_agents.tools.mcp import MCPToolManager
from mini_agents.tools.base import ToolExecutor
from mini_agents.core.config import AgentConfig
from mini_agents.core.message import Message
from mini_agents.core.llm import BaseLLMClient
from mini_agents.agents.react_agent import ReActAgent
import asyncio

# 测试用, 实际替换成请求天气的API接口
@tool_register()
def get_weather(location: str) -> str:
    """
    获取指定地点的天气情况
    :param location: 地点
    :return 当地的天气情况
    """
    return f"{location}的天气晴朗, 温度为25度"

async def main():
    await MCPToolManager().register_async(ToolExecutor)
    llm = BaseLLMClient()
    agent = ReActAgent(name="react", llm_client=llm, agent_config=AgentConfig, tool_executor=ToolExecutor)
    resp = await agent.run_async(Message(role="user", content="北京的天气情况如何?"))
    print(resp)

asyncio.run(main())
```

## Tool工具管理要点
- 本地工具(function call)通过 `@tool_register` 装饰器注册，MCP工具通过 `MCPToolManager` 批量注册，需提前配置mcp server的信息. 注册完成后统一由 `ToolExecutor`做管理调度. 
- 可选白名单：`ToolExecutor.set_allowed_tools([...])`，仅对已成功注册的工具生效，未注册的名称会被自动忽略. 
- MCP 预检连接失败会记录日志并跳过该服务，不会阻断其他工具注册. 

### 测试
运行内置示例测试：
```bash
python test/test_react_agent.py
pytest -q test/test_react_agent.py
```
（需事先配置好 LLM/MCP 网络与鉴权）

## 记忆组件(Memory)
具体的设计文档说明在docs/memory_design.md中, 记忆组件包括以下几个核心设计: 
- 组件：
    - SessionMemory: 短期记忆，驻内存，支持 TTL/容量淘汰
    - HybridLongTermMemory: 长期记忆，默认 sqlite+向量存储, 向量化需要部署xinference等服务调用, 非本地embedding模型
    - MemoryManager: 检索聚合+RRF 排序+遗忘衰减
    - MemoryRefiner: 通过LLM将单次对话中的短期记忆做提炼, 把有价值的信息结构化成长期记忆存储在sqlite或其他存储后端中
- 嵌入(embedder)：支持远程嵌入，需配置 `MEMORY_REMOTE_EMBEDDING_MODEL / BASE_URL / API_KEY / DIM`，缺失配置会报错. 
- 去重：长期记忆(LongTermMemory)写入前按 `(user_id, type, content)` 去重，避免重复记忆膨胀. 
- 遗忘(forget)：`forget_on_run_end` 控制 run 结束是否触发记忆的衰减清理, 模拟人会遗忘一些不重要的记忆, `refiner_timeout` 控制记忆精炼线程的等待时长，超时会告警但线程继续执行. 

### 执行流程(以ReAct agent+Memory为示例)
0) 测试文件: `test/test_memory_react_llm.py`
1) 请求前：根据用户输入构造 `MemoryQuery` 检索记忆，生成 `[MEMORY_START...END]` 片段注入 ReAct 提示. 
2) 运行中：用户消息/工具 observation/模型思考依次写入短期记忆，用于当前轮决策与后续检索；历史保存在 `history` 便于调试. 
3) 结束时：写入本轮最终回答到短期记忆中；若开启精炼，启动精炼线程，将本轮关键信息摘要打标写入长期记忆存储（默认 sqlite）. 

## RAG组件
RAG组件的设计文档在docs/rag_design.md中, 该文档通过与codex进行多轮对话+设计调整, 形成最终的设计方案. RAG组件将基础RAG流程分解成两个阶段:
1. 数据预处理: 将原始输入的文档统一转换为markdown格式, 通过分块处理+元数据增强，形成多个文本块，向量化后存储在目标向量库中.
    * 元数据增强: 默认使用LLM做语义增强, 使用LLM对原文本块内容做摘要、关键词、相关问题集, 丰富原文本块的语义信息.
2. 检索召回: 将输入的原始查询通过MQE+HyDE两个检索策略做查询扩展, 召回候选文本块, 同时提供rerank模型的配置选择, 可做二次精排. 
    * 多查询扩展(MQE): 指将原输入通过LLM做多角度改写, 丰富召回候选池和相关度.
    * 假设性文档嵌入(HyDE): 指先用LLM对输入问题做一轮回答(a1), 再使用a1去与向量库做相似度匹配, 召回与假设性答案a1相关的文本块.

各阶段均以组件化的形式提供对外服务, 支持扩展和独立接入, 不与其他能力相互耦合.

## 公共组件包(mini_agents/common)
此公共包是一个可选项, 封装了memory、rag、tool组件中的一些基础能力, 目的是减少业务逻辑中的重复模板代码, 可自由选择使用common包或是调用组件能力. test目录下包含有相关示例. 

## 常见问题
- 事件循环：同步环境用 `register`/`run`，异步环境用 `register_async`/`run_async`，避免在已有事件循环中调用同步接口. 
- 网络失败：若日志提示 MCP 预检或 LLM 连接失败，请检查内网可达性、DNS、代理、防火墙设置. 未连通时 MCP 会被跳过，可能导致白名单过滤后无可用工具. 
- 输出不符预期：实测下来，私有服务器部署的诸如14B、32B量化版模型的效果不佳，即使是线上版本的模型, 参数量较少的版本如qwen3-30b-a3b-instruct-2507在以react agent模式解决复杂时仍然无法正确回答问题. 而切换至线上的qwen3-max后输出符合预期. 因此, 如果发现模型回答过傻，或是无法按照要求输出，可以考虑切换模型测试或者微调系统提示词.
