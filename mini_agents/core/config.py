'''
Description: 配置中心
Author: zyq
Date: 2025-12-25 16:24:10
LastEditors: zyq
LastEditTime: 2026-01-06 17:53:06
'''
import os
import json
from typing import Optional, Dict, Any, Literal, List
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

load_dotenv()

LLMProviders = Literal["openai"]

class GeneralConfig(BaseModel):
    # log
    log_level: str = "INFO"
    log_dir: str = "storage/logs"
    log_format: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "GeneralConfig":
        return cls(
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            log_dir=os.environ.get("LOG_DIR", "storage/logs"),
            log_format=os.environ.get("LOG_FMT", None)
        )
    
    def to_dict(self):
        return self.model_dump()


class MCPServerConfig(BaseModel):
    """MCP 服务端配置"""
    name: str
    base_url: str
    api_key: Optional[str] = None
    api_key_header: str = "X-API-Key"
    tools_path: str = "/tools"
    invoke_path: str = "/tools/{tool_name}"
    timeout: int = 10
    enabled: bool = True

    @field_validator("base_url", mode="before")
    @classmethod
    def _strip_slash(cls, v: str) -> str:
        return v.rstrip("/") if isinstance(v, str) else v


class MCPConfig(BaseModel):
    """MCP 服务端列表配置"""
    servers: List[MCPServerConfig] = []

    @classmethod
    def from_env(cls) -> "MCPConfig":
        servers: List[MCPServerConfig] = []

        idx = 1
        while True:
            base_url = os.environ.get(f"MCP_{idx}_BASE_URL", "").strip()
            if not base_url:
                break
            servers.append(
                MCPServerConfig(
                    name=os.environ.get(f"MCP_{idx}_NAME", f"mcp_{idx}"),
                    base_url=base_url,
                    api_key=os.environ.get(f"MCP_{idx}_API_KEY"),
                    api_key_header=os.environ.get(f"MCP_{idx}_API_KEY_HEADER", "X-API-Key"),
                    tools_path=os.environ.get(f"MCP_{idx}_TOOLS_PATH", "/tools"),
                    invoke_path=os.environ.get(f"MCP_{idx}_INVOKE_PATH", "/tools/{tool_name}"),
                    timeout=int(os.environ.get(f"MCP_{idx}_TIMEOUT", 10)),
                    enabled=os.environ.get(f"MCP_{idx}_ENABLED", "true").lower() == "true",
                )
            )
            idx += 1

        if not servers:
            base_url = os.environ.get("MCP_BASE_URL", "").strip()
            if base_url:
                servers.append(
                    MCPServerConfig(
                        name=os.environ.get("MCP_NAME", "mcp_default"),
                        base_url=base_url,
                        api_key=os.environ.get("MCP_API_KEY"),
                        api_key_header=os.environ.get("MCP_API_KEY_HEADER", "X-API-Key"),
                        tools_path=os.environ.get("MCP_TOOLS_PATH", "/tools"),
                        invoke_path=os.environ.get("MCP_INVOKE_PATH", "/tools/{tool_name}"),
                        timeout=int(os.environ.get("MCP_TIMEOUT", 10)),
                        enabled=os.environ.get("MCP_ENABLED", "true").lower() == "true",
                    )
                )
        return cls(servers=servers)

class LLMConfig(BaseModel):
    default_provider: LLMProviders = "openai"
    default_model: str = "qwen3-max"
    default_apikey: str
    default_base_url: str
    temperature: float = 0.7
    max_tokens: int = 1024
    
    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            default_provider=os.environ.get("DEFAULT_PROVIDER", "openai"),
            default_model=os.environ.get("DEFAULT_MODEL", "qwen3-max"),
            default_apikey=os.environ.get("DEFAULT_API_KEY", ""),
            default_base_url=os.environ.get("DEFAULT_BASE_URL", ""),
            temperature=float(os.environ.get("TEMPERATURE", 0.7)),
            max_tokens=int(os.environ.get("MAX_TOKENS", 1024))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

class AgentConfig(BaseModel):
    """Agent基础配置"""
    max_round: Optional[int] = 10 # agent执行的最大轮次数
    per_tool_call_timeout_ms: Optional[int] = 20000 # 单次工具调用超时
    per_llm_req_timeout_ms: Optional[int] = 20000 # 单次LLM调用超时

    @classmethod
    def from_env(cls) -> "AgentConfig":
        return cls(
            max_round=int(os.environ.get("AGENT_MAX_ROUND", 10)),
            per_tool_timeout_ms=int(os.environ.get("AGENT_PER_TOOL_CALL_TIMEOUT_MS", 20000)),
            per_llm_timeout_ms=int(os.environ.get("AGENT_PER_LLM_REQ_TIMEOUT_MS", 20000))
        )


class MemoryConfig(BaseModel):
    """记忆系统配置"""
    enable_memory: bool = True  # 是否启用记忆
    default_top_k: int = 5  # 检索默认返回条数
    forget_on_run_end: bool = True  # 每轮结束是否执行遗忘清理
    rrf_k: int = 60  # RRF 融合平滑参数
    memory_store_provider: str = "hybrid"  # 长期存储后端: hybrid|qdrant

    session_max_items: int = 200  # 短期容量
    session_ttl_seconds: int = 3600  # 短期 TTL 秒

    hybrid_db_path: str = "data/memory/hybrid.db"  # hybrid 路径
    hybrid_min_score: float = 0.3  # hybrid 衰减删除阈值
    hybrid_decay_lambda: float = 0.05  # hybrid 时间衰减系数
    hybrid_top_k: int = 8  # hybrid 内部 top_k

    remote_embedding_model: str = ""  # 远程嵌入模型
    remote_embedding_base_url: str = ""  # 远程嵌入服务地址
    remote_embedding_api_key: str = ""  # 远程嵌入 API Key
    remote_embedding_dim: int = 1024  # 远程嵌入维度

    qdrant_url: str = ""  # Qdrant 服务地址
    qdrant_api_key: str = ""  # Qdrant API Key
    qdrant_collection: str = "agent_memory"  # Qdrant 集合名
    qdrant_prefer_grpc: bool = False  # Qdrant 是否使用 gRPC

    refiner_enabled: bool = True  # 是否启用精炼
    refiner_batch_size: int = 20  # 精炼批大小（预留）
    refiner_timeout: float = 5.0  # 精炼线程等待超时（秒，<=0表示不等待）

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        return cls(
            enable_memory=os.environ.get("MEMORY_ENABLE", "true").lower() == "true",
            default_top_k=int(os.environ.get("MEMORY_DEFAULT_TOP_K", 5)),
            forget_on_run_end=os.environ.get("MEMORY_FORGET_ON_RUN_END", "true").lower() == "true",
            rrf_k=int(os.environ.get("MEMORY_RRF_K", 60)),
            session_max_items=int(os.environ.get("MEMORY_SESSION_MAX_ITEMS", 200)),
            session_ttl_seconds=int(os.environ.get("MEMORY_SESSION_TTL_SECONDS", 3600)),
            hybrid_db_path=os.environ.get("MEMORY_HYBRID_DB_PATH", "data/memory/hybrid.db"),
            hybrid_min_score=float(os.environ.get("MEMORY_HYBRID_MIN_SCORE", 0.3)),
            hybrid_decay_lambda=float(os.environ.get("MEMORY_HYBRID_DECAY_LAMBDA", 0.05)),
            hybrid_top_k=int(os.environ.get("MEMORY_HYBRID_TOP_K", 8)),
            memory_store_provider=os.environ.get("MEMORY_STORE_PROVIDER", "hybrid"),
            remote_embedding_model=os.environ.get("MEMORY_REMOTE_EMBEDDING_MODEL", ""),
            remote_embedding_base_url=os.environ.get("MEMORY_REMOTE_EMBEDDING_BASE_URL", ""),
            remote_embedding_api_key=os.environ.get("MEMORY_REMOTE_EMBEDDING_API_KEY", ""),
            remote_embedding_dim=int(os.environ.get("MEMORY_REMOTE_EMBEDDING_DIM", 1024)),
            qdrant_url=os.environ.get("MEMORY_QDRANT_URL", ""),
            qdrant_api_key=os.environ.get("MEMORY_QDRANT_API_KEY", ""),
            qdrant_collection=os.environ.get("MEMORY_QDRANT_COLLECTION", "agent_memory"),
            qdrant_prefer_grpc=os.environ.get("MEMORY_QDRANT_PREFER_GRPC", "false").lower() == "true",
            refiner_enabled=os.environ.get("MEMORY_REFINER_ENABLED", "true").lower() == "true",
            refiner_batch_size=int(os.environ.get("MEMORY_REFINER_BATCH_SIZE", 20)),
            refiner_timeout=float(os.environ.get("MEMORY_REFINER_TIMEOUT", 2.0)),
        )


class RAGConfig(BaseModel):
    """RAG 系统配置"""

    collection: str = ""  # 业务优先，空则按前缀+biz_id 生成
    collection_prefix: str = "rag_"
    top_k: int = 5
    fetch_k: int = 20
    enable_mqe: bool = True
    enable_hyde: bool = False
    enable_rerank: bool = False
    rerank_type: str = "default"  # llm|default
    metadata_mode: str = "llm"  # llm|rule

    chunk_size: int = 800
    chunk_overlap: int = 200
    chunker_type: str = "recursive"

    embedding_model: str = ""
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_dim: int = 1024

    rerank_model: str | None = None
    rerank_base_url: str | None = None
    rerank_api_key: str | None = None

    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_min_score: float = 0.3

    trace_enabled: bool = True

    @classmethod
    def from_env(cls) -> "RAGConfig":
        return cls(
            collection=os.environ.get("RAG_COLLECTION", "") or "",
            top_k=int(os.environ.get("RAG_TOP_K", 5)),
            fetch_k=int(os.environ.get("RAG_FETCH_K", 20)),
            enable_mqe=os.environ.get("RAG_ENABLE_MQE", "true").lower() == "true",
            enable_hyde=os.environ.get("RAG_ENABLE_HYDE", "false").lower() == "true",
            enable_rerank=os.environ.get("RAG_ENABLE_RERANK", "false").lower() == "true",
            rerank_type=os.environ.get("RAG_RERANK_TYPE", "llm"),
            metadata_mode=os.environ.get("RAG_METADATA_MODE", "llm"),
            chunk_size=int(os.environ.get("RAG_CHUNK_SIZE", 800)),
            chunk_overlap=int(os.environ.get("RAG_CHUNK_OVERLAP", 200)),
            chunker_type=os.environ.get("RAG_CHUNKER_TYPE", "recursive"),
            embedding_model=os.environ.get("RAG_EMBEDDING_MODEL", ""),
            embedding_base_url=os.environ.get("RAG_EMBEDDING_BASE_URL", ""),
            embedding_api_key=os.environ.get("RAG_EMBEDDING_API_KEY", ""),
            embedding_dim=int(os.environ.get("RAG_EMBEDDING_DIM", 1024)),
            rerank_model=os.environ.get("RAG_RERANK_MODEL"),
            rerank_base_url=os.environ.get("RAG_RERANK_BASE_URL"),
            rerank_api_key=os.environ.get("RAG_RERANK_API_KEY"),
            qdrant_url=os.environ.get("RAG_QDRANT_URL", ""),
            qdrant_api_key=os.environ.get("RAG_QDRANT_API_KEY", ""),
            qdrant_min_score=float(os.environ.get("RAG_QDRANT_MIN_SCORE", 0.3)),
            trace_enabled=os.environ.get("RAG_TRACE_ENABLED", "true").lower() == "true",
        )

    def collection_name(self, biz_name: str | None = None) -> str:
        biz = biz_name or "default"
        return f"rag_{biz}"


if __name__ == "__main__":
    llm_cfg = LLMConfig.from_env()
    print(llm_cfg.to_dict())

    general_cfg = GeneralConfig.from_env()
    print(general_cfg.to_dict())
