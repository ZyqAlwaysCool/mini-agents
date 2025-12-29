'''
Description: 配置中心
Author: zyq
Date: 2025-12-25 16:24:10
LastEditors: zyq
LastEditTime: 2025-12-29 10:45:14
'''
import os
from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel
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
    

if __name__ == "__main__":
    llm_cfg = LLMConfig.from_env()
    print(llm_cfg.to_dict())

    general_cfg = GeneralConfig.from_env()
    print(general_cfg.to_dict())