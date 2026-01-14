"""
上下文工程入口
"""
from .models import ContextSourceType, ContextCandidateInfo, PipelineResult
from mini_agents.core.config import (
    ContextGatherConfig,
    ContextSelectConfig,
    ContextStructConfig,
    ContextCompressConfig,
    ContextPipelineConfig,
)
from .triggers import BaseTrigger, RoundLimitTrigger, TokenLimitTrigger
from .token_counter import TokenCounter
from .gssc.gather import Gatherer
from .gssc.selector import Selector
from .gssc.structor import Structor
from .gssc.compressor import Compressor
from .pipeline import ContextPipeline

__all__ = [
    "ContextSourceType",
    "ContextCandidateInfo",
    "ContextPipelineConfig",
    "ContextGatherConfig",
    "ContextSelectConfig",
    "ContextStructConfig",
    "ContextCompressConfig",
    "PipelineResult",
    "BaseTrigger",
    "RoundLimitTrigger",
    "TokenLimitTrigger",
    "TokenCounter",
    "Gatherer",
    "Selector",
    "Structor",
    "Compressor",
    "ContextPipeline",
]
