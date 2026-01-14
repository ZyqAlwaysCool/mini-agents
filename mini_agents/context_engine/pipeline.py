'''
Description: 上下文工程流水线
Author: zyq
Date: 2026-01-09 09:31:54
LastEditors: zyq
LastEditTime: 2026-01-09 15:19:23
'''

from __future__ import annotations

from typing import Dict, List

from loguru import logger

from .models import PipelineResult
from .triggers import BaseTrigger
from .gssc.gather import Gatherer
from .gssc.selector import Selector
from .gssc.structor import Structor
from .gssc.compressor import Compressor


class ContextPipeline:
    def __init__(
        self,
        triggers: List[BaseTrigger],
        gatherer: Gatherer,
        selector: Selector,
        structor: Structor,
        compressor: Compressor,
    ):
        self.triggers = triggers
        self.gatherer = gatherer
        self.selector = selector
        self.structor = structor
        self.compressor = compressor

    def run(self, query: str, state: Dict) -> PipelineResult:
        triggered = any(t.should_trigger(state) for t in self.triggers) if self.triggers else True # 判断是否需要触发上下文压缩
        if not triggered:
            return PipelineResult(triggered=False, context_text="", gathered=[], selected=[], sections={})
        gathered = self.gatherer.gather(state)
        selected = self.selector.select(query, gathered)
        context_text, sections = self.structor.structure(selected)
        compressed_sections, final_text = self.compressor.compress(sections)
        logger.debug(f"上下文压缩后长度: {len(final_text)}")
        return PipelineResult(
            triggered=True,
            context_text=final_text,
            gathered=gathered,
            selected=selected,
            sections=compressed_sections,
        )
