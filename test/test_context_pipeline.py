import pytest
from datetime import datetime, timedelta

from mini_agents.context_engine import (
    ContextCandidateInfo,
    ContextSourceType,
    Gatherer,
    Selector,
    Structor,
    Compressor,
    ContextPipeline,
    RoundLimitTrigger,
    TokenLimitTrigger,
)
from mini_agents.context_engine.token_counter import TokenCounter
from mini_agents.core.config import (
    ContextGatherConfig,
    ContextSelectConfig,
    ContextStructConfig,
    ContextCompressConfig,
)


def test_round_limit_trigger():
    trigger = RoundLimitTrigger(3)
    assert trigger.should_trigger({"round": 3})
    assert not trigger.should_trigger({"round": 2})


def test_token_limit_trigger():
    trigger = TokenLimitTrigger(token_limit=5, token_counter=TokenCounter())
    state = {"history_text": "hello world. this is a history text. "}
    assert trigger.should_trigger(state)

def test_token_counter():
    tc = TokenCounter()
    print(tc.count("hello world"))
    print(tc.count("你好世界"))


def test_gatherer_filters_and_limits_history():
    cfg = ContextGatherConfig(min_score=0.5, history_limit=3)
    gatherer = Gatherer(cfg)

    def good_fetcher(_):
        return [
            ContextCandidateInfo(content="系统指令", type=ContextSourceType.system_prompt, priority=10),
            ContextCandidateInfo(content="旧历史1", type=ContextSourceType.history, timestamp=1, relevance_score=0.9),
            ContextCandidateInfo(content="旧历史2", type=ContextSourceType.history, timestamp=2, relevance_score=0.8),
            ContextCandidateInfo(content="旧历史3", type=ContextSourceType.history, timestamp=3, relevance_score=0.7),
            ContextCandidateInfo(content="低分候选", type=ContextSourceType.rag, relevance_score=0.2),
        ]

    def bad_fetcher(_):
        raise RuntimeError("boom")

    gatherer.register("bad", bad_fetcher)
    gatherer.register("good", good_fetcher)

    res = gatherer.gather({})
    assert any(c.type == ContextSourceType.system_prompt for c in res)
    histories = [c for c in res if c.type == ContextSourceType.history]
    assert len(histories) == 3
    assert histories[0].content == "旧历史3"
    assert histories[1].content == "旧历史2"
    assert all(c.token_count > 0 for c in res)


def test_selector_priority_and_token_limit():
    cfg = ContextSelectConfig(top_k=1, token_limit=50, weight_similarity=0.7)
    selector = Selector(cfg, token_counter=TokenCounter())

    now = datetime.utcnow()
    cands = [
        ContextCandidateInfo(content="系统指令", type=ContextSourceType.system_prompt, priority=100, token_count=5),
        ContextCandidateInfo(content="关于Python异步", type=ContextSourceType.rag, priority=1, timestamp=now, token_count=10),
        ContextCandidateInfo(content="无关内容", type=ContextSourceType.rag, priority=0, timestamp=now - timedelta(days=2), token_count=10),
        ContextCandidateInfo(content="关于Python异步", type=ContextSourceType.rag, priority=1, timestamp=now, token_count=10),
    ]
    res = selector.select("Python", cands)
    assert any(c.type == ContextSourceType.system_prompt for c in res)
    non_sys = [c for c in res if c.type != ContextSourceType.system_prompt]
    assert len(non_sys) == 1
    assert "Python" in non_sys[0].content


def test_structor_grouping_and_history_cap():
    cfg = ContextStructConfig(history_turns=1)
    structor = Structor(cfg)
    cands = [
        ContextCandidateInfo(content="这是一个测试的系统指令\n换行测试系统指令", type=ContextSourceType.system_prompt),
        ContextCandidateInfo(content="工具结果", type=ContextSourceType.tool_result),
        ContextCandidateInfo(content="相关检索", type=ContextSourceType.rag),
        ContextCandidateInfo(content="历史1", type=ContextSourceType.history),
        ContextCandidateInfo(content="历史2", type=ContextSourceType.history),
        ContextCandidateInfo(content="长期摘要", type=ContextSourceType.long_memory),
    ]
    context_text, sections = structor.structure(cands)
    print(context_text)
    assert "系统指令区" in context_text
    assert "工具与环境约束区" in context_text
    assert len(sections["recent_history"]) == 1
    assert sections["recent_history"][0].content in {"历史1", "历史2"}


def test_compressor_reduces_tokens_by_priority_and_trim():
    cfg = ContextCompressConfig(token_limit=20, summary_token=5)
    compressor = Compressor(cfg, token_counter=TokenCounter())
    sections = {
        "system": [ContextCandidateInfo(content="系统", token_count=2)],
        "constraints": [
            ContextCandidateInfo(content="约束内容A", relevance_score=0.1, token_count=10),
            ContextCandidateInfo(content="约束内容B", relevance_score=0.2, token_count=10),
        ],
        "recent_history": [
            ContextCandidateInfo(content="历史内容较长 " * 5, relevance_score=0.5, token_count=30),
        ],
        "query_related": [],
        "long_term": [ContextCandidateInfo(content="长期摘要", relevance_score=0.9, token_count=8)],
    }
    new_sections, text = compressor.compress(sections)
    print(new_sections)
    print("======")
    print(text)
    total_tokens = sum(item.token_count for items in new_sections.values() for item in items)
    assert total_tokens <= cfg.token_limit
    assert "系统" in text


def test_pipeline_integration():
    gatherer = Gatherer(ContextGatherConfig(min_score=0.0, history_limit=3))
    selector = Selector(ContextSelectConfig(top_k=3, token_limit=200))
    structor = Structor(ContextStructConfig(history_turns=2))
    compressor = Compressor(ContextCompressConfig(token_limit=200))
    pipeline = ContextPipeline(
        triggers=[RoundLimitTrigger(1)],
        gatherer=gatherer,
        selector=selector,
        structor=structor,
        compressor=compressor,
    )

    def fetcher(_):
        return [
            ContextCandidateInfo(content="系统指令", type=ContextSourceType.system_prompt),
            ContextCandidateInfo(content="检索结果", type=ContextSourceType.rag, relevance_score=0.9),
            ContextCandidateInfo(content="最近对话", type=ContextSourceType.history, timestamp=datetime.utcnow().isoformat()),
        ]

    gatherer.register("default", fetcher)

    result = pipeline.run("用户查询", {"round": 1})
    assert result.triggered
    assert "系统指令" in result.context_text
    assert "检索结果" in result.context_text



if __name__ == "__main__":
    #test_token_limit_trigger()
    #test_token_counter()
    #test_gatherer_filters_and_limits_history()
    #test_selector_priority_and_token_limit()
    #test_structor_grouping_and_history_cap()
    #test_compressor_reduces_tokens_by_priority_and_trim()
    test_pipeline_integration()
