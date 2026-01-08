from datetime import datetime, timedelta
from pathlib import Path

from mini_agents.memory import (
    MemoryManager,
    MemoryRecord,
    MemoryQuery,
    SessionMemoryStore,
    HybridLongTermMemoryStore,
    create_memory_manager,
)
from mini_agents.memory.embedder.remote import RemoteEmbedder
from mini_agents.core.config import MemoryConfig
from mini_agents.core.llm import BaseLLMClient


def test_memory_manager_rrf_and_forget(tmp_path: Path):
    memory_cfg = MemoryConfig.from_env()
    embedder = RemoteEmbedder(
        model=memory_cfg.remote_embedding_model,
        api_key=memory_cfg.remote_embedding_api_key,
        base_url=memory_cfg.remote_embedding_base_url, # xinference
        dim=memory_cfg.remote_embedding_dim,
    )
    db_path = tmp_path / "hybrid.db"
    session_store = SessionMemoryStore(max_items=10, ttl_seconds=3600)
    hybrid_store = HybridLongTermMemoryStore(db_path=str(db_path), embedder=embedder, decay_lambda=0.0)

    manager = MemoryManager(rrf_k=60, default_top_k=5)
    manager.register_store(["session"], session_store)
    manager.register_store(["long_term", "semantic"], hybrid_store)

    # 写入短期记忆
    session_store.add(
        MemoryRecord(
            type="session",
            content="用户喜欢使用Python进行异步编程",
            metadata={"user_id": "u1", "role": "user"},
            importance=0.9,
            score=0.9,
            tags=["pref", "python"],
        )
    )
    # 写入长期记忆
    hybrid_store.add(
        MemoryRecord(
            type="long_term",
            content="多次提到用asyncio编写服务",
            metadata={"user_id": "u1", "role": "assistant"},
            importance=0.8,
            score=0.8,
            tags=["python"],
        )
    )
    # 过期记录用于测试清理
    expired = MemoryRecord(
        type="long_term",
        content="过期信息",
        metadata={"user_id": "u1"},
        score=0.1,
        expire_at=datetime.utcnow() - timedelta(days=1),
    )
    hybrid_store.add(expired)

    query = MemoryQuery(text="python 异步", user_id="u1", top_k=5)
    ctx = manager.inject_context(query)
    assert "[MEMORY_START]" in ctx and "[MEMORY_END]" in ctx
    assert "Python" in ctx or "asyncio" in ctx

    removed = manager.forget_all()
    assert any(expired.id in ids for ids in removed.values())

    ctx_after = manager.inject_context(query)
    assert "过期信息" not in ctx_after


def test_session_forget_with_ttl(tmp_path: Path):
    store = SessionMemoryStore(max_items=10, ttl_seconds=1)
    rec = MemoryRecord(type="session", content="soon expired")
    store.add(rec)
    removed = store.forget(datetime.utcnow() + timedelta(seconds=2))
    assert rec.id in removed


def test_session_search_with_bm25():
    store = SessionMemoryStore(max_items=10, ttl_seconds=100)
    rec1 = MemoryRecord(type="session", content="用户喜欢Python asyncio")
    rec2 = MemoryRecord(type="session", content="讨论数据库优化")
    store.add(rec1)
    store.add(rec2)
    hits = store.search(MemoryQuery(text="Python 异步", user_id=None, top_k=1))
    assert hits and hits[0].record.id == rec1.id


def test_refiner_promote_to_long_term(tmp_path: Path):
    db_path = tmp_path / "hybrid.db"
    hybrid_store = HybridLongTermMemoryStore(db_path=str(db_path), embedder=None, decay_lambda=0.0)
    session_store = SessionMemoryStore(max_items=10, ttl_seconds=3600)
    manager = MemoryManager()
    manager.register_store(["session"], session_store)
    manager.register_store(["long_term", "semantic"], hybrid_store)

    # 构造会话记录
    records = [
        MemoryRecord(type="session", content="用户提出需求：用Python做异步服务", metadata={"role": "user", "user_id": "u1"}),
        MemoryRecord(type="session", content="助手建议使用asyncio", metadata={"role": "assistant", "user_id": "u1"}),
    ]
    
    th = manager.refine_async(records, target_type="long_term", user_id="u1", llm_client=BaseLLMClient())
    if th:
        th.join(timeout=5)

    hits = manager.search(MemoryQuery(text="asyncio", user_id="u1", top_k=3, type_scope=["long_term"]))
    assert "asyncio" in hits[0].record.content


def test_hybrid_store_dedup(tmp_path: Path):
    db_path = tmp_path / "hybrid.db"
    hybrid_store = HybridLongTermMemoryStore(db_path=str(db_path), embedder=None, decay_lambda=0.0)
    rec1 = MemoryRecord(type="long_term", content="用户喜欢Python", metadata={"user_id": "u1"})
    rec2 = MemoryRecord(type="long_term", content="用户喜欢Python", metadata={"user_id": "u1"})
    hybrid_store.add(rec1)
    hybrid_store.add(rec2)
    rows = hybrid_store.backend.query_all(type_scope=["long_term"], user_id="u1")
    assert len(rows) == 1


def test_create_memory_manager(tmp_path: Path, monkeypatch):
    # 使用本地配置覆盖数据库路径与嵌入器
    monkeypatch.setenv("MEMORY_HYBRID_DB_PATH", str(tmp_path / "hybrid.db"))
    monkeypatch.setenv("MEMORY_SESSION_MAX_ITEMS", "5")
    monkeypatch.setenv("MEMORY_SESSION_TTL_SECONDS", "100")
    monkeypatch.setenv("MEMORY_REMOTE_EMBEDDING_MODEL", "dummy-embed")
    monkeypatch.setenv("MEMORY_REMOTE_EMBEDDING_BASE_URL", "http://127.0.0.1:9997/v1")
    monkeypatch.setenv("MEMORY_REMOTE_EMBEDDING_API_KEY", "dummy-key")
    monkeypatch.setenv("MEMORY_REMOTE_EMBEDDING_DIM", "4")
    mgr = create_memory_manager()
    # 至少包含 session 与 long_term/semantic 对应的存储
    assert len(mgr._stores) >= 2


if __name__ == "__main__":
    test_refiner_promote_to_long_term(Path("./"))