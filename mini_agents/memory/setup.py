'''
Description: 根据配置创建 MemoryManager 与默认存储
Author: zyq
Date: 2025-12-31 11:22:42
LastEditors: zyq
LastEditTime: 2026-01-04 16:37:44
'''
from __future__ import annotations

from typing import Optional

from mini_agents.core.config import MemoryConfig
from .manager import MemoryManager
from .stores import SessionMemoryStore, HybridLongTermMemoryStore, QdrantStore, SQLiteBackend
from .embedder.remote import RemoteEmbedder
from .base import BaseMemoryStore
from ..core.exceptions import MemoryException


def create_memory_manager(
    memory_config: Optional[MemoryConfig] = None,
    external_store: Optional[BaseMemoryStore] = None,
) -> MemoryManager:
    """
    基于配置创建 MemoryManager 并注册默认存储
    external_store 可用于注入外部向量库（如 QdrantStore 实现），为空则使用 HybridLongTermMemoryStore。
    """
    cfg = memory_config or MemoryConfig.from_env()
    manager = MemoryManager(rrf_k=cfg.rrf_k, default_top_k=cfg.default_top_k)

    # 会话存储(短期记忆, 与单次会话绑定)
    session_store = SessionMemoryStore(
        max_items=cfg.session_max_items,
        ttl_seconds=cfg.session_ttl_seconds,
    )
    manager.register_store(["session"], session_store)

    # 长期/语义存储
    if external_store:
        manager.register_store(["long_term", "semantic"], external_store)
    else:
        if not cfg.remote_embedding_model or not cfg.remote_embedding_base_url:
            raise MemoryException("创建记忆管理器失败：未配置远程嵌入服务（remote_embedding_model / remote_embedding_base_url）")
        embedder = RemoteEmbedder(
            model=cfg.remote_embedding_model,
            api_key=cfg.remote_embedding_api_key,
            base_url=cfg.remote_embedding_base_url,
            dim=cfg.remote_embedding_dim,
        )
        provider = cfg.memory_store_provider.lower()
        if provider == "qdrant":
            if cfg.remote_embedding_dim <= 0:
                raise MemoryException("QdrantStore 初始化失败：remote_embedding_dim 未配置或无效")
            q_store = QdrantStore(
                url=cfg.qdrant_url,
                api_key=cfg.qdrant_api_key or None,
                collection=cfg.qdrant_collection,
                embedder=embedder,
                vector_size=embedder.dim or cfg.remote_embedding_dim,
                prefer_grpc=cfg.qdrant_prefer_grpc,
                min_score=cfg.hybrid_min_score,
            )
            manager.register_store(["long_term", "semantic"], q_store)
        else:
            if not embedder:
                raise MemoryException("HybridLongTermMemoryStore 初始化失败：缺少远程嵌入器配置")
            hybrid_store = HybridLongTermMemoryStore(
                backend=SQLiteBackend(cfg.hybrid_db_path),
                embedder=embedder,
                top_k=cfg.hybrid_top_k,
                decay_lambda=cfg.hybrid_decay_lambda,
                min_score=cfg.hybrid_min_score,
            )
            manager.register_store(["long_term", "semantic"], hybrid_store)

    return manager
