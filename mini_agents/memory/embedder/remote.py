'''
Description: embedder
Author: zyq
Date: 2025-12-31 10:51:06
LastEditors: zyq
LastEditTime: 2026-01-04 16:02:35
'''

from __future__ import annotations
from loguru import logger

from typing import List
from openai import OpenAI
from .base import BaseEmbedder
from ...core.exceptions import MemoryException


class RemoteEmbedder(BaseEmbedder):
    """远程嵌入器，调用 OpenAI 兼容 embedding 接口"""

    def __init__(self, model: str, api_key: str, base_url: str, timeout: int = 30, dim: int = 0):
        if not model or not base_url:
            raise MemoryException("初始化 RemoteEmbedder 失败：缺少 model 或 base_url")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.name = "remote"
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        # dim 由第一次调用后推断，或由配置提前告知
        self.dim = dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"embedder: {texts}")
        if not texts:
            return []
        try:
            resp = self._client.embeddings.create(model=self.model, input=texts)
            embeddings: List[List[float]] = []
            for item in resp.data:
                emb = item.embedding
                embeddings.append(emb)
            if embeddings:
                self.dim = len(embeddings[0])
            return embeddings
        except Exception as e:
            raise MemoryException(f"远程嵌入调用失败: {e}") from e
