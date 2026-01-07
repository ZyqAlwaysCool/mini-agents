'''
Description: embedder
Author: zyq
Date: 2026-01-05 14:46:31
LastEditors: zyq
LastEditTime: 2026-01-06 17:24:27
'''

from __future__ import annotations

from typing import List
from loguru import logger
from openai import OpenAI


class RemoteEmbedder:
    def __init__(self, model: str, api_key: str, base_url: str, dim: int = 1024, timeout: int = 60):
        if not model or not base_url:
            raise ValueError("嵌入器初始化失败：缺少模型或 base_url")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.dim = dim
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=timeout)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        logger.info(f"触发embedder，请求条数={len(texts)}")
        res = self._client.embeddings.create(model=self.model, input=texts)
        vectors: List[List[float]] = [item.embedding for item in res.data]
        return vectors
