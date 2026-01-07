'''
Description: reranker测试
Author: zyq
Date: 2026-01-06 18:02:07
LastEditors: zyq
LastEditTime: 2026-01-07 09:11:28
'''
from dotenv import load_dotenv

from mini_agents.rag.retrieval.reranker import RemoteReranker
from mini_agents.rag.models import DocumentChunk, RetrievalHit, ChunkMetadata
from mini_agents.core.llm import BaseLLMClient

load_dotenv(override=True)


def test_remote_reranker_model(mode: str = "default"):
    if mode == "default":
        reranker = RemoteReranker(mode=mode,
                                rerank_model="bge-reranker-v2-m3",
                                rerank_base_url="http://127.0.0.1:9997/v1")
    else:
        llm_client = BaseLLMClient()
        reranker = RemoteReranker(mode=mode, llm_client=llm_client)

    res = reranker.rerank(
        "明天天气怎么样?",
        [RetrievalHit(chunk=DocumentChunk(content="今天天气晴朗", metadata=ChunkMetadata()), score=0.8),
         RetrievalHit(chunk=DocumentChunk(content="明天预计下雨", metadata=ChunkMetadata()), score=0.8),
         RetrievalHit(chunk=DocumentChunk(content="我们一起学猫叫", metadata=ChunkMetadata()), score=0.8)]
    )
    
    print(res)

if __name__ == "__main__":
    test_remote_reranker_model("llm")
    