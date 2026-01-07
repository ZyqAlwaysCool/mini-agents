'''
Description: 
Author: zyq
Date: 2026-01-05 14:58:10
LastEditors: zyq
LastEditTime: 2026-01-07 09:37:57
'''
import tempfile
from pathlib import Path
from dotenv import load_dotenv

from mini_agents.rag.pipeline import RAGPipeline
from mini_agents.rag.models import Query
from mini_agents.rag.store.qdrant_store import QdrantVectorStore
from mini_agents.core.config import RAGConfig, LLMConfig
from mini_agents.core.llm import BaseLLMClient
from mini_agents.rag.retrieval.reranker import RemoteReranker

load_dotenv(override=True)

def test_qdrant():
    from qdrant_client import QdrantClient
    cli = QdrantClient(url="http://127.0.0.1:6333")
    points, _ = cli.scroll(collection_name="rag_demo", limit=100)
    print(points)

def test_rerank():
    import requests

    resp = requests.post(
        "http://127.0.0.1:9997/v1/rerank",
        json={
            "model": "bge-reranker-v2-m3",
            "query": "明天天气怎么样？",
            "documents": ["今天天气晴朗", "明天预计下雨", "我们一起学猫叫"],
            "top_n": 2
        }
    )
    print(resp.json()["results"])

def test_rag_pipeline_ingest_and_retrieve():
    cfg = RAGConfig.from_env()
    llm_cfg = LLMConfig.from_env()
    store = QdrantVectorStore(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key)
    llm_client = BaseLLMClient()
    reranker = RemoteReranker(rerank_model=cfg.rerank_model, rerank_api_key=cfg.rerank_api_key, rerank_base_url=cfg.rerank_base_url)
    pipeline = RAGPipeline(rag_config=cfg, store=store, llm_client=llm_client, reranker=reranker)

    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "demo.txt"
        p.write_text("这是第5段内容。\n这里是第6段内容，用于检索测试。", encoding="utf-8")
        stats = pipeline.ingest(str(p), biz_name="demo")
        assert stats["written"] > 0

        q = Query(text="检索第2段内容", biz_id="demo", top_k=3, fetch_k=2, enable_mqe=True, enable_hyde=True, enable_rerank=True)
        res = pipeline.retrieve(q)
        for h in res.hits:
            print(h)
            print('='* 20)
        assert res.hits
        assert res.trace["collection"].startswith("rag_")


if __name__ == "__main__":
    test_rag_pipeline_ingest_and_retrieve()
    
    # test_rerank()
