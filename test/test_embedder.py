from mini_agents.memory.embedder.remote import RemoteEmbedder
from mini_agents.core.config import MemoryConfig

memory_cfg = MemoryConfig.from_env()

embedder = RemoteEmbedder(
    model=memory_cfg.remote_embedding_model,
    api_key=memory_cfg.remote_embedding_api_key,
    base_url=memory_cfg.remote_embedding_base_url, # xinference
    dim=memory_cfg.remote_embedding_dim,
)

vecs = embedder.embed(["向量服务测试"])

print(len(vecs), len(vecs[0]), vecs[0][:10])