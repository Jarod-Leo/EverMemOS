"""Demo 配置文件"""

# V3 API 地址
API_URL = "http://localhost:8001/api/v3/agentic/retrieve_lightweight"

# 检索配置
SEARCH_CONFIG = {
    "query": "北京旅游美食",                    # 查询文本
    "user_id": "user_001",                      # 用户ID（可选）
    "top_k": 5,                                 # 返回数量
    "data_source": "memcell",                   # memcell | event_log | semantic_memory
    "retrieval_mode": "rrf",                    # embedding | bm25 | rrf
    "memory_scope": "all",                      # all | personal | group
}
    