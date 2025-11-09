"""简单的记忆检索 Demo（50行以内）"""
import asyncio
import httpx
from demo.demo_config import API_URL, SEARCH_CONFIG


async def retrieve():
    """执行检索"""
    print(f"\n🔍 检索: {SEARCH_CONFIG['query']}")
    print(f"📋 配置: {SEARCH_CONFIG['data_source']} | {SEARCH_CONFIG['retrieval_mode']} | {SEARCH_CONFIG['memory_scope']}\n")
    
    async with httpx.AsyncClient(timeout=60.0, verify=False) as client:
        resp = await client.post(API_URL, json=SEARCH_CONFIG)
        result = resp.json()

        memories = result.get("result", {}).get("memories", [])
        metadata = result.get("result", {}).get("metadata", {})
        
        print(f"✅ 找到 {len(memories)} 条记忆 (耗时: {metadata.get('total_latency_ms', 0):.0f}ms)\n")
        
        for i, mem in enumerate(memories, 1):
            score = mem.get('score', 0)
            content = (mem.get('episode') or mem.get('content') or mem.get('atomic_fact', ''))[:100]
            mem_type = mem.get('memory_sub_type', 'unknown')
            event_id = mem.get('event_id', 'N/A')
            
            print(f"[{i}] 分数: {score:.4f} | 类型: {mem_type}")
            print(f"    ID: {event_id}")
            if mem.get('subject'):
                print(f"    主题: {mem['subject']}")
            print(f"    内容: {content}...\n")
                



if __name__ == "__main__":
    asyncio.run(retrieve())
