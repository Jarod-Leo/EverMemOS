"""单条消息记忆存储 Demo（超简单版）"""
import asyncio
import httpx
from demo_config import API_URL

# 修改为 memorize API
MEMORIZE_URL = API_URL.replace("/retrieve_lightweight", "/memorize")

# 单条消息配置
MESSAGE = {
    "group_id": "chat_user_001_assistant",
    "group_name": "用户健康咨询对话",
    "message_id": "test_msg_001",
    "create_time": "2025-11-09T23:00:00+08:00",
    "sender": "user_001",
    "sender_name": "测试用户",
    "content": "下周我会去北京旅游，可以给我一些建议吗？",
}


async def memorize():
    """存储单条消息"""
    print(f"\n📤 发送消息到 V3 API")
    print(f"   URL: {MEMORIZE_URL}")
    print(f"   内容: {MESSAGE['content']}\n")
    
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(MEMORIZE_URL, json=MESSAGE)
        result = resp.json()
        
        if result.get("status") == "ok":
            count = result.get("result", {}).get("count", 0)
            print(f"✅ 成功保存 {count} 条记忆")
            print(f"   消息: {result.get('message', '')}")
        else:
            print(f"❌ 失败: {result.get('message', '未知错误')}")
                


if __name__ == "__main__":
    asyncio.run(memorize())

