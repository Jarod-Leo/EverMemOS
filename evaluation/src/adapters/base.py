"""
Adapter 基类

定义记忆系统适配器的统一接口。
"""
from abc import ABC, abstractmethod
from typing import Any, List, Dict
from evaluation.src.core.data_models import Conversation, SearchResult


class BaseAdapter(ABC):
    """记忆系统适配器基类"""
    
    def __init__(self, config: dict):
        """
        初始化适配器
        
        Args:
            config: 系统配置字典
        """
        self.config = config
    
    @abstractmethod
    async def add(
        self, 
        conversations: List[Conversation],
        **kwargs
    ) -> Any:
        """
        摄入对话数据并构建索引（Add 阶段）
        
        这个方法封装了系统特定的数据摄入和索引构建逻辑：
        - 对于 EverMemOS: MemCell 提取 + BM25/Embedding 索引构建
        - 对于 Mem0: 直接存储到向量数据库
        - 对于其他系统: 各自的实现方式
        
        Args:
            conversations: 标准格式的对话列表
            **kwargs: 额外参数
            
        Returns:
            索引对象（系统内部格式，不同系统返回不同类型）
        """
        pass
    
    @abstractmethod
    async def search(
        self, 
        query: str,
        conversation_id: str,
        index: Any,
        **kwargs
    ) -> SearchResult:
        """
        检索相关记忆（Search 阶段）
        
        Args:
            query: 查询文本
            conversation_id: 对话 ID
            index: 索引对象（由 add() 返回）
            **kwargs: 额外参数（如 top_k）
            
        Returns:
            SearchResult: 标准格式的检索结果
        """
        pass
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        返回系统信息（用于结果记录）
        
        Returns:
            系统信息字典
        """
        return {
            "name": self.__class__.__name__,
            "config": self.config
        }

