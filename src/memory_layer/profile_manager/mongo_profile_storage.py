"""MongoDB-based profile storage for ProfileManager."""

from typing import Any, Dict, List, Optional

from memory_layer.profile_manager.storage import ProfileStorage
from core.observation.logger import get_logger
from core.di.decorators import component
from core.di import get_bean_by_type
from infra_layer.adapters.out.persistence.repository.user_profile_raw_repository import (
    UserProfileRawRepository,
)

logger = get_logger(__name__)


@component()
class MongoProfileStorage(ProfileStorage):
    """MongoDB-based profile storage implementation.
    
    使用 UserProfileRawRepository 进行数据库操作。
    """
    
    def __init__(self):
        """初始化 MongoDB Profile 存储"""
        logger.info("MongoProfileStorage initialized")
    
    def _get_repo(self) -> UserProfileRawRepository:
        """获取 repository 实例"""
        return get_bean_by_type(UserProfileRawRepository)
    
    async def save_profile(
        self,
        user_id: str,
        profile: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """保存用户画像"""
        metadata = metadata or {}
        group_id = metadata.get("group_id", "default")
        
        # 将 ProfileMemory 对象转换为字典
        if hasattr(profile, 'to_dict'):
            profile_data = profile.to_dict()
        elif isinstance(profile, dict):
            profile_data = profile
        else:
            profile_data = {"data": str(profile)}
        
        result = await self._get_repo().upsert(user_id, group_id, profile_data, metadata)
        return result is not None
    
    async def get_profile(self, user_id: str, group_id: str = "default") -> Optional[Any]:
        """获取用户画像"""
        user_profile = await self._get_repo().get_by_user_and_group(user_id, group_id)
        if user_profile is None:
            return None
        return user_profile.profile_data
    
    async def get_all_profiles(self, group_id: str = "default") -> Dict[str, Any]:
        """获取群组内所有用户画像"""
        user_profiles = await self._get_repo().get_all_by_group(group_id)
        return {up.user_id: up.profile_data for up in user_profiles}
    
    async def get_profile_history(
        self,
        user_id: str,
        group_id: str = "default",
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """获取用户画像历史版本（当前实现只返回最新版本）"""
        user_profile = await self._get_repo().get_by_user_and_group(user_id, group_id)
        if user_profile is None:
            return []
        
        history = [{
            "version": user_profile.version,
            "profile": user_profile.profile_data,
            "confidence": user_profile.confidence,
            "updated_at": user_profile.updated_at,
            "cluster_id": user_profile.last_updated_cluster,
            "memcell_count": user_profile.memcell_count
        }]
        return history[:limit] if limit else history
    
    async def clear(self, group_id: Optional[str] = None) -> bool:
        """清除用户画像"""
        repo = self._get_repo()
        if group_id is None:
            await repo.delete_all()
        else:
            await repo.delete_by_group(group_id)
        return True
