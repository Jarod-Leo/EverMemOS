"""
UserProfile 原生 CRUD 仓库

基于 Beanie ODM 的用户画像数据访问层，提供完整的 CRUD 操作。
"""

from typing import Optional, Dict, Any, List
from core.observation.logger import get_logger
from core.di.decorators import repository
from core.oxm.mongo.base_repository import BaseRepository

from infra_layer.adapters.out.persistence.document.memory.user_profile import UserProfile

logger = get_logger(__name__)


@repository("user_profile_raw_repository", primary=True)
class UserProfileRawRepository(BaseRepository[UserProfile]):
    """
    UserProfile 原生 CRUD 仓库
    
    提供对 UserProfile 文档的直接数据库操作
    """
    
    def __init__(self):
        """初始化仓库"""
        super().__init__(UserProfile)
    
    async def get_by_user_and_group(
        self,
        user_id: str,
        group_id: str
    ) -> Optional[UserProfile]:
        """
        根据 user_id 和 group_id 获取用户画像
        
        Args:
            user_id: 用户ID
            group_id: 群组ID
            
        Returns:
            UserProfile 实例或 None
        """
        try:
            return await self.model.find_one(
                UserProfile.user_id == user_id,
                UserProfile.group_id == group_id
            )
        except Exception as e:
            logger.error(f"获取用户画像失败: user_id={user_id}, group_id={group_id}, error={e}")
            return None
    
    async def get_all_by_group(self, group_id: str) -> List[UserProfile]:
        """
        获取群组内所有用户画像
        
        Args:
            group_id: 群组ID
            
        Returns:
            UserProfile 列表
        """
        try:
            return await self.model.find(UserProfile.group_id == group_id).to_list()
        except Exception as e:
            logger.error(f"获取群组用户画像失败: group_id={group_id}, error={e}")
            return []
    
    async def upsert(
        self,
        user_id: str,
        group_id: str,
        profile_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[UserProfile]:
        """
        更新或插入用户画像
        
        Args:
            user_id: 用户ID
            group_id: 群组ID
            profile_data: 画像数据
            metadata: 元数据
            
        Returns:
            更新后的 UserProfile 实例或 None
        """
        try:
            metadata = metadata or {}
            existing = await self.get_by_user_and_group(user_id, group_id)
            
            if existing:
                # 更新现有文档
                existing.profile_data = profile_data
                existing.version += 1
                existing.confidence = metadata.get("confidence", existing.confidence)
                
                if "cluster_id" in metadata:
                    cluster_id = metadata["cluster_id"]
                    if cluster_id not in existing.cluster_ids:
                        existing.cluster_ids.append(cluster_id)
                    existing.last_updated_cluster = cluster_id
                
                if "memcell_count" in metadata:
                    existing.memcell_count = metadata["memcell_count"]
                
                await existing.save()
                logger.debug(f"更新用户画像: user_id={user_id}, group_id={group_id}, version={existing.version}")
                return existing
            else:
                # 创建新文档
                user_profile = UserProfile(
                    user_id=user_id,
                    group_id=group_id,
                    profile_data=profile_data,
                    scenario=metadata.get("scenario", "group_chat"),
                    confidence=metadata.get("confidence", 0.0),
                    version=1,
                    cluster_ids=[metadata["cluster_id"]] if "cluster_id" in metadata else [],
                    memcell_count=metadata.get("memcell_count", 0),
                    last_updated_cluster=metadata.get("cluster_id")
                )
                await user_profile.insert()
                logger.info(f"创建用户画像: user_id={user_id}, group_id={group_id}")
                return user_profile
        
        except Exception as e:
            logger.error(f"保存用户画像失败: user_id={user_id}, group_id={group_id}, error={e}")
            return None
    
    async def delete_by_group(self, group_id: str) -> int:
        """
        删除指定群组的所有用户画像
        
        Args:
            group_id: 群组ID
            
        Returns:
            删除的文档数量
        """
        try:
            result = await self.model.find(UserProfile.group_id == group_id).delete()
            count = result.deleted_count if result else 0
            logger.info(f"删除群组用户画像: group_id={group_id}, count={count}")
            return count
        except Exception as e:
            logger.error(f"删除群组用户画像失败: group_id={group_id}, error={e}")
            return 0
    
    async def delete_all(self) -> int:
        """
        删除所有用户画像
        
        Returns:
            删除的文档数量
        """
        try:
            result = await self.model.delete_all()
            count = result.deleted_count if result else 0
            logger.info(f"删除所有用户画像: {count} 条")
            return count
        except Exception as e:
            logger.error(f"删除所有用户画像失败: {e}")
            return 0

