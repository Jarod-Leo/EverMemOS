"""
租户信息数据模型

本模块定义了租户相关的数据类，用于统一管理租户信息。
"""

import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


class TenantPatchKey(str, Enum):
    """
    租户信息 Patch 缓存键枚举

    用于统一管理 tenant_info_patch 中的缓存键，避免字符串硬编码导致的键名冲突。

    设计说明：
    - 所有 patch 缓存键都应该在这里定义
    - 使用枚举可以提供更好的类型提示和防止拼写错误
    - 继承自 str，使其可以直接作为字典 key 使用

    缓存键说明：
    - MONGO_CLIENT_CACHE_KEY: 用于缓存 MongoDB 客户端的 cache_key（连接参数的哈希值）
    - ACTUAL_DATABASE_NAME: 用于缓存实际使用的数据库名称
    - REAL_DATABASE_PREFIX: 用于缓存数据库对象的键前缀（实际使用时需要拼接数据库名）
    - ES_CONNECTION_CACHE_KEY: 用于缓存 Elasticsearch 连接别名的 cache_key
    """

    # MongoDB 客户端相关
    MONGO_CLIENT_CACHE_KEY = "mongo_client_cache_key"

    # MongoDB 数据库相关
    ACTUAL_DATABASE_NAME = "actual_database_name"
    MONGO_REAL_DATABASE = "mongo_real_database"  # 真实的 MongoDB 数据库对象

    # Milvus 连接相关
    MILVUS_CONNECTION_CACHE_KEY = "milvus_connection_cache_key"

    # Elasticsearch 连接相关
    ES_CONNECTION_CACHE_KEY = "es_connection_cache_key"


@dataclass
class TenantDetail:
    """
    租户详细信息数据类

    这个类用于存储经过适配后的租户详细信息，外部的租户信息会被适配成这个数据模型。

    Attributes:
        tenant_info: 租户相关信息的字典，存放转换后的租户数据
                    结构示例: {
                        "hash_key": "...",
                        "account_id": "...",
                        "space_id": "...",
                        "organization_id": "..."
                    }
        storage_info: 存储相关配置信息的字典，可选字段
                     结构示例: {
                         "mongodb": {"host": "...", "port": 27017, ...},
                         "redis": {"host": "...", "port": 6379, ...}
                     }
    """

    tenant_info: Optional[Dict[str, Any]] = field(default=None)
    storage_info: Optional[Dict[str, Any]] = field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        """
        将 TenantDetail 转换为字典

        Returns:
            Dict[str, Any]: 包含 tenant_info 和 storage_info 的字典
        """
        return {'tenant_info': self.tenant_info, 'storage_info': self.storage_info}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TenantDetail':
        """
        从字典创建 TenantDetail 实例

        Args:
            data: 包含 tenant_info 和 storage_info 的字典

        Returns:
            TenantDetail: 新创建的实例
        """
        return cls(
            tenant_info=data.get('tenant_info'), storage_info=data.get('storage_info')
        )


@dataclass
class TenantInfo:
    """
    租户信息数据类

    这个类是租户信息的主要数据模型，包含租户的核心信息。

    Attributes:
        tenant_id: 租户唯一标识符
        tenant_detail: 经过适配后的租户详细信息
        origin_tenant_data: 外部直接传入的原始租户数据，保持原样不做适配
        tenant_info_patch: 租户相关的缓存数据，用于存储计算后的值（如 actual_database_name, real_client 等）
                          生命周期与 tenant_info 一致，避免重复计算
    """

    tenant_id: str
    tenant_detail: TenantDetail
    origin_tenant_data: Dict[str, Any] = field(default_factory=dict)
    tenant_info_patch: Dict[str, Any] = field(default_factory=dict)

    def get_storage_info(self, storage_type: str) -> Optional[Dict[str, Any]]:
        """
        获取指定类型的存储配置信息

        此方法从 tenant_detail.storage_info 中获取指定类型的存储配置。

        Args:
            storage_type: 存储类型，如 "mongodb", "redis", "elasticsearch" 等

        Returns:
            指定存储类型的配置信息字典，如果不存在则返回 None

        Examples:
            >>> tenant_info = TenantInfo(
            ...     tenant_id="tenant_001",
            ...     tenant_detail=TenantDetail(storage_info={
            ...         "mongodb": {"host": "localhost", "port": 27017}
            ...     }),
            ...     origin_tenant_data={}
            ... )
            >>> mongo_config = tenant_info.get_storage_info("mongodb")
            >>> print(mongo_config)
            {'host': 'localhost', 'port': 27017}
        """
        # 检查 tenant_detail.storage_info 是否存在
        if self.tenant_detail.storage_info is None:
            return None

        # 从 storage_info 中获取指定类型的配置
        return self.tenant_detail.storage_info.get(storage_type)

    def get_patch_value(self, key: str, default: Any = None) -> Any:
        """
        从 tenant_info_patch 中获取缓存值

        此方法用于获取缓存在 tenant_info_patch 中的计算值，
        避免重复计算，提高性能。

        Args:
            key: 缓存键名
            default: 如果键不存在时的默认返回值

        Returns:
            缓存的值，如果不存在则返回 default

        Examples:
            >>> tenant_info = TenantInfo(
            ...     tenant_id="tenant_001",
            ...     tenant_detail=TenantDetail(),
            ...     origin_tenant_data={}
            ... )
            >>> # 获取缓存的数据库名称
            >>> db_name = tenant_info.get_patch_value(TenantPatchKey.ACTUAL_DATABASE_NAME)
        """
        return self.tenant_info_patch.get(key, default)

    def set_patch_value(self, key: str, value: Any) -> None:
        """
        设置 tenant_info_patch 中的缓存值

        此方法用于缓存计算结果，避免后续重复计算。
        缓存的生命周期与 TenantInfo 实例一致。

        Args:
            key: 缓存键名
            value: 要缓存的值

        Examples:
            >>> tenant_info = TenantInfo(
            ...     tenant_id="tenant_001",
            ...     tenant_detail=TenantDetail(),
            ...     origin_tenant_data={}
            ... )
            >>> # 缓存数据库名称
            >>> tenant_info.set_patch_value(TenantPatchKey.ACTUAL_DATABASE_NAME, "tenant_001_db")
            >>> # 缓存客户端的 cache_key
            >>> tenant_info.set_patch_value(TenantPatchKey.MONGO_CLIENT_CACHE_KEY, "cache_key_value")
        """
        self.tenant_info_patch[key] = value

    def has_patch_value(self, key: str) -> bool:
        """
        检查 tenant_info_patch 中是否存在指定的缓存值

        Args:
            key: 缓存键名

        Returns:
            如果存在返回 True，否则返回 False

        Examples:
            >>> tenant_info = TenantInfo(
            ...     tenant_id="tenant_001",
            ...     tenant_detail=TenantDetail(),
            ...     origin_tenant_data={}
            ... )
            >>> tenant_info.set_patch_value(TenantPatchKey.ACTUAL_DATABASE_NAME, "tenant_001_db")
            >>> tenant_info.has_patch_value(TenantPatchKey.ACTUAL_DATABASE_NAME)
            True
            >>> tenant_info.has_patch_value("non_existent_key")
            False
        """
        return key in self.tenant_info_patch

    def clear_patch_cache(self) -> None:
        """
        清除所有缓存数据

        在某些情况下可能需要清除缓存（如租户配置更新时），
        此方法会清空 tenant_info_patch 中的所有数据。

        Examples:
            >>> tenant_info = TenantInfo(
            ...     tenant_id="tenant_001",
            ...     tenant_detail=TenantDetail(),
            ...     origin_tenant_data={}
            ... )
            >>> tenant_info.set_patch_value("key1", "value1")
            >>> tenant_info.clear_patch_cache()
            >>> tenant_info.has_patch_value("key1")
            False
        """
        self.tenant_info_patch.clear()

    def invalidate_patch(self, key: Optional[str] = None) -> bool:
        """
        使缓存失效（删除特定 key 或全部）

        用于刷新或删除 tenant_info_patch 中的缓存对象。
        当缓存的资源需要重建（如连接断开、配置变更）时调用此方法。

        Args:
            key: 要删除的缓存键名。如果为 None，则清除所有缓存。

        Returns:
            bool: 如果指定了 key，返回该 key 是否存在并被删除；
                  如果 key 为 None（清除全部），总是返回 True。

        Examples:
            >>> tenant_info = TenantInfo(
            ...     tenant_id="tenant_001",
            ...     tenant_detail=TenantDetail(),
            ...     origin_tenant_data={}
            ... )
            >>> tenant_info.set_patch_value(TenantPatchKey.MONGO_REAL_DATABASE, some_db_obj)
            >>> # 删除特定缓存
            >>> tenant_info.invalidate_patch(TenantPatchKey.MONGO_REAL_DATABASE)
            True
            >>> # 清除所有缓存
            >>> tenant_info.invalidate_patch()
            True
        """
        if key is None:
            # 清除所有缓存
            self.tenant_info_patch.clear()
            return True
        else:
            # 删除特定 key
            if key in self.tenant_info_patch:
                del self.tenant_info_patch[key]
                return True
            return False

    # ==================== 序列化/反序列化方法 ====================
    # 用于异步任务传递租户上下文到其他进程

    def to_dict(self) -> Dict[str, Any]:
        """
        将 TenantInfo 转换为字典（用于序列化）

        注意：tenant_info_patch 不会被序列化，因为其中可能包含不可序列化的对象（如数据库连接）。
        反序列化后 tenant_info_patch 会被初始化为空字典。

        Returns:
            Dict[str, Any]: 包含 tenant_id、tenant_detail、origin_tenant_data 的字典
        """
        return {
            'tenant_id': self.tenant_id,
            'tenant_detail': self.tenant_detail.to_dict(),
            'origin_tenant_data': self.origin_tenant_data,
        }

    def to_json(self) -> str:
        """
        将 TenantInfo 序列化为 JSON 字符串

        用于异步任务传递租户上下文到其他进程。

        注意：tenant_info_patch 不会被序列化。

        Returns:
            str: JSON 字符串

        Examples:
            >>> tenant_info = TenantInfo(
            ...     tenant_id="t1234567890abcdef",
            ...     tenant_detail=TenantDetail(tenant_info={"account_id": "acc_001"}),
            ...     origin_tenant_data={"X-Organization-Id": "org_001"}
            ... )
            >>> json_str = tenant_info.to_json()
        """
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TenantInfo':
        """
        从字典创建 TenantInfo 实例

        Args:
            data: 包含 tenant_id、tenant_detail、origin_tenant_data 的字典

        Returns:
            TenantInfo: 新创建的实例（tenant_info_patch 为空字典）
        """
        tenant_detail_data = data.get('tenant_detail', {})
        return cls(
            tenant_id=data['tenant_id'],
            tenant_detail=TenantDetail.from_dict(tenant_detail_data),
            origin_tenant_data=data.get('origin_tenant_data', {}),
            # tenant_info_patch 不从序列化数据恢复，初始化为空
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'TenantInfo':
        """
        从 JSON 字符串反序列化创建 TenantInfo 实例

        用于异步任务从其他进程接收租户上下文。

        Args:
            json_str: JSON 字符串

        Returns:
            TenantInfo: 新创建的实例（tenant_info_patch 为空字典）

        Examples:
            >>> json_str = '{"tenant_id": "t1234567890abcdef", ...}'
            >>> tenant_info = TenantInfo.from_json(json_str)
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
