# -*- coding: utf-8 -*-
"""
租户上下文检查装饰器

用于要求接口必须携带有效的租户上下文（X-Organization-Id 和 X-Space-Id）。
"""

from functools import wraps
from typing import Callable, Any

from fastapi import HTTPException

from core.tenants.tenant_contextvar import get_current_tenant_id


def require_tenant(func: Callable) -> Callable:
    """
    要求租户上下文装饰器

    用于装饰 Controller 的接口方法，确保请求携带了有效的租户上下文。
    如果没有租户上下文，返回 400 错误。

    使用示例：
        @post("/init-db")
        @require_tenant
        async def init_tenant_database(self) -> TenantInitResponse:
            tenant_id = get_current_tenant_id()
            # tenant_id 一定不为 None
            ...

    Args:
        func: 被装饰的异步函数

    Returns:
        Callable: 包装后的函数
    """

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        # 检查租户上下文
        tenant_id = get_current_tenant_id()
        if not tenant_id:
            raise HTTPException(
                status_code=400,
                detail="缺少租户上下文，请确保请求携带了 X-Organization-Id 和 X-Space-Id",
            )

        # 调用原函数
        return await func(*args, **kwargs)

    return wrapper
