"""对外公开 API 路由模块。

只提供用户界面所需的公开企业资料展示，不暴露后台密钥、内部账号或敏感配置。
遇到未注册/不存在租户统一返回 HTTP 404，杜绝自动创建污染。
"""

from fastapi import APIRouter, HTTPException, Query

from back.domain.tenant import TenantPublicProfile, validate_tenant_id
from back.bootstrap import get_profile_service


router = APIRouter(prefix="/public", tags=["Public Profile"])


@router.get("/profile", response_model=TenantPublicProfile)
def get_public_profile(
    tenant_id: str = Query(
        default="default",
        description="租户ID",
    )
):
    """获取企业对公展示资料（名称、简介、欢迎语、联系方式等）。"""
    try:
        safe_tenant_id = validate_tenant_id(tenant_id)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err

    profile = get_profile_service().public(safe_tenant_id)
    if profile is None:
        raise HTTPException(
            status_code=404,
            detail=f"未找到租户 '{safe_tenant_id}' 的公开资料",
        )

    return profile
