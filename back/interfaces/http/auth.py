"""网关认证：仅处理接入凭证与租户授权。"""
import json
import os
import secrets
from fastapi import HTTPException
from back.domain.tenant import validate_tenant_id

def get_public_chat_tenants() -> set[str]:
    """获取显式允许无需凭证公开访问的租户白名单集合。"""
    raw = os.getenv("PUBLIC_CHAT_TENANTS", "").strip()
    if not raw:
        return set()
    try:
        if raw.startswith("["):
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return {str(item).strip() for item in parsed if str(item).strip()}
        return {item.strip() for item in raw.split(",") if item.strip()}
    except Exception:
        return {item.strip() for item in raw.split(",") if item.strip()}

def verify_chat_tenant_authorization(
    requested_tenant_id: str,
    x_tenant_token: str | None = None,
) -> str:
    """校验聊天端租户访问权限与站点凭证绑定，采用默认拒绝策略防止越权。"""
    # 1. 单租户部署模式：由服务端环境变量锁定租户
    locked_tenant = os.getenv("APP_TENANT_ID", "").strip()
    if locked_tenant:
        try:
            safe_locked = validate_tenant_id(locked_tenant)
        except ValueError as err:
            raise HTTPException(
                status_code=503,
                detail=f"服务端 APP_TENANT_ID 配置不合法：{err}",
            ) from err
        if requested_tenant_id != safe_locked:
            raise HTTPException(
                status_code=403,
                detail=f"当前站点已锁定为租户 '{safe_locked}'，拒绝跨租户访问 '{requested_tenant_id}'",
            )
        # 锁定只限制可访问的租户；其访问凭据仍由下面的规则校验。

    # 2. 多租户受控模式：严格默认拒绝策略
    access_keys_env = os.getenv("TENANT_ACCESS_KEYS", "").strip()
    if access_keys_env:
        try:
            access_keys: dict[str, str] = json.loads(access_keys_env)
            if not isinstance(access_keys, dict):
                raise ValueError("TENANT_ACCESS_KEYS 必须为 JSON 对象映射")
        except Exception as err:
            raise HTTPException(
                status_code=503,
                detail=f"租户访问配置异常 (TENANT_ACCESS_KEYS JSON解析失败: {err})，聊天接口已安全熔断",
            ) from err

        public_tenants = get_public_chat_tenants()

        # 检查是否在显式公开白名单中
        if requested_tenant_id in public_tenants:
            return requested_tenant_id

        # 默认拒绝：未配置在 access_keys 中且不在公开白名单中
        if requested_tenant_id not in access_keys:
            raise HTTPException(
                status_code=403,
                detail=f"租户 '{requested_tenant_id}' 未授权公开访问且未配置访问凭证，拒绝访问",
            )

        expected_token = str(access_keys[requested_tenant_id]).strip()

        if not x_tenant_token or not x_tenant_token.strip():
            raise HTTPException(
                status_code=401,
                detail="访问受控租户需提供站点访问凭证 (X-Tenant-Token)",
            )

        provided_token = x_tenant_token.strip()
        if secrets.compare_digest(provided_token, expected_token):
            return requested_tenant_id

        # 检查是否为其他租户凭证（越权拦截）
        for other_tid, other_tok in access_keys.items():
            if other_tid != requested_tenant_id and secrets.compare_digest(
                provided_token, str(other_tok).strip()
            ):
                raise HTTPException(
                    status_code=403,
                    detail=f"越权访问：您提供的站点凭证属于租户 '{other_tid}'，无权访问租户 '{requested_tenant_id}' 的客服与知识库",
                )

        raise HTTPException(
            status_code=401,
            detail="无效的租户访问凭证 (X-Tenant-Token)",
        )

    # 3. 未配置 TENANT_ACCESS_KEYS 时：若配置了 PUBLIC_CHAT_TENANTS，限制仅白名单可访问
    public_tenants = get_public_chat_tenants()
    if public_tenants and requested_tenant_id not in public_tenants:
        raise HTTPException(
            status_code=403,
            detail=f"租户 '{requested_tenant_id}' 未列入公开白名单 (PUBLIC_CHAT_TENANTS)，拒绝访问",
        )

    return requested_tenant_id

def is_admin_auth_disabled() -> bool:
    """仅在显式开启免密管理模式时跳过后台密钥校验。"""
    value = os.getenv("ADMIN_AUTH_DISABLED", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}

def verify_admin_authorization(
    tenant_id: str,
    x_admin_key: str | None,
) -> None:
    """验证管理员密钥对指定租户的访问权限（使用 secrets.compare_digest 防时序攻击）。"""
    if is_admin_auth_disabled():
        return

    global_key = os.getenv("ADMIN_API_KEY", "").strip()
    tenant_keys_json = os.getenv("TENANT_ADMIN_KEYS", "").strip()
    tenant_keys: dict[str, str] = {}
    if tenant_keys_json:
        try:
            tenant_keys = json.loads(tenant_keys_json)
            if not isinstance(tenant_keys, dict):
                raise ValueError("TENANT_ADMIN_KEYS 必须为 JSON 对象映射")
        except Exception as error:
            raise HTTPException(
                status_code=503,
                detail=f"管理员密钥配置异常，后台接口已禁用：{error}",
            ) from error

    specific_tenant_key = os.getenv(f"TENANT_KEY_{tenant_id.upper()}", "").strip()
    if not specific_tenant_key and tenant_id in tenant_keys:
        specific_tenant_key = str(tenant_keys[tenant_id]).strip()

    # 如果系统未配置任何密钥，拒绝请求或禁用接口
    if not global_key and not specific_tenant_key and not tenant_keys:
        raise HTTPException(
            status_code=503,
            detail="后台管理功能未配置管理员密钥 (ADMIN_API_KEY)，管理接口已禁用",
        )

    if not x_admin_key or not x_admin_key.strip():
        raise HTTPException(
            status_code=401,
            detail="未提供管理员密钥 (X-Admin-Key)",
        )

    provided_key = x_admin_key.strip()

    # 1. 检查是否匹配该租户专属密钥
    if specific_tenant_key and secrets.compare_digest(provided_key, specific_tenant_key):
        return

    # 2. 检查是否匹配全局 Master 密钥
    if global_key and secrets.compare_digest(provided_key, global_key):
        return

    # 3. 检查提供的 key 是否属于其他租户（检测越权）
    for other_tid, other_k in tenant_keys.items():
        if other_tid != tenant_id and secrets.compare_digest(provided_key, str(other_k).strip()):
            raise HTTPException(
                status_code=403,
                detail=f"越权访问：您提供的密钥仅属于租户 '{other_tid}'，无权操作租户 '{tenant_id}' 的数据",
            )

    raise HTTPException(
        status_code=401,
        detail="无效的 Admin API Key (X-Admin-Key)",
    )
