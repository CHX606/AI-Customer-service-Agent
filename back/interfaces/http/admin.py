"""管理网关：认证、参数校验和用例调用。"""
from fastapi import APIRouter, File, Form, Header, HTTPException, Query, UploadFile
from starlette.concurrency import run_in_threadpool
from back.bootstrap import get_knowledge_service, get_profile_service
from back.application.knowledge import MAX_FILE_BYTES
from back.interfaces.http.auth import is_admin_auth_disabled, verify_admin_authorization
from back.domain.tenant import KnowledgeSource, TenantProfile, TenantProfileUpdate, validate_tenant_id

router = APIRouter(prefix="/admin", tags=["Admin Management"])

# ── 企业资料管理 ──


@router.post(
    "/tenants",
    response_model=TenantProfile,
)
def create_admin_tenant(
    profile: TenantProfile,
    x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
):
    """显式创建新租户（受管理员鉴权保护）。"""
    try:
        safe_tenant_id = validate_tenant_id(profile.tenant_id)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err

    verify_admin_authorization(safe_tenant_id, x_admin_key)
    return get_profile_service().create(profile)


@router.get(
    "/profile",
    response_model=TenantProfile,
)
def get_admin_profile(
    tenant_id: str = Query(default="default"),
    x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
):
    """获取企业完整配置信息，不存在时返回 404。"""
    try:
        safe_tenant_id = validate_tenant_id(tenant_id)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err

    verify_admin_authorization(safe_tenant_id, x_admin_key)
    profile = get_profile_service().get(safe_tenant_id)
    if profile is None:
        raise HTTPException(
            status_code=404,
            detail=f"未找到租户 '{safe_tenant_id}' 的配置资料",
        )
    return profile


@router.put(
    "/profile",
    response_model=TenantProfile,
)
def update_admin_profile(
    update_req: TenantProfileUpdate,
    x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
):
    """更新企业资料配置，若租户不存在返回 404。"""
    try:
        safe_tenant_id = validate_tenant_id(update_req.tenant_id)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err

    verify_admin_authorization(safe_tenant_id, x_admin_key)
    profile = get_profile_service().update(update_req)
    if profile is None:
        raise HTTPException(
            status_code=404,
            detail=f"未找到租户 '{safe_tenant_id}'，无法更新配置",
        )
    return profile




def _authorize(tenant_id, key):
    try:
        tenant_id = validate_tenant_id(tenant_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    verify_admin_authorization(tenant_id, key)
    return tenant_id


@router.post("/knowledge/files", response_model=KnowledgeSource)
async def upload_knowledge_file(file: UploadFile = File(), tenant_id: str = Form(default="default"),
                                x_admin_key: str | None = Header(default=None, alias="X-Admin-Key")):
    tenant = _authorize(tenant_id, x_admin_key)
    content = await file.read(MAX_FILE_BYTES + 1)
    return await run_in_threadpool(get_knowledge_service().upload, tenant, file.filename or "", content)


@router.get("/knowledge/files", response_model=list[KnowledgeSource])
def list_knowledge_files(tenant_id: str = Query(default="default"),
                         x_admin_key: str | None = Header(default=None, alias="X-Admin-Key")):
    return get_knowledge_service().list(_authorize(tenant_id, x_admin_key))


@router.get("/knowledge/files/{source_id}", response_model=KnowledgeSource)
def get_knowledge_file_detail(source_id: str, tenant_id: str = Query(default="default"),
                              x_admin_key: str | None = Header(default=None, alias="X-Admin-Key")):
    return get_knowledge_service().get(_authorize(tenant_id, x_admin_key), source_id)


@router.delete("/knowledge/files/{source_id}")
def delete_knowledge_file(source_id: str, tenant_id: str = Query(default="default"),
                          x_admin_key: str | None = Header(default=None, alias="X-Admin-Key")):
    tenant = _authorize(tenant_id, x_admin_key)
    get_knowledge_service().delete(tenant, source_id)
    return {"success": True, "source_id": source_id, "tenant_id": tenant}


@router.post("/knowledge/files/{source_id}/reindex", response_model=KnowledgeSource)
def reindex_knowledge_file(source_id: str, tenant_id: str = Query(default="default"),
                           x_admin_key: str | None = Header(default=None, alias="X-Admin-Key")):
    return get_knowledge_service().reindex(_authorize(tenant_id, x_admin_key), source_id)
