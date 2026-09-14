"""租户与企业配置业务逻辑层。"""

import hashlib
import shutil
from datetime import datetime
from pathlib import Path

from back.core.paths import DATA_DIR as DEFAULT_DATA_DIR
from back.core.paths import KNOWLEDGE_RESOURCES_DIR
from back.tenant.defaults import DEFAULT_KELECLOUD_PROFILE
from back.domain.tenant import (
    KnowledgeSource,
    KnowledgeSourceStatus,
    TenantProfile,
    TenantProfileUpdate,
    TenantPublicProfile,
    validate_tenant_id,
)
from back.infrastructure.persistence.tenants import (
    delete_knowledge_source_record,
    init_db,
    list_knowledge_sources,
    load_knowledge_source,
    load_tenant_profile,
    save_knowledge_source,
    save_tenant_profile,
)


DATA_DIR = DEFAULT_DATA_DIR
LEGACY_DOCX_PATH = KNOWLEDGE_RESOURCES_DIR / "可乐云客服操作文档.docx"
EXTERNAL_APPS_GUIDE_PATH = KNOWLEDGE_RESOURCES_DIR / "外网常用应用使用与排障.md"
EXTERNAL_APPS_SOURCE_ID = "builtin_external_apps_guide"


def get_tenant_upload_dir(tenant_id: str, source_id: str) -> Path:
    """获取指定租户和数据源的文件上传保存目录。"""
    safe_tenant_id = validate_tenant_id(tenant_id)
    upload_dir = DATA_DIR / "tenants" / safe_tenant_id / "uploads" / source_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def init_tenant_system() -> None:
    """初始化租户系统数据库与 default 租户数据。"""
    init_db()

    default_profile = load_tenant_profile("default")
    if default_profile is None:
        save_tenant_profile(DEFAULT_KELECLOUD_PROFILE)

    # 注册旧的可乐云操作文档为 legacy_kelecloud_docx
    legacy_source = load_knowledge_source("legacy_kelecloud_docx")
    if legacy_source is None and LEGACY_DOCX_PATH.exists():
        content_bytes = LEGACY_DOCX_PATH.read_bytes()
        content_hash = hashlib.sha256(content_bytes).hexdigest()
        duplicate_source = next(
            (
                source
                for source in list_knowledge_sources("default")
                if source.content_hash == content_hash
                and source.status in {"processing", "ready"}
            ),
            None,
        )
        if duplicate_source is not None:
            legacy_source = duplicate_source

    if (
        legacy_source is None
        and LEGACY_DOCX_PATH.exists()
    ):
        content_bytes = LEGACY_DOCX_PATH.read_bytes()
        content_hash = hashlib.sha256(content_bytes).hexdigest()
        registered_source = KnowledgeSource(
            source_id="legacy_kelecloud_docx",
            tenant_id="default",
            original_filename="可乐云客服操作文档.docx",
            stored_filename="可乐云客服操作文档.docx",
            file_type="docx",
            content_hash=content_hash,
            status="ready",
            error_message=None,
            # 旧数据源尚未通过当前切块器重索，不能伪造固定分片数。
            # 首次显式重索后由索引服务写入真实数量。
            chunk_count=0,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        save_knowledge_source(registered_source)

    # 内置外网应用指南按独立数据源登记，便于单独更新和重建索引。
    if EXTERNAL_APPS_GUIDE_PATH.exists():
        content_bytes = EXTERNAL_APPS_GUIDE_PATH.read_bytes()
        content_hash = hashlib.sha256(content_bytes).hexdigest()
        external_source = load_knowledge_source(EXTERNAL_APPS_SOURCE_ID)
        if external_source is None or external_source.content_hash != content_hash:
            upload_dir = get_tenant_upload_dir(
                "default", EXTERNAL_APPS_SOURCE_ID
            )
            stored_path = upload_dir / EXTERNAL_APPS_GUIDE_PATH.name
            stored_path.write_bytes(content_bytes)
            registered_source = KnowledgeSource(
                source_id=EXTERNAL_APPS_SOURCE_ID,
                tenant_id="default",
                original_filename=EXTERNAL_APPS_GUIDE_PATH.name,
                stored_filename=EXTERNAL_APPS_GUIDE_PATH.name,
                file_type="md",
                content_hash=content_hash,
                status="ready",
                error_message=None,
                chunk_count=0,
                created_at=(
                    external_source.created_at
                    if external_source is not None
                    else datetime.now()
                ),
                updated_at=datetime.now(),
            )
            save_knowledge_source(registered_source)


def get_tenant_profile(tenant_id: str = "default") -> TenantProfile | None:
    """查询租户完整配置信息，不存在时严格返回 None，禁止自动创建。"""
    safe_id = validate_tenant_id(tenant_id)
    return load_tenant_profile(safe_id)


def create_tenant_profile(profile: TenantProfile) -> TenantProfile:
    """显式创建新租户（仅允许由管理员鉴权接口调用）。"""
    save_tenant_profile(profile, create_only=True)
    return profile


def get_tenant_public_profile(tenant_id: str = "default") -> TenantPublicProfile | None:
    """获取租户对公展示资料，不存在时返回 None。"""
    profile = get_tenant_profile(tenant_id)
    if profile is None:
        return None

    return TenantPublicProfile(
        tenant_id=profile.tenant_id,
        company_name=profile.company_name,
        brand_name_en=profile.brand_name_en,
        assistant_name=profile.assistant_name,
        short_description=profile.short_description,
        business_scope=profile.business_scope,
        business_hours=profile.business_hours,
        public_contact=profile.public_contact,
        welcome_title=profile.welcome_title,
        welcome_description=profile.welcome_description,
        tone=profile.tone,
        handoff_message=profile.handoff_message,
        suggested_questions=profile.suggested_questions,
        updated_at=profile.updated_at,
    )


def update_tenant_profile(update_req: TenantProfileUpdate) -> TenantProfile | None:
    """更新租户配置信息，若租户不存在返回 None。"""
    current_profile = get_tenant_profile(update_req.tenant_id)
    if current_profile is None:
        return None

    update_data = update_req.model_dump(exclude_unset=True)
    update_data["updated_at"] = datetime.now()

    updated_dict = current_profile.model_dump()
    updated_dict.update(update_data)

    new_profile = TenantProfile(**updated_dict)
    save_tenant_profile(new_profile)
    return new_profile


def register_knowledge_source(
    tenant_id: str,
    source_id: str,
    original_filename: str,
    stored_filename: str,
    file_type: str,
    content_hash: str,
    status: KnowledgeSourceStatus = "processing",
) -> KnowledgeSource:
    """登记新知识库数据源。"""
    safe_tenant_id = validate_tenant_id(tenant_id)
    source = KnowledgeSource(
        source_id=source_id,
        tenant_id=safe_tenant_id,
        original_filename=original_filename,
        stored_filename=stored_filename,
        file_type=file_type.lower(),
        content_hash=content_hash,
        status=status,
        chunk_count=0,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    save_knowledge_source(source)
    return source


def update_knowledge_source_status(
    source_id: str,
    status: KnowledgeSourceStatus,
    chunk_count: int | None = None,
    error_message: str | None = None,
) -> KnowledgeSource | None:
    """更新数据源处理状态与分片数。"""
    source = load_knowledge_source(source_id)
    if source is None:
        return None

    updated_dict = source.model_dump()
    updated_dict["status"] = status
    updated_dict["updated_at"] = datetime.now()
    if chunk_count is not None:
        updated_dict["chunk_count"] = chunk_count
    if error_message is not None:
        updated_dict["error_message"] = error_message
    elif status == "ready":
        updated_dict["error_message"] = None

    updated_source = KnowledgeSource(**updated_dict)
    save_knowledge_source(updated_source)
    return updated_source


def get_knowledge_source(source_id: str) -> KnowledgeSource | None:
    """获取单个知识库文件记录。"""
    return load_knowledge_source(source_id)


def get_all_knowledge_sources(tenant_id: str = "default") -> list[KnowledgeSource]:
    """获取租户名下的所有知识库文件。"""
    safe_tenant_id = validate_tenant_id(tenant_id)
    return list_knowledge_sources(safe_tenant_id)


def delete_knowledge_source(source_id: str, tenant_id: str) -> bool:
    """删除知识库文件磁盘文件与数据库记录。"""
    safe_tenant_id = validate_tenant_id(tenant_id)
    source = load_knowledge_source(source_id)
    if source is None or source.tenant_id != safe_tenant_id:
        return False

    # 删除磁盘文件
    source_dir = DATA_DIR / "tenants" / safe_tenant_id / "uploads" / source_id
    if source_dir.exists():
        shutil.rmtree(source_dir, ignore_errors=True)

    return delete_knowledge_source_record(source_id)
