"""测试租户存储、校验、外键约束与级联删除。"""

from datetime import datetime
from pathlib import Path
import pytest

from back.tenant import service as tenant_service
from back.domain.tenant import (
    KnowledgeSource,
    TenantProfile,
    validate_tenant_id,
)
from back.infrastructure.persistence.tenants import (
    delete_knowledge_source_record,
    delete_tenant_profile,
    get_db_connection,
    init_db,
    list_knowledge_sources,
    load_knowledge_source,
    load_tenant_profile,
    save_knowledge_source,
    save_tenant_profile,
)


def test_default_tenant_registers_external_apps_knowledge_source():
    profile = tenant_service.get_tenant_profile("default")
    source = tenant_service.get_knowledge_source(
        tenant_service.EXTERNAL_APPS_SOURCE_ID
    )

    assert profile is not None
    assert any("外网应用" in item for item in profile.business_scope)
    assert source is not None
    assert source.status == "ready"
    stored_path = (
        tenant_service.get_tenant_upload_dir("default", source.source_id)
        / source.stored_filename
    )
    assert stored_path.exists()


def test_tenant_id_validation():
    # 合法 ID
    assert validate_tenant_id("default") == "default"
    assert validate_tenant_id("tenant_123") == "tenant_123"
    assert validate_tenant_id("company-a") == "company-a"
    assert validate_tenant_id("a" * 64) == "a" * 64

    # 非法 ID 抛出 ValueError
    invalid_cases = [
        "",
        "   ",
        "tenant/../bad",
        "../bad",
        "tenant bad",
        "tenant@#$",
        "tenant.bad",
        "a" * 65,
    ]
    for invalid in invalid_cases:
        with pytest.raises(ValueError):
            validate_tenant_id(invalid)


def test_sqlite_foreign_keys_and_cascade(tmp_path: Path):
    db_path = tmp_path / "fk_test.db"
    init_db(db_path)

    # 1. 验证外键约束处于开启状态
    conn = get_db_connection(db_path)
    try:
        fk_status = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
        assert fk_status == 1, "SQLite PRAGMA foreign_keys 必须启用"
    finally:
        conn.close()

    # 2. 插入租户与关联知识库数据源
    tenant = TenantProfile(
        tenant_id="cascade_tenant",
        company_name="级联测试公司",
        brand_name_en="CascadeCo",
        assistant_name="级联客服",
        short_description="测试外键级联",
        business_scope=[],
        business_hours=None,
        public_contact=None,
        welcome_title="你好",
        welcome_description="欢迎",
        tone="简洁",
        handoff_message="转人工",
        suggested_questions=[],
    )
    save_tenant_profile(tenant, db_path=db_path)

    source = KnowledgeSource(
        source_id="cascade_src_001",
        tenant_id="cascade_tenant",
        original_filename="doc.docx",
        stored_filename="doc.docx",
        file_type="docx",
        content_hash="hash123",
        status="ready",
        chunk_count=5,
    )
    save_knowledge_source(source, db_path=db_path)

    assert load_tenant_profile("cascade_tenant", db_path=db_path) is not None
    assert load_knowledge_source("cascade_src_001", db_path=db_path) is not None

    # 3. 删除租户，验证关联的数据源被级联删除
    deleted = delete_tenant_profile("cascade_tenant", db_path=db_path)
    assert deleted is True
    assert load_tenant_profile("cascade_tenant", db_path=db_path) is None
    assert load_knowledge_source("cascade_src_001", db_path=db_path) is None


def test_tenant_profile_store(tmp_path: Path):
    db_path = tmp_path / "profile_test.db"
    init_db(db_path)

    profile = TenantProfile(
        tenant_id="test_company",
        company_name="测试科技有限公司",
        brand_name_en="TestTech",
        assistant_name="小测助理",
        short_description="提供全方位智能化软件测试服务",
        business_scope=["自动化测试", "性能压测", "接口联调"],
        business_hours=None,
        public_contact=None,
        welcome_title="你好，我是小测助理",
        welcome_description="欢迎咨询测试相关问题。",
        tone="热情、严谨",
        handoff_message="请稍候，正在为您转接人工工程师。",
        suggested_questions=["如何发起性能压测？", "测试报告多久出具？"],
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    save_tenant_profile(profile, db_path=db_path)

    loaded = load_tenant_profile("test_company", db_path=db_path)
    assert loaded is not None
    assert loaded.company_name == "测试科技有限公司"
    assert loaded.brand_name_en == "TestTech"
    assert loaded.business_scope == ["自动化测试", "性能压测", "接口联调"]
    assert loaded.suggested_questions == ["如何发起性能压测？", "测试报告多久出具？"]


def test_knowledge_source_store(tmp_path: Path):
    db_path = tmp_path / "ks_test.db"
    init_db(db_path)

    # 必须先有租户
    tenant = TenantProfile(
        tenant_id="test_company",
        company_name="测试公司",
        brand_name_en=None,
        assistant_name="助理",
        short_description="测试",
        business_scope=[],
        welcome_title="你好",
        welcome_description="欢迎",
        tone="简洁",
        handoff_message="转人工",
        suggested_questions=[],
    )
    save_tenant_profile(tenant, db_path=db_path)

    source = KnowledgeSource(
        source_id="src_001",
        tenant_id="test_company",
        original_filename="测试文档.docx",
        stored_filename="测试文档_001.docx",
        file_type="docx",
        content_hash="abc123hash",
        status="ready",
        error_message=None,
        chunk_count=10,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    save_knowledge_source(source, db_path=db_path)

    loaded = load_knowledge_source("src_001", db_path=db_path)
    assert loaded is not None
    assert loaded.original_filename == "测试文档.docx"
    assert loaded.chunk_count == 10
    assert loaded.status == "ready"

    sources = list_knowledge_sources("test_company", db_path=db_path)
    assert len(sources) == 1

    deleted = delete_knowledge_source_record("src_001", db_path=db_path)
    assert deleted is True
    assert load_knowledge_source("src_001", db_path=db_path) is None
