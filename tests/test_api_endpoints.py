from back.bootstrap import get_chat_service, get_knowledge_service
"""测试公开与后台 API 接口、鉴权、越权防护、数据隔离与安全控制。"""

import hashlib
import io
import json
import os
from pathlib import Path
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
import pytest

import back.interfaces.http.app
from back.interfaces.http.app import app
from back.domain.tenant import TenantProfile
from back.tenant.service import get_tenant_profile, save_tenant_profile
from back.infrastructure.persistence.tenants import get_db_connection


client = TestClient(app)

MASTER_KEY = "master-admin-key"
TENANT_A_KEY = "key-tenant-a"
TENANT_B_KEY = "key-tenant-b"
TENANT_A_CHAT_TOKEN = "token-tenant-a"
TENANT_B_CHAT_TOKEN = "token-tenant-b"


def test_public_profile_endpoint():
    # 1. 正常查询已存在的 default 租户 -> 200
    response = client.get("/public/profile?tenant_id=default")
    assert response.status_code == 200
    data = response.json()
    assert data["company_name"] == "可乐云"
    assert data["tenant_id"] == "default"
    assert "welcome_title" in data
    assert "created_at" not in data

    # 2. 查询不存在的未注册租户 -> 404 且绝不自动创建
    conn = get_db_connection()
    try:
        initial_tenant_count = conn.execute("SELECT COUNT(*) FROM tenants").fetchone()[0]
    finally:
        conn.close()

    res_unregistered = client.get("/public/profile?tenant_id=unregistered_tenant")
    assert res_unregistered.status_code == 404
    assert "未找到租户" in res_unregistered.json()["detail"]

    # 断言数据库行数绝对没有增加
    conn = get_db_connection()
    try:
        current_tenant_count = conn.execute("SELECT COUNT(*) FROM tenants").fetchone()[0]
        assert current_tenant_count == initial_tenant_count, "查询未注册租户绝不能在数据库自建数据！"
    finally:
        conn.close()

    # 3. 非法 tenant_id 格式校验 -> 422
    for bad_id in ["../bad", "tenant bad", "tenant@#$", "a" * 65]:
        res_bad = client.get(f"/public/profile?tenant_id={bad_id}")
        assert res_bad.status_code == 422, f"bad_id={bad_id} 预期返回 422，实际返回 {res_bad.status_code}"


def test_chat_endpoints_authorization_and_validation(monkeypatch: pytest.MonkeyPatch):
    def fake_graph_invoke(state):
        return {
            **state,
            "messages": [*state["messages"], AIMessage(content="测试回复")],
        }

    monkeypatch.setattr(get_chat_service().engine.graph, "invoke", fake_graph_invoke)

    # 注册租户 A, B, C
    for tid, cname in [("tenant_a", "A公司"), ("tenant_b", "B公司"), ("tenant_c", "C公司")]:
        save_tenant_profile(
            TenantProfile(
                tenant_id=tid,
                company_name=cname,
                assistant_name=f"{cname}客服",
                short_description=f"{cname}服务",
                welcome_title="你好",
                welcome_description="欢迎",
                tone="专业",
                handoff_message="转人工",
            )
        )

    # 1. 查询不存在的租户 -> 404
    res_not_found = client.post(
        "/chat",
        json={
            "tenant_id": "non_existent_tenant",
            "message": "你好",
            "session_id": "sess_01",
        },
    )
    assert res_not_found.status_code == 404

    # 2. 非法 tenant_id 返回 422
    for bad_id in ["../bad", "tenant bad", "tenant@#$", "a" * 65]:
        res = client.post(
            "/chat",
            json={
                "tenant_id": bad_id,
                "message": "你好",
                "session_id": "sess_01",
            },
        )
        assert res.status_code == 422

    # 3. 受控模式下未提供 X-Tenant-Token -> 401
    res_no_tok = client.post(
        "/chat",
        json={
            "tenant_id": "tenant_a",
            "message": "你好",
            "session_id": "sess_01",
        },
    )
    assert res_no_tok.status_code == 401

    # 4. 拿租户 A 的 Token 访问 租户 B -> 403 Forbidden（跨租户越权拦截）
    res_hack_b = client.post(
        "/chat",
        headers={"X-Tenant-Token": TENANT_A_CHAT_TOKEN},
        json={
            "tenant_id": "tenant_b",
            "message": "你们的退款政策是什么？",
            "session_id": "sess_01",
        },
    )
    assert res_hack_b.status_code == 403
    assert "越权" in res_hack_b.json()["detail"]

    # 5. 租户 C 已在数据库注册，但未在 TENANT_ACCESS_KEYS 映射中 -> 严格默认拒绝 403
    res_unlisted_c = client.post(
        "/chat",
        json={
            "tenant_id": "tenant_c",
            "message": "你好",
            "session_id": "sess_01",
        },
    )
    assert res_unlisted_c.status_code == 403
    assert "未授权公开访问" in res_unlisted_c.json()["detail"]

    # 6. TENANT_ACCESS_KEYS 存在但 JSON 非法 -> 503 熔断保护
    monkeypatch.setenv("TENANT_ACCESS_KEYS", "invalid-json{abc:")
    res_bad_json = client.post(
        "/chat",
        json={
            "tenant_id": "tenant_a",
            "message": "你好",
            "session_id": "sess_01",
        },
    )
    assert res_bad_json.status_code == 503
    assert "熔断" in res_bad_json.json()["detail"]

    # 7. 显式配置 PUBLIC_CHAT_TENANTS 白名单后，白名单租户免 Token 访问 -> 200 OK
    monkeypatch.setenv("TENANT_ACCESS_KEYS", json.dumps({"tenant_a": "token-tenant-a"}))
    monkeypatch.setenv("PUBLIC_CHAT_TENANTS", "tenant_c,default")
    res_public_c = client.post(
        "/chat",
        json={
            "tenant_id": "tenant_c",
            "message": "你好",
            "session_id": "sess_01",
        },
    )
    assert res_public_c.status_code == 200

    # 8. 拿租户 A 的 Token 正常访问 租户 A -> 200 OK
    res_ok_a = client.post(
        "/chat",
        headers={"X-Tenant-Token": TENANT_A_CHAT_TOKEN},
        json={
            "tenant_id": "tenant_a",
            "message": "你好",
            "session_id": "sess_01",
        },
    )
    assert res_ok_a.status_code == 200

    # 9. 单租户锁定模式测试 (APP_TENANT_ID=default)
    monkeypatch.delenv("PUBLIC_CHAT_TENANTS", raising=False)
    monkeypatch.setenv("APP_TENANT_ID", "default")
    res_locked_tamper = client.post(
        "/chat",
        json={
            "tenant_id": "tenant_a",
            "message": "你好",
            "session_id": "sess_01",
        },
    )
    assert res_locked_tamper.status_code == 403
    assert "已锁定" in res_locked_tamper.json()["detail"]


def test_admin_auth_and_unconfigured(monkeypatch: pytest.MonkeyPatch):
    # 密钥未配置时 -> 503
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    monkeypatch.delenv("TENANT_ADMIN_KEYS", raising=False)
    res_no_sys_key = client.get(
        "/admin/profile?tenant_id=default",
        headers={"X-Admin-Key": "some-key"},
    )
    assert res_no_sys_key.status_code == 503
    assert "未配置" in res_no_sys_key.json()["detail"]

    # 配置存在但格式损坏时必须失败关闭，不能静默忽略后放行。
    monkeypatch.setenv("TENANT_ADMIN_KEYS", "not-valid-json{")
    res_bad_config = client.get(
        "/admin/profile?tenant_id=default",
        headers={"X-Admin-Key": "some-key"},
    )
    assert res_bad_config.status_code == 503
    assert "配置异常" in res_bad_config.json()["detail"]


def test_admin_create_and_cross_tenant_unauthorized_access():
    # 显式使用创建租户接口 POST /admin/tenants
    new_tenant_a = TenantProfile(
        tenant_id="tenant_a",
        company_name="A公司",
        assistant_name="A客服",
        short_description="A公司服务",
        welcome_title="你好",
        welcome_description="欢迎",
        tone="专业",
        handoff_message="转人工",
    )
    res_create_a = client.post(
        "/admin/tenants",
        headers={"X-Admin-Key": TENANT_A_KEY},
        json=new_tenant_a.model_dump(mode="json"),
    )
    assert res_create_a.status_code == 200

    new_tenant_b = TenantProfile(
        tenant_id="tenant_b",
        company_name="B公司",
        assistant_name="B客服",
        short_description="B公司服务",
        welcome_title="你好",
        welcome_description="欢迎",
        tone="专业",
        handoff_message="转人工",
    )
    res_create_b = client.post(
        "/admin/tenants",
        headers={"X-Admin-Key": TENANT_B_KEY},
        json=new_tenant_b.model_dump(mode="json"),
    )
    assert res_create_b.status_code == 200

    # 1. 租户 A 的 Key 访问 租户 A -> 200 OK
    res_a_ok = client.get(
        "/admin/profile?tenant_id=tenant_a",
        headers={"X-Admin-Key": TENANT_A_KEY},
    )
    assert res_a_ok.status_code == 200
    assert res_a_ok.json()["company_name"] == "A公司"

    # 2. 租户 A 的 Key 尝试访问 租户 B 的资料 -> 403 Forbidden（越权拦截）
    res_a_hack_b = client.get(
        "/admin/profile?tenant_id=tenant_b",
        headers={"X-Admin-Key": TENANT_A_KEY},
    )
    assert res_a_hack_b.status_code == 403
    assert "越权" in res_a_hack_b.json()["detail"]

    # 3. 租户 A 的 Key 尝试修改 租户 B 的资料 -> 403 Forbidden（越权拦截）
    res_a_hack_update_b = client.put(
        "/admin/profile",
        headers={"X-Admin-Key": TENANT_A_KEY},
        json={
            "tenant_id": "tenant_b",
            "company_name": "被篡改的公司",
        },
    )
    assert res_a_hack_update_b.status_code == 403

    # 完整资料中的必填字段不能被 PATCH 语义请求显式清空为 null。
    res_null_required = client.put(
        "/admin/profile",
        headers={"X-Admin-Key": TENANT_A_KEY},
        json={"tenant_id": "tenant_a", "company_name": None},
    )
    assert res_null_required.status_code == 422

    # 4. 全局 Master Key 访问 租户 A 和 租户 B -> 均 200 OK
    res_master_a = client.get(
        "/admin/profile?tenant_id=tenant_a",
        headers={"X-Admin-Key": MASTER_KEY},
    )
    assert res_master_a.status_code == 200

    res_master_b = client.get(
        "/admin/profile?tenant_id=tenant_b",
        headers={"X-Admin-Key": MASTER_KEY},
    )
    assert res_master_b.status_code == 200


def test_knowledge_file_security_and_lifecycle(monkeypatch):
    # API 生命周期测试不依赖外部 OpenSearch；检索存储由专项测试覆盖。
    monkeypatch.setattr(
        "back.knowledge.indexing.service.add_source_documents",
        lambda tenant_id, source_id, documents: len(documents),
    )
    monkeypatch.setattr(
        "back.infrastructure.search.opensearch.delete_source_documents",
        lambda tenant_id, source_id: 1,
    )
    save_tenant_profile(
        TenantProfile(
            tenant_id="tenant_a",
            company_name="A公司",
            assistant_name="A客服",
            short_description="A公司服务",
            welcome_title="你好",
            welcome_description="欢迎",
            tone="专业",
            handoff_message="转人工",
        )
    )

    # 1. 路径穿越上传尝试 -> 400
    res_traversal = client.post(
        "/admin/knowledge/files",
        headers={"X-Admin-Key": TENANT_A_KEY},
        data={"tenant_id": "tenant_a"},
        files={"file": ("../../malicious.txt", b"evil content", "text/plain")},
    )
    assert res_traversal.status_code == 400
    assert "路径穿越" in res_traversal.json()["detail"]

    # 2. 不支持的文件格式 -> 400
    res_bad_ext = client.post(
        "/admin/knowledge/files",
        headers={"X-Admin-Key": TENANT_A_KEY},
        data={"tenant_id": "tenant_a"},
        files={"file": ("malicious.exe", b"binary content", "application/octet-stream")},
    )
    assert res_bad_ext.status_code == 400
    assert "不支持" in res_bad_ext.json()["detail"]

    # 合法扩展名但损坏的文档应明确返回 422，并保留 failed 状态用于排查。
    res_corrupt_pdf = client.post(
        "/admin/knowledge/files",
        headers={"X-Admin-Key": TENANT_A_KEY},
        data={"tenant_id": "tenant_a"},
        files={"file": ("broken.pdf", b"not-a-real-pdf", "application/pdf")},
    )
    assert res_corrupt_pdf.status_code == 422
    assert "解析或索引失败" in res_corrupt_pdf.json()["detail"]

    # 3. 正常上传合法 TXT 文档
    test_content = (
        "VIP退款政策\n\n"
        "VIP用户购买后享有30天极速退款保障。"
    ).encode("utf-8")

    res_upload = client.post(
        "/admin/knowledge/files",
        headers={"X-Admin-Key": TENANT_A_KEY},
        data={"tenant_id": "tenant_a"},
        files={"file": ("vip_refund.txt", test_content, "text/plain")},
    )
    assert res_upload.status_code == 200
    source_data = res_upload.json()
    source_id = source_data["source_id"]
    assert source_data["status"] == "ready"
    assert source_data["chunk_count"] >= 1

    # 4. 租户 B 的 Key 试图查询或删除 租户 A 的文件 -> 403 Forbidden
    res_b_hack_list = client.get(
        "/admin/knowledge/files?tenant_id=tenant_a",
        headers={"X-Admin-Key": TENANT_B_KEY},
    )
    assert res_b_hack_list.status_code == 403

    res_b_hack_del = client.delete(
        f"/admin/knowledge/files/{source_id}?tenant_id=tenant_a",
        headers={"X-Admin-Key": TENANT_B_KEY},
    )
    assert res_b_hack_del.status_code == 403

    # 5. 租户 A 的 Key 正常查询与删除
    res_list = client.get(
        "/admin/knowledge/files?tenant_id=tenant_a",
        headers={"X-Admin-Key": TENANT_A_KEY},
    )
    assert res_list.status_code == 200
    assert any(item["source_id"] == source_id for item in res_list.json())

    res_del = client.delete(
        f"/admin/knowledge/files/{source_id}?tenant_id=tenant_a",
        headers={"X-Admin-Key": TENANT_A_KEY},
    )
    assert res_del.status_code == 200
    assert res_del.json()["success"] is True
