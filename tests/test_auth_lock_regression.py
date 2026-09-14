"""租户锁定与聊天凭据校验组合的回归测试。"""

import json

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from back.interfaces.http.app import app
from back.interfaces.http.auth import verify_chat_tenant_authorization


@pytest.fixture(autouse=True)
def locked_private_tenant(monkeypatch):
    monkeypatch.setenv("APP_TENANT_ID", "default")
    monkeypatch.setenv("PUBLIC_CHAT_TENANTS", "")
    monkeypatch.setenv(
        "TENANT_ACCESS_KEYS",
        json.dumps({"default": "default-chat-token", "other": "other-chat-token"}),
    )


@pytest.mark.parametrize(
    ("token", "expected_status"),
    [(None, 401), ("", 401), ("   ", 401), ("wrong-token", 401), ("other-chat-token", 403)],
)
def test_locked_tenant_still_rejects_invalid_credentials(token, expected_status):
    with pytest.raises(HTTPException) as error:
        verify_chat_tenant_authorization("default", token)
    assert error.value.status_code == expected_status


def test_locked_tenant_accepts_its_configured_credentials():
    assert verify_chat_tenant_authorization("default", "default-chat-token") == "default"


@pytest.mark.parametrize("public_tenants", ["default", '["default"]'])
def test_locked_public_tenant_remains_accessible_without_credentials(monkeypatch, public_tenants):
    monkeypatch.setenv("PUBLIC_CHAT_TENANTS", public_tenants)
    assert verify_chat_tenant_authorization("default") == "default"


@pytest.mark.parametrize("public_tenants", ["", "default"])
def test_locked_tenant_without_configured_credentials_keeps_public_chat(monkeypatch, public_tenants):
    monkeypatch.delenv("TENANT_ACCESS_KEYS", raising=False)
    monkeypatch.setenv("PUBLIC_CHAT_TENANTS", public_tenants)
    assert verify_chat_tenant_authorization("default") == "default"


@pytest.mark.parametrize("public_tenants", ["", "other"])
def test_tenant_lock_rejects_other_tenants_even_with_valid_credentials(monkeypatch, public_tenants):
    monkeypatch.setenv("PUBLIC_CHAT_TENANTS", public_tenants)
    with pytest.raises(HTTPException) as error:
        verify_chat_tenant_authorization("other", "other-chat-token")
    assert error.value.status_code == 403
    assert "已锁定" in error.value.detail


def test_locked_tenant_rejects_invalid_access_key_configuration(monkeypatch):
    monkeypatch.setenv("TENANT_ACCESS_KEYS", "invalid-json{")
    with pytest.raises(HTTPException) as error:
        verify_chat_tenant_authorization("default")
    assert error.value.status_code == 503


@pytest.mark.parametrize("endpoint", ["/chat", "/chat/stream", "/chat/image"])
def test_all_chat_endpoints_reject_missing_token_before_running_models(endpoint):
    client = TestClient(app)
    if endpoint == "/chat/image":
        response = client.post(
            endpoint,
            data={"tenant_id": "default", "session_id": "locked-session", "message": "你好"},
            files={"image": ("test.png", b"unused-image", "image/png")},
        )
    else:
        response = client.post(
            endpoint,
            json={"tenant_id": "default", "session_id": "locked-session", "message": "你好"},
        )
    assert response.status_code == 401
    assert "X-Tenant-Token" in response.json()["detail"]
