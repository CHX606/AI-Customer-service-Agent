"""后台免密管理模式测试。"""

import pytest
from fastapi.testclient import TestClient

from back.interfaces.http.admin import is_admin_auth_disabled
from back.interfaces.http.app import app


client = TestClient(app)


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_admin_auth_disabled_truthy_values(monkeypatch, value):
    monkeypatch.setenv("ADMIN_AUTH_DISABLED", value)
    assert is_admin_auth_disabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "unexpected"])
def test_admin_auth_disabled_false_values(monkeypatch, value):
    monkeypatch.setenv("ADMIN_AUTH_DISABLED", value)
    assert is_admin_auth_disabled() is False


def test_profile_can_be_read_without_key_when_disabled(monkeypatch):
    monkeypatch.setenv("ADMIN_AUTH_DISABLED", "1")
    response = client.get("/admin/profile?tenant_id=default")
    assert response.status_code == 200
    assert response.json()["tenant_id"] == "default"


def test_profile_can_be_updated_without_key_when_disabled(monkeypatch):
    monkeypatch.setenv("ADMIN_AUTH_DISABLED", "1")
    response = client.put(
        "/admin/profile",
        json={"tenant_id": "default", "company_name": "免密配置测试企业"},
    )
    assert response.status_code == 200
    assert response.json()["company_name"] == "免密配置测试企业"


def test_knowledge_list_can_be_read_without_key_when_disabled(monkeypatch):
    monkeypatch.setenv("ADMIN_AUTH_DISABLED", "1")
    response = client.get("/admin/knowledge/files?tenant_id=default")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_disabled_mode_ignores_broken_key_configuration(monkeypatch):
    monkeypatch.setenv("ADMIN_AUTH_DISABLED", "1")
    monkeypatch.setenv("TENANT_ADMIN_KEYS", "broken-json{")
    response = client.get("/admin/profile?tenant_id=default")
    assert response.status_code == 200


def test_default_mode_still_rejects_missing_key(monkeypatch):
    monkeypatch.setenv("ADMIN_AUTH_DISABLED", "0")
    response = client.get("/admin/profile?tenant_id=default")
    assert response.status_code == 401
