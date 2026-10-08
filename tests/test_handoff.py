"""人工申请持久化、租户授权、幂等提交与 SMTP 失败边界。"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Barrier, Event
from unittest.mock import MagicMock
import smtplib

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
import pytest

from back.application.handoff import HandoffService
from back.domain.errors import Conflict, NotFound
from back.infrastructure.notifications import EmailNotifier
from back.infrastructure.persistence.handoffs import SQLiteHandoffRepository, init_handoff_db
from back.infrastructure.persistence.sessions import SQLiteSessionRepository
from back.infrastructure.services import SQLiteTenantRepository
from back.interfaces.http.app import create_app
from back.interfaces.http.handoff import get_handoff_service, router
from back.tenant.service import get_tenant_profile, save_tenant_profile


class FakeNotifier:
    def __init__(self, available=True):
        self.available = available
        self.calls = []
        self.error = None

    def availability(self, tenant_id):
        return self.available, None if self.available else "邮件通知尚未配置。"

    def send(self, ticket):
        self.calls.append(ticket.ticket_id)
        if self.error:
            raise self.error


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("PUBLIC_CHAT_TENANTS", "default")
    for variable in ("SMTP_HOST", "SMTP_PORT", "SMTP_USERNAME", "SMTP_PASSWORD", "SMTP_FROM",
                     "SMTP_SECURITY", "SMTP_TIMEOUT_SECONDS", "HANDOFF_NOTIFICATION_TO",
                     "HANDOFF_NOTIFICATION_RECIPIENTS", "HANDOFF_ADMIN_URL"):
        monkeypatch.delenv(variable, raising=False)
    default = get_tenant_profile("default")
    for tenant_id in ("tenant_a", "tenant_b"):
        save_tenant_profile(default.model_copy(update={"tenant_id": tenant_id}))
    init_handoff_db()
    clock = [datetime(2026, 9, 30, tzinfo=timezone.utc)]
    notifier = FakeNotifier()
    sessions = SQLiteSessionRepository(tmp_path / "handoff_sessions.db")
    sessions.load(("default", "chat"))
    result = HandoffService(SQLiteHandoffRepository(), notifier, sessions,
                           SQLiteTenantRepository(), now=lambda: clock[0])
    result.test_clock = clock
    return result


@pytest.fixture
def client(service):
    app = create_app()
    if not any(getattr(route, "path", None) == "/handoff" for route in app.routes):
        app.include_router(router)
    app.dependency_overrides[get_handoff_service] = lambda: service
    return TestClient(app)


def submit(service, key="same-submission", tenant="default", issue="支付后订单未到账"):
    return service.submit(tenant, "chat", key, issue)


def test_tickets_survive_restart_with_server_history_and_isolate_tenants(service):
    service.sessions.save(("default", "chat"), {
        "messages": [HumanMessage(content="此前的订单问题"), AIMessage(content="请提供订单号")],
    }, 0)
    ticket = submit(service)
    restarted = SQLiteHandoffRepository()
    saved = restarted.get("default", ticket.ticket_id)
    assert saved.issue == "支付后订单未到账"
    assert saved.messages == [{"role": "user", "content": "此前的订单问题"},
                              {"role": "assistant", "content": "请提供订单号"}]
    assert restarted.get("tenant_a", ticket.ticket_id) is None
    with pytest.raises(NotFound):
        service.get("tenant_a", ticket.ticket_id)
    assert restarted.update("tenant_a", ticket.ticket_id, "resolved", "越权", saved.updated_at) is None
    assert restarted.get("default", ticket.ticket_id).status == "pending"


def test_submission_retry_is_idempotent_and_cannot_change_original_request(service):
    first = submit(service)
    assert submit(service).ticket_id == first.ticket_id
    with pytest.raises(Conflict):
        submit(service, issue="另一个不同的问题")
    with pytest.raises(Conflict):
        service.submit("default", "other-session", "same-submission", first.issue)
    assert submit(service, tenant="tenant_a").ticket_id != first.ticket_id
    assert len(service.list("default")) == 1


def test_external_channel_ticket_keeps_given_messages_and_is_idempotent(service):
    messages = [
        {"role": "user", "content": "付了钱没到账"},
        {"role": "assistant", "content": "原客服AI：请发付款截图"},
        {"role": "system", "content": "不应保存"},
        {"role": "user", "content": "   "},
    ]
    ticket, created = service.submit_external("default", "crisp:session_abc", "crisp-relay:session_abc:2026-10-01",
                                              "【Crisp 接力上报】需要人工办理：补发套餐", messages)
    assert created is True
    assert ticket.messages == messages[:2]
    assert ticket.notification_status == "pending"

    again, created_again = service.submit_external("default", "crisp:session_abc", "crisp-relay:session_abc:2026-10-01",
                                                   "【Crisp 接力上报】需要人工办理：补发套餐", messages)
    assert created_again is False and again.ticket_id == ticket.ticket_id
    with pytest.raises(Conflict):
        service.submit_external("default", "crisp:session_abc", "crisp-relay:session_abc:2026-10-01",
                                "【Crisp 接力上报】另一个原因", messages)
    assert service.dispatch_pending() == 1
    assert service.notifier.calls == [ticket.ticket_id]


def test_concurrent_submissions_create_one_ticket(service):
    barrier = Barrier(2)

    def create():
        barrier.wait(timeout=5)
        return submit(service).ticket_id

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [future.result(timeout=10) for future in [pool.submit(create), pool.submit(create)]]
    assert results[0] == results[1]
    assert len(service.list("default")) == 1


def test_concurrent_dispatch_and_successful_retry_send_only_one_email(service):
    ticket = submit(service)
    entered, release = Event(), Event()
    notifier = service.notifier
    original = notifier.send

    def send_blocked(value):
        original(value)
        entered.set()
        assert release.wait(timeout=5)

    notifier.send = send_blocked
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(service.dispatch, ticket.ticket_id, "default")
        assert entered.wait(timeout=5)
        second = pool.submit(service.dispatch, ticket.ticket_id, "default")
        assert second.result(timeout=5) is False
        release.set()
        assert first.result(timeout=5) is True
    assert service.retry_notification("default", ticket.ticket_id).notification_status == "sent"
    assert service.dispatch(ticket.ticket_id, "default") is False
    assert notifier.calls == [ticket.ticket_id]


def test_unconfigured_notification_keeps_ticket_and_recovers_after_configuration(service):
    service.notifier.available = False
    ticket = submit(service)
    assert ticket.notification_status == "unconfigured"
    assert service.dispatch(ticket.ticket_id, "default") is False
    assert service.get("default", ticket.ticket_id).status == "pending"
    assert service.notifier.calls == []
    service.notifier.available = True
    service.test_clock[0] += timedelta(seconds=61)
    assert service.dispatch_pending() == 1
    assert service.get("default", ticket.ticket_id).notification_status == "sent"


def test_notification_failures_back_off_stop_and_allow_manual_retry(service, caplog):
    service.notifier.error = smtplib.SMTPAuthenticationError(535, b"secret-password private-body")
    ticket = submit(service)
    assert service.dispatch(ticket.ticket_id, "default") is False
    assert service.dispatch_pending() == 0
    for _ in range(2):
        service.test_clock[0] += timedelta(minutes=2)
        assert service.dispatch_pending() == 0
    saved = service.get("default", ticket.ticket_id)
    assert saved.notification_status == "failed"
    assert saved.notification_attempts == 3
    assert "SMTPAuthenticationError" in saved.notification_error
    assert "secret-password" not in saved.notification_error + caplog.text
    assert "private-body" not in saved.notification_error + caplog.text
    service.test_clock[0] += timedelta(hours=1)
    assert service.dispatch_pending() == 0
    assert len(service.notifier.calls) == 3
    service.notifier.error = None
    service.retry_notification("default", ticket.ticket_id)
    assert service.dispatch(ticket.ticket_id, "default") is True
    assert service.get("default", ticket.ticket_id).notification_attempts == 1


def test_interrupted_send_is_marked_unknown_without_automatic_duplicate(service):
    ticket = submit(service)
    assert service.repository.claim("default", ticket.ticket_id, "crashed-worker", ticket.created_at, 3)
    with pytest.raises(Conflict):
        service.retry_notification("default", ticket.ticket_id)
    service.test_clock[0] += timedelta(minutes=6)
    assert service.dispatch_pending() == 0
    saved = service.get("default", ticket.ticket_id)
    assert saved.notification_status == "failed"
    assert "状态未知" in saved.notification_error
    assert service.notifier.calls == []
    service.retry_notification("default", ticket.ticket_id)
    assert service.dispatch(ticket.ticket_id, "default") is True


def test_pending_outbox_is_sent_by_restarted_service(service):
    ticket = submit(service)
    restarted = HandoffService(SQLiteHandoffRepository(), service.notifier, service.sessions,
                               SQLiteTenantRepository(), now=service.now)
    assert restarted.dispatch_pending() == 1
    assert restarted.dispatch_pending() == 0
    assert restarted.get("default", ticket.ticket_id).sent_at is not None


def test_public_receipt_is_minimal_and_duplicate_post_does_not_email_twice(client, service):
    body = {"session_id": "chat", "submission_id": "api-submit", "issue": "订单未到账"}
    response = client.post("/handoff", json=body)
    assert response.status_code == 200
    assert set(response.json()) == {"ticket_id", "status", "notification_status", "message"}
    assert response.json()["notification_status"] == "pending"
    assert "messages" not in response.text and "@" not in response.text
    duplicate = client.post("/handoff", json=body)
    assert duplicate.json()["ticket_id"] == response.json()["ticket_id"]
    assert duplicate.json()["notification_status"] == "pending"
    assert service.notifier.calls == []
    assert service.dispatch_pending() == 1
    completed = client.post("/handoff", json=body)
    assert completed.json()["notification_status"] == "sent"
    assert service.dispatch_pending() == 0
    assert len(service.notifier.calls) == 1
    forbidden = client.post("/handoff", json={**body, "tenant_id": "tenant_a"})
    assert forbidden.status_code == 401
    forbidden = client.post("/handoff", json={**body, "tenant_id": "tenant_a"},
                            headers={"X-Tenant-Token": "token-tenant-b"})
    assert forbidden.status_code == 403
    assert service.list("tenant_a") == []


@pytest.mark.parametrize("change", [{"issue": "   "}, {"issue": "x" * 2001},
                                    {"tenant_id": "../other"}, {"session_id": " "},
                                    {"submission_id": " "}])
def test_handoff_rejects_invalid_requests(client, service, change):
    response = client.post("/handoff", json={"session_id": "chat", "submission_id": "key",
                                            "issue": "问题", **change})
    assert response.status_code == 422
    assert service.list("default") == []


def test_admin_authorization_tenant_scope_status_filter_and_processing(client, service):
    first = submit(service, "first")
    second = submit(service, "second")
    headers = {"X-Admin-Key": "default-admin-key"}
    assert client.get("/admin/handoffs").status_code == 401
    assert client.get("/admin/handoffs", headers={"X-Admin-Key": "key-tenant-a"}).status_code == 403
    updated = client.put(f"/admin/handoffs/{first.ticket_id}", headers=headers,
                         json={"status": "processing", "handling_note": "检查订单记录"})
    assert updated.status_code == 200
    assert updated.json()["handling_note"] == "检查订单记录"
    filtered = client.get("/admin/handoffs?status=processing", headers=headers)
    assert [entry["ticket_id"] for entry in filtered.json()] == [first.ticket_id]
    assert len(client.get("/admin/handoffs?limit=1", headers=headers).json()) == 1
    assert client.get("/admin/handoffs?limit=101", headers=headers).status_code == 422
    cross = client.get(f"/admin/handoffs/{first.ticket_id}?tenant_id=tenant_a",
                       headers={"X-Admin-Key": "key-tenant-a"})
    assert cross.status_code == 404
    cross = client.put(f"/admin/handoffs/{first.ticket_id}",
                       headers={"X-Admin-Key": "key-tenant-a"},
                       json={"tenant_id": "tenant_a", "status": "resolved"})
    assert cross.status_code == 404
    assert service.get("default", first.ticket_id).status == "processing"
    assert service.get("default", second.ticket_id).status == "pending"


def test_admin_can_retry_failed_notification_without_resending_sent(client, service):
    ticket = submit(service)
    service.notifier.error = RuntimeError("private")
    service.dispatch(ticket.ticket_id, "default")
    service.notifier.error = None
    path = f"/admin/handoffs/{ticket.ticket_id}/notification/retry"
    assert client.post(path).status_code == 401
    response = client.post(path, headers={"X-Admin-Key": "default-admin-key"})
    assert response.status_code == 200
    assert response.json()["notification_status"] == "pending"
    assert service.get("default", ticket.ticket_id).notification_status == "pending"
    assert len(service.notifier.calls) == 1
    assert service.dispatch_pending() == 1
    assert service.get("default", ticket.ticket_id).notification_status == "sent"
    client.post(path, headers={"X-Admin-Key": "default-admin-key"})
    assert service.dispatch_pending() == 0
    assert len(service.notifier.calls) == 2


def configure_smtp(monkeypatch, security="starttls", host="smtp.example.com"):
    for key, value in {"SMTP_HOST": host, "SMTP_PORT": "465" if security == "ssl" else "587",
                       "SMTP_USERNAME": "owner@example.com", "SMTP_PASSWORD": "password",
                       "SMTP_FROM": "owner@example.com", "SMTP_SECURITY": security,
                       "HANDOFF_NOTIFICATION_TO": "receiver@example.com",
                       "HANDOFF_ADMIN_URL": "https://example.com/admin"}.items():
        monkeypatch.setenv(key, value)


@pytest.mark.parametrize("security", ["starttls", "ssl"])
def test_smtp_uses_validated_tls_timeout_and_explicit_recipient(service, monkeypatch, security):
    configure_smtp(monkeypatch, security)
    connection = MagicMock()
    connection.send_message.return_value = {}
    factory = MagicMock()
    factory.return_value.__enter__.return_value = connection
    name = "SMTP_SSL" if security == "ssl" else "SMTP"
    monkeypatch.setattr(f"back.infrastructure.notifications.smtplib.{name}", factory)
    notifier = EmailNotifier()
    ticket = submit(service)
    assert notifier.availability("default") == (True, None)
    notifier.send(ticket)
    assert factory.call_args.kwargs["timeout"] == 15
    if security == "starttls":
        context = connection.starttls.call_args.kwargs["context"]
        assert connection.ehlo.call_count == 2
    else:
        context = factory.call_args.kwargs["context"]
    assert context.check_hostname is True
    connection.login.assert_called_once_with("owner@example.com", "password")
    message = connection.send_message.call_args.args[0]
    assert ticket.issue in message.get_content()
    assert ticket.ticket_id in message.get_content()
    assert "https://example.com/admin" in message.get_content()
    assert connection.send_message.call_args.kwargs["to_addrs"] == ["receiver@example.com"]


def test_gmail_grouped_application_password_is_normalized_only_for_gmail(service, monkeypatch):
    configure_smtp(monkeypatch, host="smtp.gmail.com")
    monkeypatch.setenv("SMTP_PASSWORD", " abcd efgh ijkl mnop ")
    connection = MagicMock()
    connection.send_message.return_value = {}
    factory = MagicMock()
    factory.return_value.__enter__.return_value = connection
    monkeypatch.setattr("back.infrastructure.notifications.smtplib.SMTP", factory)
    notifier = EmailNotifier()
    ticket = submit(service)
    notifier.send(ticket)
    connection.login.assert_called_with("owner@example.com", "abcdefghijklmnop")
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    notifier.send(ticket)
    connection.login.assert_called_with("owner@example.com", " abcd efgh ijkl mnop ")


def test_smtp_configuration_and_cross_tenant_recipient_fail_closed(service, monkeypatch):
    notifier = EmailNotifier()
    assert notifier.availability("default")[0] is False
    configure_smtp(monkeypatch)
    monkeypatch.setenv("SMTP_PASSWORD", "")
    assert notifier.availability("default")[0] is False
    monkeypatch.setenv("SMTP_PASSWORD", "password")
    assert notifier.availability("tenant_a")[0] is False
    monkeypatch.setenv("HANDOFF_NOTIFICATION_RECIPIENTS", '{"tenant_a":"a@example.com"}')
    assert notifier.availability("tenant_a")[0] is True
    monkeypatch.setenv("HANDOFF_NOTIFICATION_RECIPIENTS", '{"tenant_a":"a@example.com\\nBcc:evil@example.com"}')
    assert notifier.availability("tenant_a")[0] is False
    monkeypatch.delenv("HANDOFF_NOTIFICATION_RECIPIENTS")
    monkeypatch.setenv("SMTP_FROM", "owner@example.com\nBcc:evil@example.com")
    assert notifier.availability("default")[0] is False


def test_smtp_recipient_rejection_is_a_failed_notification(service, monkeypatch):
    configure_smtp(monkeypatch)
    connection = MagicMock()
    connection.send_message.return_value = {"receiver@example.com": (550, b"refused")}
    factory = MagicMock()
    factory.return_value.__enter__.return_value = connection
    monkeypatch.setattr("back.infrastructure.notifications.smtplib.SMTP", factory)
    service.notifier = EmailNotifier()
    ticket = submit(service)
    assert service.dispatch(ticket.ticket_id, "default") is False
    saved = service.get("default", ticket.ticket_id)
    assert saved.notification_status == "failed"
    assert "SMTPRecipientsRefused" in saved.notification_error


def test_public_unconfigured_receipt_is_truthful_and_keeps_saved_request(client, service):
    service.notifier = EmailNotifier()
    response = client.post("/handoff", json={"session_id": "chat", "submission_id": "unconfigured",
                                             "issue": "需要检查掉单"})
    assert response.status_code == 200
    receipt = response.json()
    assert receipt["notification_status"] == "unconfigured"
    assert "暂未发出" in receipt["message"]
    assert "邮件已发送" not in receipt["message"]
    saved = service.get("default", receipt["ticket_id"])
    assert saved.issue == "需要检查掉单"
    assert saved.notification_attempts == 0
    assert saved.notification_status == "unconfigured"


@pytest.mark.parametrize("admin_url", ["https://[invalid", "file:///admin", "https://user:password@example.com/admin"])
def test_invalid_admin_url_is_unconfigured_without_exposing_value(service, monkeypatch, admin_url):
    configure_smtp(monkeypatch)
    monkeypatch.setenv("HANDOFF_ADMIN_URL", admin_url)
    available, error = EmailNotifier().availability("default")
    assert available is False
    assert admin_url not in error
