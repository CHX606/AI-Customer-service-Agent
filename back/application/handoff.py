"""保存人工申请，隔离邮件故障，并管理可重试的通知投递。"""
from __future__ import annotations
import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Protocol
from uuid import uuid4

from back.application.ports import SessionRepository, TenantReader
from back.domain.errors import InvalidRequest, NotFound
from back.domain.handoff import HandoffTicket
from back.domain.tenant import validate_tenant_id

logger = logging.getLogger(__name__)
MAX_NOTIFICATION_ATTEMPTS = 3


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


class HandoffRepository(Protocol):
    def create(self, ticket: HandoffTicket) -> HandoffTicket: ...
    def get(self, tenant_id: str, ticket_id: str) -> HandoffTicket | None: ...
    def list(self, tenant_id: str, *, status: str | None, limit: int, offset: int) -> list[HandoffTicket]: ...
    def update(self, tenant_id: str, ticket_id: str, status: str, handling_note: str, now: str) -> HandoffTicket | None: ...
    def claim(self, tenant_id: str, ticket_id: str, claim: str, now: str, max_attempts: int) -> HandoffTicket | None: ...
    def finish(self, tenant_id: str, ticket_id: str, claim: str, *, status: str, error: str | None,
               now: str, next_retry_at: str | None = None) -> None: ...
    def mark_unconfigured(self, tenant_id: str, ticket_id: str, error: str, now: str, next_retry_at: str) -> None: ...
    def pending(self, now: str, max_attempts: int, limit: int) -> list[HandoffTicket]: ...
    def queue_retry(self, tenant_id: str, ticket_id: str, now: str) -> HandoffTicket | None: ...
    def recover_interrupted(self, cutoff: str, now: str, max_attempts: int) -> None: ...


class HandoffNotifier(Protocol):
    def availability(self, tenant_id: str) -> tuple[bool, str | None]: ...
    def send(self, ticket: HandoffTicket) -> None: ...


class HandoffService:
    def __init__(self, repository: HandoffRepository, notifier: HandoffNotifier,
                 sessions: SessionRepository, tenants: TenantReader,
                 *, now: Callable[[], datetime] = _utcnow):
        self.repository = repository
        self.notifier = notifier
        self.sessions = sessions
        self.tenants = tenants
        self.now = now

    def _tenant(self, tenant_id: str) -> str:
        try:
            tenant_id = validate_tenant_id(tenant_id)
        except ValueError as exc:
            raise InvalidRequest(str(exc)) from exc
        if self.tenants.get(tenant_id) is None:
            raise NotFound(f"未找到租户 '{tenant_id}' 的配置资料")
        return tenant_id

    def submit(self, tenant_id: str, session_id: str, submission_id: str, issue: str) -> HandoffTicket:
        tenant_id, session_id, submission_id, issue = self._validated(tenant_id, session_id, submission_id, issue)
        snapshot = self.sessions.load((tenant_id, session_id))
        messages = []
        for message in snapshot.state.get("messages", [])[-40:]:
            role = {"human": "user", "ai": "assistant"}.get(getattr(message, "type", ""))
            content = getattr(message, "content", None)
            if role and isinstance(content, str):
                messages.append({"role": role, "content": content[:4000]})
        return self._create(tenant_id, session_id, submission_id, issue, messages)[0]

    def submit_external(self, tenant_id: str, session_id: str, submission_id: str, issue: str,
                        messages: list[dict[str, str]]) -> tuple[HandoffTicket, bool]:
        """登记来自外部客服渠道（如 Crisp）的会话；聊天记录由调用方提供，不读取本系统会话。

        返回 (申请, 是否新建)；同一提交编号重复登记时返回已有申请，不会重复发邮件。
        """
        tenant_id, session_id, submission_id, issue = self._validated(tenant_id, session_id, submission_id, issue)
        cleaned = [
            {"role": message["role"], "content": str(message["content"])[:4000]}
            for message in messages[-40:]
            if message.get("role") in {"user", "assistant"} and str(message.get("content") or "").strip()
        ]
        return self._create(tenant_id, session_id, submission_id, issue, cleaned)

    def _validated(self, tenant_id: str, session_id: str, submission_id: str,
                   issue: str) -> tuple[str, str, str, str]:
        tenant_id = self._tenant(tenant_id)
        session_id, submission_id, issue = session_id.strip(), submission_id.strip(), issue.strip()
        if not session_id or len(session_id) > 100:
            raise InvalidRequest("会话编号不能为空或超过100个字符。")
        if not submission_id or len(submission_id) > 200:
            raise InvalidRequest("提交编号不能为空或超过200个字符。")
        if not issue or len(issue) > 2000:
            raise InvalidRequest("请填写需要人工处理的问题，最多2000个字符。")
        return tenant_id, session_id, submission_id, issue

    def _create(self, tenant_id: str, session_id: str, submission_id: str, issue: str,
                messages: list[dict[str, str]]) -> tuple[HandoffTicket, bool]:
        available, error = self.notifier.availability(tenant_id)
        now = _stamp(self.now())
        ticket_id = str(uuid4())
        saved = self.repository.create(HandoffTicket(
            ticket_id=ticket_id, tenant_id=tenant_id, session_id=session_id,
            submission_id=submission_id, issue=issue, messages=messages,
            created_at=now, updated_at=now,
            notification_status="pending" if available else "unconfigured",
            notification_error=None if available else error,
        ))
        return saved, saved.ticket_id == ticket_id

    def get(self, tenant_id: str, ticket_id: str) -> HandoffTicket:
        ticket = self.repository.get(self._tenant(tenant_id), ticket_id)
        if ticket is None:
            raise NotFound("未找到该人工处理申请。")
        return ticket

    def list(self, tenant_id: str, *, status: str | None = None, limit: int = 50,
             offset: int = 0) -> list[HandoffTicket]:
        if status is not None and status not in {"pending", "processing", "resolved"}:
            raise InvalidRequest("申请状态无效。")
        if limit < 1 or limit > 100 or offset < 0:
            raise InvalidRequest("分页参数无效。")
        return self.repository.list(self._tenant(tenant_id), status=status, limit=limit, offset=offset)

    def update(self, tenant_id: str, ticket_id: str, status: str,
               handling_note: str | None = None) -> HandoffTicket:
        ticket = self.get(tenant_id, ticket_id)
        if status not in {"pending", "processing", "resolved"}:
            raise InvalidRequest("申请状态无效。")
        note = ticket.handling_note if handling_note is None else handling_note.strip()
        if len(note) > 4000:
            raise InvalidRequest("处理备注最多4000个字符。")
        saved = self.repository.update(tenant_id, ticket_id, status, note, _stamp(self.now()))
        if saved is None:
            raise NotFound("未找到该人工处理申请。")
        return saved

    def retry_notification(self, tenant_id: str, ticket_id: str) -> HandoffTicket:
        self.get(tenant_id, ticket_id)
        ticket = self.repository.queue_retry(tenant_id, ticket_id, _stamp(self.now()))
        if ticket is None:
            raise NotFound("未找到该人工处理申请。")
        return ticket

    def dispatch(self, ticket_id: str, tenant_id: str) -> bool:
        ticket = self.repository.get(tenant_id, ticket_id)
        if ticket is None or ticket.notification_status in {"sent", "sending"}:
            return False
        available, error = self.notifier.availability(tenant_id)
        moment = self.now()
        if not available:
            self.repository.mark_unconfigured(tenant_id, ticket_id, error or "邮件通知尚未配置。",
                _stamp(moment), _stamp(moment + timedelta(seconds=60)))
            return False
        claim = str(uuid4())
        ticket = self.repository.claim(tenant_id, ticket_id, claim, _stamp(moment), MAX_NOTIFICATION_ATTEMPTS)
        if ticket is None:
            return False
        try:
            self.notifier.send(ticket)
        except Exception as exc:
            # Never log exception text: SMTP errors can contain credentials or message content.
            error_type = type(exc).__name__
            logger.warning("人工申请邮件通知失败 ticket=%s type=%s", ticket.ticket_id, error_type)
            delay = 30 * (2 ** (ticket.notification_attempts - 1))
            self.repository.finish(tenant_id, ticket_id, claim, status="failed",
                error=f"邮件通知失败（{error_type}），可在后台重试。", now=_stamp(self.now()),
                next_retry_at=_stamp(self.now() + timedelta(seconds=delay)))
            return False
        self.repository.finish(tenant_id, ticket_id, claim, status="sent", error=None, now=_stamp(self.now()))
        return True

    def dispatch_pending(self, limit: int = 5) -> int:
        if limit < 1 or limit > 100:
            raise InvalidRequest("通知批次数量无效。")
        moment = self.now()
        self.repository.recover_interrupted(_stamp(moment - timedelta(minutes=5)),
                                            _stamp(moment), MAX_NOTIFICATION_ATTEMPTS)
        sent = 0
        for ticket in self.repository.pending(_stamp(moment), MAX_NOTIFICATION_ATTEMPTS, limit):
            sent += bool(self.dispatch(ticket.ticket_id, ticket.tenant_id))
        return sent
