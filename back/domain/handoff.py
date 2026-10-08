"""人工处理申请的数据契约；不包含邮件或数据库实现。"""
from dataclasses import dataclass, field
from typing import Literal

HandoffStatus = Literal["pending", "processing", "resolved"]
NotificationStatus = Literal["pending", "sending", "sent", "failed", "unconfigured"]


@dataclass(frozen=True)
class HandoffTicket:
    ticket_id: str
    tenant_id: str
    session_id: str
    submission_id: str
    issue: str
    created_at: str
    updated_at: str
    status: HandoffStatus = "pending"
    notification_status: NotificationStatus = "pending"
    notification_error: str | None = None
    handling_note: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)
    notification_attempts: int = 0
    next_retry_at: str | None = None
    notification_claim: str | None = None
    notification_claim_at: str | None = None
    sent_at: str | None = None
