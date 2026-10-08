"""人工申请与后台处理网关，复用现有聊天和管理员鉴权。"""
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from back.domain.handoff import HandoffTicket
from back.domain.tenant import validate_tenant_id
from back.interfaces.http.auth import verify_admin_authorization, verify_chat_tenant_authorization

router = APIRouter(tags=["Human Handoff"])


def get_handoff_service():
    # Local import avoids a router/bootstrap initialization cycle.
    from back.bootstrap import get_handoff_service as service_factory
    return service_factory()


class HandoffRequest(BaseModel):
    tenant_id: str = "default"
    session_id: str = Field(min_length=1, max_length=100)
    submission_id: str = Field(min_length=1, max_length=200)
    issue: str = Field(min_length=1, max_length=2000)

    @field_validator("tenant_id")
    @classmethod
    def tenant(cls, value: str) -> str:
        return validate_tenant_id(value)

    @field_validator("session_id", "submission_id", "issue")
    @classmethod
    def nonblank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("该字段不能为空。")
        return value


class HandoffUpdate(BaseModel):
    tenant_id: str = "default"
    status: Literal["pending", "processing", "resolved"]
    handling_note: str | None = Field(default=None, max_length=4000)

    @field_validator("tenant_id")
    @classmethod
    def tenant(cls, value: str) -> str:
        return validate_tenant_id(value)


def _authorize(tenant_id: str, key: str | None) -> str:
    try:
        tenant_id = validate_tenant_id(tenant_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    verify_admin_authorization(tenant_id, key)
    return tenant_id


def _receipt(ticket: HandoffTicket) -> dict:
    message = "问题已登记，等待人工查看处理。"
    if ticket.notification_status == "sent":
        message += "通知邮件已发送。"
    elif ticket.notification_status in {"unconfigured", "failed"}:
        message += "通知邮件暂未发出，申请已保存在后台。"
    else:
        message += "正在安排通知。"
    return {"ticket_id": ticket.ticket_id, "status": ticket.status,
            "notification_status": ticket.notification_status, "message": message}


def _summary(ticket: HandoffTicket) -> dict:
    return {"ticket_id": ticket.ticket_id, "issue": ticket.issue, "created_at": ticket.created_at,
            "updated_at": ticket.updated_at, "status": ticket.status,
            "notification_status": ticket.notification_status, "notification_error": ticket.notification_error}


@router.post("/handoff")
def submit_handoff(request: HandoffRequest,
                   x_tenant_token: str | None = Header(default=None, alias="X-Tenant-Token"),
                   service=Depends(get_handoff_service)):
    tenant_id = verify_chat_tenant_authorization(request.tenant_id, x_tenant_token)
    ticket = service.submit(tenant_id, request.session_id, request.submission_id, request.issue)
    return _receipt(ticket)


@router.get("/admin/handoffs")
def list_handoffs(tenant_id: str = Query(default="default"),
                  status: Literal["pending", "processing", "resolved"] | None = Query(default=None),
                  limit: int = Query(default=50, ge=1, le=100), offset: int = Query(default=0, ge=0),
                  x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
                  service=Depends(get_handoff_service)):
    tenant_id = _authorize(tenant_id, x_admin_key)
    return [_summary(ticket) for ticket in service.list(tenant_id, status=status, limit=limit, offset=offset)]


@router.get("/admin/handoffs/{ticket_id}")
def get_handoff(ticket_id: str, tenant_id: str = Query(default="default"),
                x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
                service=Depends(get_handoff_service)):
    ticket = service.get(_authorize(tenant_id, x_admin_key), ticket_id)
    detail = _summary(ticket)
    detail.update(tenant_id=ticket.tenant_id, session_id=ticket.session_id,
                  messages=ticket.messages, handling_note=ticket.handling_note, sent_at=ticket.sent_at)
    return detail


@router.put("/admin/handoffs/{ticket_id}")
def update_handoff(ticket_id: str, request: HandoffUpdate,
                   x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
                   service=Depends(get_handoff_service)):
    ticket = service.update(_authorize(request.tenant_id, x_admin_key), ticket_id,
                            request.status, request.handling_note)
    detail = _summary(ticket)
    detail.update(tenant_id=ticket.tenant_id, session_id=ticket.session_id,
                  messages=ticket.messages, handling_note=ticket.handling_note, sent_at=ticket.sent_at)
    return detail


@router.post("/admin/handoffs/{ticket_id}/notification/retry")
def retry_handoff_notification(ticket_id: str,
                               tenant_id: str = Query(default="default"),
                               x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
                               service=Depends(get_handoff_service)):
    tenant_id = _authorize(tenant_id, x_admin_key)
    ticket = service.retry_notification(tenant_id, ticket_id)
    return _receipt(ticket)
