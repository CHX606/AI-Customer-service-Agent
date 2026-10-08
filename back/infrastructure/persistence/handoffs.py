"""持久化人工申请与邮件待发状态，使用原子领取避免并发重复投递。"""
from __future__ import annotations
import json
from contextlib import closing
from pathlib import Path

from back.domain.errors import Conflict
from back.domain.handoff import HandoffTicket
from back.infrastructure.persistence import tenants
from back.infrastructure.persistence.errors import storage_operation


@storage_operation
def init_handoff_db(db_path: Path | str | None = None) -> None:
    with closing(tenants.get_db_connection(db_path)) as conn, conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS handoff_requests (
            ticket_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            submission_id TEXT NOT NULL,
            issue TEXT NOT NULL,
            messages_json TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK(status IN ('pending','processing','resolved')),
            handling_note TEXT NOT NULL DEFAULT '',
            notification_status TEXT NOT NULL
                CHECK(notification_status IN ('pending','sending','sent','failed','unconfigured')),
            notification_error TEXT,
            notification_attempts INTEGER NOT NULL DEFAULT 0,
            next_retry_at TEXT,
            notification_claim TEXT,
            notification_claim_at TEXT,
            sent_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(tenant_id, submission_id),
            FOREIGN KEY(tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE
        )""")
        conn.execute("""CREATE INDEX IF NOT EXISTS idx_handoff_tenant_created
            ON handoff_requests(tenant_id, created_at DESC)""")
        conn.execute("""CREATE INDEX IF NOT EXISTS idx_handoff_notification_due
            ON handoff_requests(notification_status, next_retry_at)""")


def _ticket(row) -> HandoffTicket:
    values = dict(row)
    values["messages"] = json.loads(values.pop("messages_json"))
    return HandoffTicket(**values)


class SQLiteHandoffRepository:
    def __init__(self, db_path: Path | str | None = None):
        # None intentionally follows the current tenant database (including test isolation).
        self.db_path = db_path

    def _connect(self):
        init_handoff_db(self.db_path)
        return tenants.get_db_connection(self.db_path)

    @storage_operation
    def create(self, ticket: HandoffTicket) -> HandoffTicket:
        with closing(self._connect()) as conn, conn:
            conn.execute("""INSERT INTO handoff_requests(
                ticket_id,tenant_id,session_id,submission_id,issue,messages_json,
                notification_status,notification_error,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(tenant_id,submission_id) DO NOTHING""",
                (ticket.ticket_id, ticket.tenant_id, ticket.session_id, ticket.submission_id,
                 ticket.issue, json.dumps(ticket.messages, ensure_ascii=False),
                 ticket.notification_status, ticket.notification_error,
                 ticket.created_at, ticket.updated_at))
            row = conn.execute("""SELECT * FROM handoff_requests
                WHERE tenant_id=? AND submission_id=?""",
                (ticket.tenant_id, ticket.submission_id)).fetchone()
            saved = _ticket(row)
            if saved.session_id != ticket.session_id or saved.issue != ticket.issue:
                raise Conflict("这次提交的编号已被使用，请重新提交问题。")
            return saved

    @storage_operation
    def get(self, tenant_id: str, ticket_id: str) -> HandoffTicket | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM handoff_requests WHERE tenant_id=? AND ticket_id=?",
                               (tenant_id, ticket_id)).fetchone()
        return _ticket(row) if row else None

    @storage_operation
    def list(self, tenant_id: str, *, status: str | None = None, limit: int = 50,
             offset: int = 0) -> list[HandoffTicket]:
        condition = " AND status=?" if status is not None else ""
        parameters = (tenant_id, status, limit, offset) if status is not None else (tenant_id, limit, offset)
        with closing(self._connect()) as conn:
            rows = conn.execute(f"""SELECT * FROM handoff_requests WHERE tenant_id=?{condition}
                ORDER BY created_at DESC,ticket_id DESC LIMIT ? OFFSET ?""", parameters).fetchall()
        return [_ticket(row) for row in rows]

    @storage_operation
    def update(self, tenant_id: str, ticket_id: str, status: str, handling_note: str,
               now: str) -> HandoffTicket | None:
        with closing(self._connect()) as conn, conn:
            conn.execute("""UPDATE handoff_requests SET status=?,handling_note=?,updated_at=?
                WHERE tenant_id=? AND ticket_id=?""", (status, handling_note, now, tenant_id, ticket_id))
        return self.get(tenant_id, ticket_id)

    @storage_operation
    def claim(self, tenant_id: str, ticket_id: str, claim: str, now: str,
              max_attempts: int) -> HandoffTicket | None:
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute("""UPDATE handoff_requests SET notification_status='sending',
                notification_claim=?,notification_claim_at=?,notification_error=NULL,
                notification_attempts=notification_attempts+1,updated_at=?
                WHERE tenant_id=? AND ticket_id=?
                AND notification_status IN ('pending','failed','unconfigured')
                AND notification_attempts<? AND (next_retry_at IS NULL OR next_retry_at<=?)""",
                (claim, now, now, tenant_id, ticket_id, max_attempts, now))
            if cursor.rowcount != 1:
                return None
            row = conn.execute("SELECT * FROM handoff_requests WHERE tenant_id=? AND ticket_id=?",
                               (tenant_id, ticket_id)).fetchone()
            return _ticket(row)

    @storage_operation
    def finish(self, tenant_id: str, ticket_id: str, claim: str, *, status: str,
               error: str | None, now: str, next_retry_at: str | None = None) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("""UPDATE handoff_requests SET notification_status=?,notification_error=?,
                next_retry_at=?,notification_claim=NULL,notification_claim_at=NULL,
                sent_at=CASE WHEN ?='sent' THEN ? ELSE sent_at END,updated_at=?
                WHERE tenant_id=? AND ticket_id=? AND notification_status='sending'
                AND notification_claim=?""",
                (status, error, next_retry_at, status, now, now, tenant_id, ticket_id, claim))

    @storage_operation
    def mark_unconfigured(self, tenant_id: str, ticket_id: str, error: str,
                          now: str, next_retry_at: str) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("""UPDATE handoff_requests SET notification_status='unconfigured',
                notification_error=?,next_retry_at=?,updated_at=?
                WHERE tenant_id=? AND ticket_id=? AND notification_status IN ('pending','failed','unconfigured')""",
                (error, next_retry_at, now, tenant_id, ticket_id))

    @storage_operation
    def pending(self, now: str, max_attempts: int, limit: int = 5) -> list[HandoffTicket]:
        with closing(self._connect()) as conn:
            rows = conn.execute("""SELECT * FROM handoff_requests
                WHERE notification_status IN ('pending','failed','unconfigured')
                AND notification_attempts<? AND (next_retry_at IS NULL OR next_retry_at<=?)
                ORDER BY created_at,ticket_id LIMIT ?""", (max_attempts, now, limit)).fetchall()
        return [_ticket(row) for row in rows]

    @storage_operation
    def queue_retry(self, tenant_id: str, ticket_id: str, now: str) -> HandoffTicket | None:
        with closing(self._connect()) as conn, conn:
            row = conn.execute("SELECT * FROM handoff_requests WHERE tenant_id=? AND ticket_id=?",
                               (tenant_id, ticket_id)).fetchone()
            if row is None:
                return None
            if row["notification_status"] == "sending":
                raise Conflict("邮件通知正在发送，请稍后查看。")
            if row["notification_status"] != "sent":
                conn.execute("""UPDATE handoff_requests SET notification_status='pending',
                    notification_error=NULL,notification_attempts=0,next_retry_at=NULL,updated_at=?
                    WHERE tenant_id=? AND ticket_id=? AND notification_status!='sending'
                    AND notification_status!='sent'""", (now, tenant_id, ticket_id))
        return self.get(tenant_id, ticket_id)

    @storage_operation
    def recover_interrupted(self, cutoff: str, now: str, max_attempts: int) -> None:
        # SMTP may have accepted the email before a crash. Require manual retry instead of duplicating it.
        with closing(self._connect()) as conn, conn:
            conn.execute("""UPDATE handoff_requests SET notification_status='failed',
                notification_error='邮件投递状态未知，请在后台确认后重试。',
                notification_attempts=?,notification_claim=NULL,notification_claim_at=NULL,
                next_retry_at=NULL,updated_at=?
                WHERE notification_status='sending' AND notification_claim_at<?""",
                (max_attempts, now, cutoff))
