"""独立 SQLite 会话库；JSON 序列化、租户隔离和乐观并发控制。"""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from langchain_core.messages import messages_from_dict, messages_to_dict

from back.application.ports import SessionRepository
from back.domain.chat import ChatState, SessionKey, SessionSnapshot
from back.domain.errors import Conflict, DependencyUnavailable


class SQLiteSessionRepository(SessionRepository):
    def __init__(self, path: Path, history_limit: int = 40):
        self.path = Path(path)
        self.history_limit = history_limit

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=5)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""CREATE TABLE IF NOT EXISTS chat_sessions (
                tenant_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                state_json TEXT NOT NULL,
                version INTEGER NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (tenant_id, session_id)
            )""")
            conn.commit()
            return conn
        except Exception:
            conn.close()
            raise

    def load(self, key: SessionKey) -> SessionSnapshot:
        try:
            with closing(self._connect()) as conn:
                row = conn.execute(
                    "SELECT state_json, version FROM chat_sessions WHERE tenant_id=? AND session_id=?",
                    key,
                ).fetchone()
            if row is None:
                return SessionSnapshot()
            state = json.loads(row[0])
            state["messages"] = messages_from_dict(state.get("messages", []))
            return SessionSnapshot(state, row[1])
        except (OSError, sqlite3.Error, ValueError, TypeError, KeyError) as exc:
            raise DependencyUnavailable("会话读取失败，请稍后重试。") from exc

    def save(self, key: SessionKey, state: ChatState, expected_version: int) -> None:
        # 检索候选和向量不属于后续对话必需状态，不持久化大块文档。
        persisted = {
            "messages": messages_to_dict(state.get("messages", [])[-self.history_limit:]),
            "active_issue": state.get("active_issue"),
        }
        try:
            payload = json.dumps(persisted, ensure_ascii=False)
            with closing(self._connect()) as conn, conn:
                if expected_version == 0:
                    cursor = conn.execute(
                        """INSERT INTO chat_sessions (tenant_id, session_id, state_json, version)
                        VALUES (?, ?, ?, 1) ON CONFLICT(tenant_id, session_id) DO NOTHING""",
                        (*key, payload),
                    )
                else:
                    cursor = conn.execute(
                        """UPDATE chat_sessions SET state_json=?, version=version+1,
                        updated_at=CURRENT_TIMESTAMP
                        WHERE tenant_id=? AND session_id=? AND version=?""",
                        (payload, *key, expected_version),
                    )
                if cursor.rowcount != 1:
                    raise Conflict("此会话已有新的回复，请刷新后重试。")
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            raise DependencyUnavailable("会话保存失败，请稍后重试。") from exc
