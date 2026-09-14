"""聊天用例的数据契约，不包含传输协议或数据库字段。"""

from dataclasses import dataclass, field
from typing import Any

ChatState = dict[str, Any]
SessionKey = tuple[str, str]


@dataclass(frozen=True)
class ChatCommand:
    message: str
    session_id: str
    tenant_id: str = "default"
    turn_id: str | None = None
    regenerate: bool = False
    previous_answer_hash: str | None = None


@dataclass(frozen=True)
class ChatResult:
    answer: str
    session_id: str
    tenant_id: str


@dataclass(frozen=True)
class SessionSnapshot:
    state: ChatState = field(default_factory=dict)
    version: int = 0


@dataclass(frozen=True)
class PreparedChat:
    tenant_id: str
    session_id: str
    session_key: SessionKey
    graph_input: ChatState
    version: int = 0
    cache_revision: int | None = 0
    regenerate: bool = False


@dataclass(frozen=True)
class CacheHit:
    answer: str
    similarity: float
    exact: bool
