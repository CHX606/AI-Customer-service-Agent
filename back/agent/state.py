"""
该文件规范了系统内部状态，LangGraph节点与节点之间的信息传递都需要按照该文件所规范的去做。
"""


from typing import Any, Literal, NotRequired

from langgraph.graph import MessagesState


RelationType = Literal[
    "continue",
    "new_issue",
    "correction",
    "uncertain",
]


class CustomerServiceState(MessagesState):
    active_issue: NotRequired[dict[str, Any] | None]

    relation: NotRequired[RelationType | None]

    resolved_query: NotRequired[str | None]

    context_reason: NotRequired[str | None]