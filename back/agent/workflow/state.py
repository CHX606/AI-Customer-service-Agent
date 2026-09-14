"""LangGraph 运行状态；业务枚举和问题状态定义在 domain。"""
from typing import NotRequired
from langchain_core.documents import Document
from langgraph.graph import MessagesState
from back.domain.conversation import (RelationType, RouteSourceType, ScopeType, IntentType, IssueStatus, ActiveIssue, IntentAction, EvidenceStatus)

class CustomerServiceState(MessagesState):
    tenant_id: NotRequired[str]

    active_issue: NotRequired[ActiveIssue | None]

    relation: NotRequired[RelationType | None]

    route_source: NotRequired[RouteSourceType | None]

    resolved_query: NotRequired[str | None]

    context_reason: NotRequired[str | None]

    scope: NotRequired[ScopeType | None]

    scope_reason: NotRequired[str | None]

    intent: NotRequired[IntentType | None]

    action: NotRequired[IntentAction | None]

    known_information: NotRequired[list[str]]

    missing_information: NotRequired[list[str]]

    clarifying_question: NotRequired[str | None]

    intent_reason: NotRequired[str | None]

    rewritten_queries: NotRequired[list[str]]

    rewrite_reason: NotRequired[str | None]

    search_queries: NotRequired[list[str]]

    retrieval_candidates: NotRequired[list[Document]]

    retrieved_documents: NotRequired[list[Document]]

    evidence_status: NotRequired[EvidenceStatus | None]

    supporting_document_indexes: NotRequired[list[int]]

    supporting_documents: NotRequired[list[Document]]

    evidence_missing_information: NotRequired[list[str]]

    evidence_clarifying_question: NotRequired[str | None]

    evidence_reason: NotRequired[str | None]
