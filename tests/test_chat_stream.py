from back.bootstrap import get_chat_service, get_knowledge_service
"""聊天渐进式响应接口测试。"""

import json
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

import back.interfaces.http.app
from back.interfaces.http.app import app
from back.infrastructure.agent import _partial_json_string
from back.interfaces.http.chat import (
    PreparedChat,
    _prepare_chat,
    _stream_line,
    _stream_prepared_chat,
)
from back.interfaces.http.schemas import ChatRequest


client = TestClient(app)


def prepared(session_id="stream-session") -> PreparedChat:
    return PreparedChat(
        tenant_id="default",
        session_id=session_id,
        session_key=("default", session_id),
        graph_input={
            "tenant_id": "default",
            "messages": [HumanMessage(content="续费后流量没有重置")],
            "active_issue": None,
        },
    )


def final_state(answer="知识库回答"):
    return {
        "tenant_id": "default",
        "messages": [
            HumanMessage(content="续费后流量没有重置"),
            AIMessage(content=answer),
        ],
        "scope": "in_scope",
        "action": "retrieve",
        "route_source": "fast_rule",
        "intent": "traffic",
        "search_queries": ["续费后流量没有重置"],
        "retrieval_candidates": [],
        "retrieved_documents": [],
        "evidence_status": "not_found",
    }


def parse_events(lines):
    return [json.loads(line) for line in lines]


def test_stream_line_preserves_chinese_and_is_newline_delimited():
    line = _stream_line({"type": "status", "message": "正在理解您的问题…"})
    assert line.endswith("\n")
    assert "正在理解" in line
    assert json.loads(line)["type"] == "status"


def test_partial_json_string_decodes_streamed_escapes():
    assert _partial_json_string('{"answer":"第一行\\n第二', "answer") == (
        "第一行\n第二",
        False,
    )
    assert _partial_json_string('{"answer":"包含\\\"引号"}', "answer") == (
        '包含"引号',
        True,
    )


def test_grounded_answer_streams_only_after_valid_evidence_prefix():
    reranked = {
        **prepared().graph_input,
        "scope": "in_scope",
        "action": "retrieve",
        "retrieved_documents": [Document(page_content="资料")],
    }
    chunks = [
        '{"evidence_status":"sufficient",',
        '"supporting_document_indexes":[1],"answer":"续费',
        '后流量按套餐周期重置。",',
        '"missing_information":[],"clarifying_question":null,',
        '"decision_reason":"资料支持"}',
    ]
    graph_events = [("values", reranked)]
    graph_events.extend(
        (
            "messages",
            (
                AIMessageChunk(content=content),
                {"langgraph_node": "grounded_answer"},
            ),
        )
        for content in chunks
    )
    graph_events.append(("values", final_state("续费后流量按套餐周期重置。")))

    with patch.object(
        get_chat_service().engine.graph,
        "stream",
        return_value=iter(graph_events),
    ):
        events = parse_events(_stream_prepared_chat(prepared()))

    assert "".join(
        event["delta"] for event in events if event["type"] == "token"
    ) == "续费后流量按套餐周期重置。"
    assert events[-1]["type"] == "final"


def test_grounded_answer_does_not_stream_unvalidated_content():
    reranked = {
        **prepared().graph_input,
        "scope": "in_scope",
        "action": "retrieve",
        "retrieved_documents": [Document(page_content="资料")],
    }
    unsafe_json = (
        '{"evidence_status":"sufficient",'
        '"supporting_document_indexes":[],"answer":"不应提前输出",'
        '"missing_information":[],"clarifying_question":null,'
        '"decision_reason":"无有效资料编号"}'
    )
    with patch.object(
        get_chat_service().engine.graph,
        "stream",
        return_value=iter(
            [
                ("values", reranked),
                (
                    "messages",
                    (
                        AIMessageChunk(content=unsafe_json),
                        {"langgraph_node": "grounded_answer"},
                    ),
                ),
                ("values", final_state("转人工处理")),
            ]
        ),
    ):
        events = parse_events(_stream_prepared_chat(prepared()))

    assert not [event for event in events if event["type"] == "token"]
    assert events[-1]["answer"] == "转人工处理"


def test_retrieval_stream_emits_ordered_progress_and_final_answer():
    states = [
        prepared().graph_input,
        {
            **prepared().graph_input,
            "scope": "in_scope",
            "action": "retrieve",
            "route_source": "fast_rule",
        },
        {
            **prepared().graph_input,
            "scope": "in_scope",
            "action": "retrieve",
            "retrieval_candidates": [],
        },
        {
            **prepared().graph_input,
            "scope": "in_scope",
            "action": "retrieve",
            "retrieval_candidates": [],
            "retrieved_documents": [
                Document(
                    page_content="资料",
                    metadata={
                        "reranker_queue_wait_ms": 12.5,
                        "reranker_inference_ms": 980.0,
                        "reranker_max_concurrency": 2,
                    },
                )
            ],
        },
        final_state(),
    ]
    with patch.object(get_chat_service().engine.graph, "stream", return_value=iter(states)):
        events = parse_events(_stream_prepared_chat(prepared()))

    assert [event["type"] for event in events] == [
        "status",
        "status",
        "status",
        "status",
        "final",
    ]
    assert [event["message"] for event in events[:-1]] == [
        "正在理解您的问题…",
        "正在查询知识库…",
        "正在筛选相关资料…",
        "正在生成回答…",
    ]
    assert events[-1]["answer"] == "知识库回答"
    assert events[-1]["session_id"] == "stream-session"
    assert events[1]["route_source"] == "fast_rule"
    assert events[3]["reranker_queue_wait_ms"] == 12.5


def test_non_retrieval_stream_does_not_emit_fake_search_progress():
    state = {
        **final_state("您好，请问有什么可以帮您？"),
        "scope": "chitchat",
        "action": "clarify",
    }
    state.pop("retrieval_candidates", None)
    state.pop("retrieved_documents", None)
    with patch.object(
        get_chat_service().engine.graph,
        "stream",
        return_value=iter([prepared().graph_input, state]),
    ):
        events = parse_events(_stream_prepared_chat(prepared()))

    messages = [event.get("message") for event in events if event["type"] == "status"]
    assert messages == ["正在理解您的问题…", "正在组织回复…"]
    assert events[-1]["type"] == "final"


def test_repeated_graph_values_do_not_duplicate_progress_events():
    repeated = {
        **prepared().graph_input,
        "scope": "in_scope",
        "action": "retrieve",
        "retrieval_candidates": [],
        "retrieved_documents": [],
    }
    with patch.object(
        get_chat_service().engine.graph,
        "stream",
        return_value=iter([repeated, repeated, final_state()]),
    ):
        events = parse_events(_stream_prepared_chat(prepared()))

    progress = [event["message"] for event in events if event["type"] == "status"]
    assert progress.count("正在查询知识库…") == 1
    assert progress.count("正在筛选相关资料…") == 1
    assert progress.count("正在生成回答…") == 1


def test_successful_stream_saves_session_state():
    target = prepared("saved-session")
    with patch.object(
        get_chat_service().engine.graph,
        "stream",
        return_value=iter([final_state("已保存回答")]),
    ):
        events = parse_events(_stream_prepared_chat(target))

    assert events[-1]["type"] == "final"
    saved = get_chat_service().sessions.load(('default', 'saved-session')).state
    assert saved["messages"][-1].content == "已保存回答"


def test_failed_stream_returns_generic_error_and_does_not_save_state():
    target = prepared("failed-session")

    def fail_stream(*_args, **_kwargs):
        raise RuntimeError("secret internal failure")
        yield  # pragma: no cover

    with patch.object(get_chat_service().engine.graph, "stream", fail_stream):
        events = parse_events(_stream_prepared_chat(target))

    assert events[-1] == {
        "type": "error",
        "message": "客服请求处理失败，请稍后重试。",
    }
    assert "secret" not in json.dumps(events, ensure_ascii=False)
    assert not get_chat_service().sessions.load(('default', 'failed-session')).state


def test_prepare_chat_reuses_only_same_tenant_session_history(monkeypatch):
    monkeypatch.setenv("PUBLIC_CHAT_TENANTS", "default")
    get_chat_service().sessions.save(('default', 'same-session'), {
        "messages": [HumanMessage(content="旧消息"), AIMessage(content="旧回答")],
        "active_issue": {"summary": "旧问题", "status": "answered"},
    }, expected_version=0)
    request = ChatRequest(
        tenant_id="default",
        session_id="same-session",
        message="继续提问",
    )
    result = _prepare_chat(request)

    assert [message.content for message in result.graph_input["messages"]] == [
        "旧消息",
        "旧回答",
        "继续提问",
    ]
    assert result.graph_input["active_issue"]["summary"] == "旧问题"


def test_stream_endpoint_rejects_unauthorized_tenant_before_streaming():
    response = client.post(
        "/chat/stream",
        json={
            "tenant_id": "default",
            "session_id": "unauthorized",
            "message": "你好",
        },
    )
    assert response.status_code == 403
    assert response.headers["content-type"].startswith("application/json")


def test_stream_endpoint_returns_ndjson_and_no_buffer_headers(monkeypatch):
    monkeypatch.setenv("PUBLIC_CHAT_TENANTS", "default")
    with patch.object(
        get_chat_service().engine.graph,
        "stream",
        return_value=iter([final_state("接口回答")]),
    ):
        response = client.post(
            "/chat/stream",
            json={
                "tenant_id": "default",
                "session_id": "endpoint-stream",
                "message": "你好",
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert response.headers["x-accel-buffering"] == "no"
    events = parse_events(response.text.splitlines())
    assert events[0]["type"] == "status"
    assert events[-1]["answer"] == "接口回答"


def test_stream_endpoint_validates_empty_message_before_streaming(monkeypatch):
    monkeypatch.setenv("PUBLIC_CHAT_TENANTS", "default")
    response = client.post(
        "/chat/stream",
        json={
            "tenant_id": "default",
            "session_id": "invalid-message",
            "message": "",
        },
    )
    assert response.status_code == 422
