"""LangGraph 适配器：将框架事件转换为应用事件，按需加载模型。"""
import json

def _partial_json_string(payload: str, field: str) -> tuple[str, bool] | None:
    """从尚未完成的 JSON 中安全解码字符串字段当前已经生成的部分。"""
    marker = f'"{field}"'
    marker_index = payload.find(marker)
    if marker_index < 0:
        return None
    colon_index = payload.find(":", marker_index + len(marker))
    if colon_index < 0:
        return None
    cursor = colon_index + 1
    while cursor < len(payload) and payload[cursor].isspace():
        cursor += 1
    if cursor >= len(payload) or payload[cursor] != '"':
        return None
    cursor += 1

    decoded: list[str] = []
    escape_map = {
        '"': '"',
        "\\": "\\",
        "/": "/",
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
    }
    while cursor < len(payload):
        character = payload[cursor]
        if character == '"':
            return "".join(decoded), True
        if character != "\\":
            decoded.append(character)
            cursor += 1
            continue
        if cursor + 1 >= len(payload):
            break
        escaped = payload[cursor + 1]
        if escaped == "u":
            digits = payload[cursor + 2 : cursor + 6]
            if len(digits) < 4:
                break
            try:
                decoded.append(chr(int(digits, 16)))
            except ValueError:
                return None
            cursor += 6
            continue
        replacement = escape_map.get(escaped)
        if replacement is None:
            return None
        decoded.append(replacement)
        cursor += 2
    return "".join(decoded), False

def _grounded_answer_is_streamable(payload: str, document_count: int) -> bool:
    """仅在模型已经声明证据充分且提供有效资料编号后开放回答流。"""
    try:
        prefix = payload.split('"answer"', maxsplit=1)[0]
        partial = json.loads(prefix.rstrip().rstrip(",") + "}")
    except (json.JSONDecodeError, TypeError, ValueError):
        return False
    if partial.get("evidence_status") != "sufficient":
        return False
    indexes = partial.get("supporting_document_indexes")
    return isinstance(indexes, list) and any(
        isinstance(index, int) and 1 <= index <= document_count
        for index in indexes
    )


class LangGraphEngine:
    @property
    def graph(self):
        from back.agent.workflow.graph import customer_service_graph
        return customer_service_graph

    def invoke(self, state):
        return self.graph.invoke(state)

    def stream(self, state):
        final_state = None
        emitted = set()
        streamed_answer = ""
        grounded_payload = ""
        for stream_event in self.graph.stream(
            state,
            stream_mode=["values", "messages"],
        ):
            if (
                isinstance(stream_event, tuple)
                and len(stream_event) == 2
                and stream_event[0] in {"values", "messages"}
            ):
                stream_mode, stream_data = stream_event
            else:
                # 兼容旧图实现以及测试替身返回的 values 事件。
                stream_mode, stream_data = "values", stream_event

            if stream_mode == "messages":
                chunk, metadata = stream_data
                content = chunk.content if isinstance(chunk.content, str) else ""
                node = metadata.get("langgraph_node")
                delta = ""
                if node in {"answer_profile", "answer", "chitchat"}:
                    delta = content
                elif node == "grounded_answer":
                    grounded_payload += content
                    document_count = len(
                        (final_state or {}).get("retrieved_documents", [])
                    )
                    partial_answer = _partial_json_string(
                        grounded_payload,
                        "answer",
                    )
                    if (
                        partial_answer is not None
                        and _grounded_answer_is_streamable(
                            grounded_payload,
                            document_count,
                        )
                    ):
                        answer_prefix, _complete = partial_answer
                        delta = answer_prefix[len(streamed_answer) :]
                if delta:
                    streamed_answer += delta
                    yield dict({"type": "token", "delta": delta})
                continue

            state = stream_data
            final_state = state
            yield {"type": "state", "state": state}

            if state.get("scope") is not None and "analyzed" not in emitted:
                emitted.add("analyzed")
                if state.get("scope") == "in_scope" and state.get("action") == "retrieve":
                    yield dict(
                        {
                            "type": "status",
                            "message": "正在查询知识库…",
                            "route_source": state.get("route_source"),
                        }
                    )
                else:
                    yield dict(
                        {
                            "type": "status",
                            "message": "正在组织回复…",
                            "route_source": state.get("route_source"),
                        }
                    )

            if "retrieval_candidates" in state and "retrieved" not in emitted:
                emitted.add("retrieved")
                yield dict({"type": "status", "message": "正在筛选相关资料…"})

            if "retrieved_documents" in state and "reranked" not in emitted:
                emitted.add("reranked")
                reranked_documents = state.get("retrieved_documents", [])
                reranker_metadata = (
                    reranked_documents[0].metadata
                    if reranked_documents
                    else {}
                )
                yield dict(
                    {
                        "type": "status",
                        "message": "正在生成回答…",
                        "reranker_queue_wait_ms": reranker_metadata.get(
                            "reranker_queue_wait_ms"
                        ),
                        "reranker_inference_ms": reranker_metadata.get(
                            "reranker_inference_ms"
                        ),
                        "reranker_max_concurrency": reranker_metadata.get(
                            "reranker_max_concurrency"
                        ),
                    }
                )
