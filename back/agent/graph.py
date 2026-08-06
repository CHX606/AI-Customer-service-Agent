"""
LangGraph客服流程定义模块。

本模块负责注册和连接各个Agent节点，包括：

1. 对话上下文边界判断。
2. 客服业务范围判断。
3. 业务意图和信息充分性判断。
4. 混合检索、Reranker精排和证据判断。
5. 追问、拒答、寒暄和知识库回答路由。
"""

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.graph import END, START, StateGraph

from back.agent.evidence_analyzer import evaluate_evidence
from back.agent.intent_analyzer import analyze_intent
from back.agent.issue_lifecycle import (
    set_issue_intent,
    start_or_continue_issue,
    transition_issue,
    user_confirms_resolution,
)
from back.agent.query_analyzer import analyze_context
from back.agent.query_rewriter import rewrite_query
from back.agent.scope_analyzer import analyze_scope
from back.agent.state import CustomerServiceState
from back.llm import model
from back.rag.hybrid_retriever import (
    RRF_CANDIDATES,
    retrieve_documents_multi_query,
)
from back.rag.reranker import rerank_documents


SYSTEM_PROMPT = """
你是可乐云的AI客服助手。

请遵守以下规则：

1. 回答业务问题时，只能使用系统提供的知识库参考资料。
2. 多份资料可以互补，但不能添加资料中不存在的业务规则。
3. 不要编造知识库中不存在的规则、价格或处理方式。
4. 如果知识库没有明确答案，请说明暂未查到，并建议联系人工客服。
5. 使用简洁、友好、容易理解的中文回答。
"""


def build_recent_messages(
    messages: list[BaseMessage],
    limit: int = 6,
) -> list[dict[str, str]]:
    """将LangChain消息转换成上下文分析需要的格式。"""

    recent_messages = []

    for message in messages:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue

        if not message.content:
            continue

        recent_messages.append(
            {
                "role": role,
                "content": str(message.content),
            }
        )

    return recent_messages[-limit:]


def analyze_context_node(state: CustomerServiceState):
    """判断当前消息与正在处理的问题之间的关系。"""

    messages = state["messages"]
    current_user_index = None

    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            current_user_index = index
            break

    if current_user_index is None:
        raise ValueError("消息列表中没有找到用户消息")

    current_query = str(messages[current_user_index].content)
    previous_messages = messages[:current_user_index]
    recent_messages = build_recent_messages(previous_messages)

    analysis = analyze_context(
        llm=model,
        current_query=current_query,
        recent_messages=recent_messages,
        active_issue=state.get("active_issue"),
    )

    return {
        "relation": analysis.relation,
        "resolved_query": analysis.resolved_query,
        "context_reason": analysis.decision_reason,
    }


def analyze_scope_node(state: CustomerServiceState):
    """判断完整用户问题是否属于客服业务范围。"""

    resolved_query = state.get("resolved_query")

    if not resolved_query:
        raise ValueError("状态中没有找到resolved_query")

    active_issue = state.get("active_issue")

    current_query = ""

    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            current_query = str(message.content)
            break

    if active_issue and user_confirms_resolution(current_query):
        return {
            "scope": "chitchat",
            "scope_reason": "用户明确表示当前业务问题已经解决。",
            "active_issue": transition_issue(
                active_issue,
                "resolved",
            ),
        }

    analysis = analyze_scope(
        llm=model,
        resolved_query=resolved_query,
    )

    relation = state.get("relation")

    if analysis.scope == "in_scope":
        active_issue = start_or_continue_issue(
            active_issue=active_issue,
            relation=relation,
            summary=resolved_query,
        )

    return {
        "scope": analysis.scope,
        "scope_reason": analysis.scope_reason,
        "active_issue": active_issue,
    }


def analyze_intent_node(state: CustomerServiceState):
    """识别业务意图并判断当前信息是否足以检索。"""

    resolved_query = state.get("resolved_query")

    if not resolved_query:
        raise ValueError("状态中没有找到resolved_query")

    analysis = analyze_intent(
        llm=model,
        resolved_query=resolved_query,
    )

    active_issue = set_issue_intent(
        state.get("active_issue"),
        analysis.intent,
    )

    return {
        "active_issue": active_issue,
        "intent": analysis.intent,
        "action": analysis.action,
        "known_information": analysis.known_information,
        "missing_information": analysis.missing_information,
        "clarifying_question": analysis.clarifying_question,
        "intent_reason": analysis.decision_reason,
    }


def rewrite_query_node(state: CustomerServiceState):
    """生成标准检索问题，并与原问题组成检索查询列表。"""

    resolved_query = state.get("resolved_query")
    intent = state.get("intent")

    if not resolved_query:
        raise ValueError("状态中没有找到resolved_query")

    if not intent:
        raise ValueError("状态中没有找到intent")

    analysis = rewrite_query(
        llm=model,
        resolved_query=resolved_query,
        intent=intent,
        known_information=state.get("known_information", []),
    )

    search_queries = []
    seen_queries = set()

    for query in [
        resolved_query,
        *analysis.rewritten_queries,
    ]:
        cleaned_query = " ".join(query.split())

        if not cleaned_query or cleaned_query in seen_queries:
            continue

        seen_queries.add(cleaned_query)
        search_queries.append(cleaned_query)

    return {
        "rewritten_queries": analysis.rewritten_queries,
        "rewrite_reason": analysis.rewrite_reason,
        "search_queries": search_queries,
    }


def hybrid_search_node(state: CustomerServiceState):
    """对原问题和改写问题执行多查询宽召回。"""

    search_queries = state.get("search_queries", [])
    resolved_query = state.get("resolved_query")

    if not search_queries and resolved_query:
        search_queries = [resolved_query]

    if not search_queries:
        raise ValueError("状态中没有找到可用的检索问题")

    documents = retrieve_documents_multi_query(
        queries=search_queries,
        limit=RRF_CANDIDATES,
    )

    return {
        "retrieval_candidates": documents,
    }


def rerank_node(state: CustomerServiceState):
    """使用CrossEncoder对RRF候选资料进行精排。"""

    resolved_query = state.get("resolved_query")
    candidates = state.get("retrieval_candidates", [])

    if not resolved_query:
        raise ValueError("状态中没有找到resolved_query")

    documents = rerank_documents(
        query=resolved_query,
        documents=candidates,
    )

    return {
        "retrieved_documents": documents,
    }


def evaluate_evidence_node(state: CustomerServiceState):
    """判断检索候选资料是否足以支持回答。"""

    resolved_query = state.get("resolved_query")
    intent = state.get("intent")
    documents = state.get("retrieved_documents", [])

    if not resolved_query:
        raise ValueError("状态中没有找到resolved_query")

    if not intent:
        raise ValueError("状态中没有找到intent")

    if not documents:
        return {
            "evidence_status": "not_found",
            "supporting_document_indexes": [],
            "supporting_documents": [],
            "evidence_missing_information": [],
            "evidence_clarifying_question": None,
            "evidence_reason": "混合检索没有返回候选资料。",
        }

    analysis = evaluate_evidence(
        llm=model,
        resolved_query=resolved_query,
        intent=intent,
        documents=documents,
    )

    valid_indexes = []

    for index in analysis.supporting_document_indexes:
        if (
            1 <= index <= len(documents)
            and index not in valid_indexes
        ):
            valid_indexes.append(index)

    evidence_status = analysis.evidence_status
    evidence_reason = analysis.decision_reason

    if evidence_status == "sufficient" and not valid_indexes:
        evidence_status = "not_found"
        evidence_reason = (
            "模型判断证据充分，但没有提供有效的支持文档编号，"
            "因此按证据不足处理。"
        )

    if evidence_status == "conflict" and len(valid_indexes) < 2:
        evidence_status = "not_found"
        evidence_reason = (
            "模型判断资料存在冲突，但没有提供至少两份有效资料，"
            "因此按暂未找到可靠答案处理。"
        )

    if evidence_status == "not_found":
        valid_indexes = []

    supporting_documents = []

    if evidence_status == "sufficient":
        supporting_documents = [
            documents[index - 1]
            for index in valid_indexes
        ]

    clarifying_question = analysis.clarifying_question
    missing_information = analysis.missing_information

    if evidence_status == "insufficient":
        if not missing_information:
            missing_information = [
                "确定处理方式所需的关键情况"
            ]

        if not clarifying_question:
            clarifying_question = (
                "可以再说明一下当前页面显示的具体状态吗？"
            )
    else:
        missing_information = []
        clarifying_question = None

    return {
        "evidence_status": evidence_status,
        "supporting_document_indexes": valid_indexes,
        "supporting_documents": supporting_documents,
        "evidence_missing_information": missing_information,
        "evidence_clarifying_question": clarifying_question,
        "evidence_reason": evidence_reason,
    }


def respond_out_of_scope(_state: CustomerServiceState):
    """对明确不属于客服业务的问题进行固定拒答。"""

    return {
        "messages": [
            AIMessage(
                content=(
                    "抱歉，我目前只能处理可乐云相关的账号、套餐、"
                    "续费、流量、节点和软件使用等客服问题。"
                )
            )
        ]
    }


def respond_scope_uncertain(_state: CustomerServiceState):
    """要求用户补充具体的业务问题。"""

    return {
        "messages": [
            AIMessage(
                content="把问题说的具体一点，并且附上问题截图"
            )
        ]
    }


def respond_clarify(state: CustomerServiceState):
    """返回意图分析节点生成的针对性追问。"""

    question = state.get("clarifying_question")

    if not question:
        question = "可以具体说说您遇到的是哪一种情况吗？"

    return {
        "active_issue": transition_issue(
            state.get("active_issue"),
            "awaiting_user",
            last_clarifying_question=question,
        ),
        "messages": [
            AIMessage(content=question)
        ]
    }


def respond_evidence_clarify(state: CustomerServiceState):
    """根据证据判断结果向用户追问关键缺失信息。"""

    question = state.get("evidence_clarifying_question")

    if not question:
        question = "可以再说明一下当前页面显示的具体状态吗？"

    return {
        "active_issue": transition_issue(
            state.get("active_issue"),
            "awaiting_user",
            last_clarifying_question=question,
        ),
        "messages": [
            AIMessage(content=question)
        ]
    }


def respond_handoff(state: CustomerServiceState):
    """在资料无答案或存在冲突时建议转人工。"""

    if state.get("evidence_status") == "conflict":
        message = (
            "抱歉，目前检索到的资料存在不一致，"
            "我无法安全地给出确定结论，建议联系人工客服核实。"
        )
    else:
        message = (
            "抱歉，目前知识库中暂未查到能够明确回答该问题的资料，"
            "建议联系人工客服进一步处理。"
        )

    return {
        "active_issue": transition_issue(
            state.get("active_issue"),
            "handed_off",
        ),
        "messages": [
            AIMessage(content=message)
        ]
    }


def respond_chitchat(state: CustomerServiceState):
    """处理简单寒暄，但不提供知识库工具。"""

    current_message = None

    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            current_message = message
            break

    if current_message is None:
        raise ValueError("消息列表中没有找到用户消息")

    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "你是可乐云AI客服。只回应用户当前的简单问候、"
                    "感谢或告别，使用自然、友好、简洁的中文。"
                    "不要查询知识库，也不要继续之前的业务问题。"
                )
            ),
            current_message,
        ]
    )

    return {
        "messages": [response],
    }


def route_after_scope(state: CustomerServiceState) -> str:
    """根据业务范围判断结果选择下一节点。"""

    scope = state.get("scope")

    if scope == "in_scope":
        return "analyze_intent"

    if scope == "chitchat":
        return "chitchat"

    if scope == "out_of_scope":
        return "out_of_scope"

    return "scope_uncertain"


def route_after_intent(state: CustomerServiceState) -> str:
    """根据用户信息是否充分选择检索或追问。"""

    if state.get("action") == "retrieve":
        return "rewrite_query"

    return "clarify"


def route_after_evidence(state: CustomerServiceState) -> str:
    """根据证据充分性选择回答、追问或转人工。"""

    evidence_status = state.get("evidence_status")

    if evidence_status == "sufficient":
        return "answer"

    if evidence_status == "insufficient":
        return "evidence_clarify"

    return "handoff"


def answer_with_knowledge(state: CustomerServiceState):
    """让大模型严格根据混合检索资料生成回答。"""

    resolved_query = state.get("resolved_query")
    relation = state.get("relation")
    scope = state.get("scope")
    intent = state.get("intent")
    search_queries = state.get("search_queries", [])
    documents = state.get("supporting_documents", [])

    formatted_search_queries = "\n".join(
        f"- {query}"
        for query in search_queries
    )

    formatted_documents = "\n\n".join(
        (
            f"参考资料 {index}\n"
            f"章节：{document.metadata.get('section_id', '未知')} "
            f"{document.metadata.get('section_title', '')}\n"
            f"内容：{document.page_content}"
        )
        for index, document in enumerate(
            documents,
            start=1,
        )
    )

    if not formatted_documents:
        formatted_documents = "没有检索到参考资料。"

    context_prompt = f"""
问题分析结果：

当前需要处理的完整问题：
{resolved_query}

当前消息与上一问题的关系：
{relation}

当前问题的业务范围：
{scope}

当前业务意图：
{intent}

原问题和改写后的检索候选：
{formatted_search_queries}

知识库参考资料：
{formatted_documents}

只能根据“知识库参考资料”回答用户问题。
问题分析结果只用于理解用户意图，不能作为业务知识或回答依据。
"""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=context_prompt),
        *state["messages"],
    ]

    response = model.invoke(messages)

    return {
        "active_issue": transition_issue(
            state.get("active_issue"),
            "answered",
        ),
        "messages": [response],
    }


graph_builder = StateGraph(CustomerServiceState)

graph_builder.add_node("analyze_context", analyze_context_node)
graph_builder.add_node("analyze_scope", analyze_scope_node)
graph_builder.add_node("analyze_intent", analyze_intent_node)
graph_builder.add_node("rewrite_query", rewrite_query_node)
graph_builder.add_node("hybrid_search", hybrid_search_node)
graph_builder.add_node("rerank", rerank_node)
graph_builder.add_node("evaluate_evidence", evaluate_evidence_node)
graph_builder.add_node("out_of_scope", respond_out_of_scope)
graph_builder.add_node("scope_uncertain", respond_scope_uncertain)
graph_builder.add_node("clarify", respond_clarify)
graph_builder.add_node("evidence_clarify", respond_evidence_clarify)
graph_builder.add_node("handoff", respond_handoff)
graph_builder.add_node("chitchat", respond_chitchat)
graph_builder.add_node("answer", answer_with_knowledge)

graph_builder.add_edge(START, "analyze_context")
graph_builder.add_edge("analyze_context", "analyze_scope")

graph_builder.add_conditional_edges(
    "analyze_scope",
    route_after_scope,
    {
        "analyze_intent": "analyze_intent",
        "chitchat": "chitchat",
        "out_of_scope": "out_of_scope",
        "scope_uncertain": "scope_uncertain",
    },
)

graph_builder.add_conditional_edges(
    "analyze_intent",
    route_after_intent,
    {
        "rewrite_query": "rewrite_query",
        "clarify": "clarify",
    },
)

graph_builder.add_edge("rewrite_query", "hybrid_search")
graph_builder.add_edge("hybrid_search", "rerank")
graph_builder.add_edge("rerank", "evaluate_evidence")

graph_builder.add_conditional_edges(
    "evaluate_evidence",
    route_after_evidence,
    {
        "answer": "answer",
        "evidence_clarify": "evidence_clarify",
        "handoff": "handoff",
    },
)

graph_builder.add_edge("chitchat", END)
graph_builder.add_edge("out_of_scope", END)
graph_builder.add_edge("scope_uncertain", END)
graph_builder.add_edge("clarify", END)
graph_builder.add_edge("evidence_clarify", END)
graph_builder.add_edge("handoff", END)
graph_builder.add_edge("answer", END)

customer_service_graph = graph_builder.compile()


if __name__ == "__main__":
    question = "续费之后为什么流量没有重置？"

    result = customer_service_graph.invoke(
        {
            "messages": [
                HumanMessage(content=question),
            ],
        }
    )

    for message in result["messages"]:
        message.pretty_print()
