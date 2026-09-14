"""旧版分节点流程，仅用于兼容验证；当前主图不注册。"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from back.agent.analysis.evidence import evaluate_evidence
from back.agent.analysis.intent import analyze_intent
from back.agent.workflow.lifecycle import set_issue_intent, start_or_continue_issue, transition_issue, user_confirms_resolution
from back.agent.analysis.context import analyze_context
from back.agent.analysis.rewrite import rewrite_query
from back.agent.analysis.scope import analyze_scope
from back.agent.workflow.state import CustomerServiceState
from back.tenant.service import get_tenant_profile
from back.core import llm as models
from back.agent.workflow.context_nodes import build_recent_messages
from back.agent.workflow.response_nodes import _response_llm, build_agent_identity_prompt

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
        llm=models.model,
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

    tenant_id = state.get("tenant_id", "default")
    profile = get_tenant_profile(tenant_id)
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
        llm=models.model,
        resolved_query=resolved_query,
        company_name=profile.company_name,
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
    """识别业务意图并判断动作（profile / retrieve / clarify）。"""
    resolved_query = state.get("resolved_query")
    if not resolved_query:
        raise ValueError("状态中没有找到resolved_query")

    tenant_id = state.get("tenant_id", "default")
    profile = get_tenant_profile(tenant_id)

    analysis = analyze_intent(
        llm=models.model,
        resolved_query=resolved_query,
        company_name=profile.company_name,
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
        llm=models.model,
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
        llm=models.model,
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
            documents[index - 1] for index in valid_indexes
        ]

    clarifying_question = analysis.clarifying_question
    missing_information = analysis.missing_information

    if evidence_status == "insufficient":
        if not missing_information:
            missing_information = ["确定处理方式所需的关键情况"]
        if not clarifying_question:
            clarifying_question = "可以再说明一下当前页面显示的具体状态吗？"
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
        "messages": [AIMessage(content=question)],
    }

def respond_handoff(state: CustomerServiceState):
    """在资料无答案或存在冲突时建议转人工。"""
    tenant_id = state.get("tenant_id", "default")
    profile = get_tenant_profile(tenant_id)

    if state.get("evidence_status") == "conflict":
        message = (
            "抱歉，目前检索到的资料存在不一致，"
            "我无法安全地给出确定结论，建议联系人工客服核实。"
        )
    else:
        message = (
            profile.handoff_message
            or "抱歉，目前知识库中暂未查到能够明确回答该问题的资料，建议联系人工客服进一步处理。"
        )

    return {
        "active_issue": transition_issue(
            state.get("active_issue"),
            "handed_off",
        ),
        "messages": [AIMessage(content=message)],
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
    """根据用户信息是否充分选择企业资料回答、检索或追问。"""
    action = state.get("action")
    if action == "profile":
        return "answer_profile"
    if action == "retrieve":
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
    tenant_id = state.get("tenant_id", "default")
    profile = get_tenant_profile(tenant_id)

    resolved_query = state.get("resolved_query")
    relation = state.get("relation")
    scope = state.get("scope")
    intent = state.get("intent")
    search_queries = state.get("search_queries", [])
    documents = state.get("supporting_documents", [])

    formatted_search_queries = "\n".join(
        f"- {query}" for query in search_queries
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

    system_prompt = f"""
{build_agent_identity_prompt(profile)}

请遵守以下规则：
1. 回答业务问题时，只能使用系统提供的知识库参考资料。
2. 多份资料可以互补，但不能添加资料中不存在的业务规则。
3. 不要编造知识库中不存在的规则、价格或处理方式。
4. 如果知识库没有明确答案，请说明暂未查到，并建议联系人工客服。
5. 使用{profile.tone}的中文回答。
"""

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
        SystemMessage(content=system_prompt),
        SystemMessage(content=context_prompt),
        *state["messages"],
    ]

    response = _response_llm().invoke(messages)

    return {
        "active_issue": transition_issue(
            state.get("active_issue"),
            "answered",
        ),
        "messages": [response],
    }
