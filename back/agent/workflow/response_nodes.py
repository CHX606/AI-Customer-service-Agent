"""客服工作流职责模块：response_nodes"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from back.agent.response.grounded import GroundedResponse, generate_grounded_response
from back.agent.workflow.lifecycle import transition_issue
from back.agent.workflow.state import CustomerServiceState
from back.domain.tenant import TenantProfile
from back.tenant.service import get_tenant_profile
from back.core import llm as models

def _response_llm():
    """测试默认模型可被 patch；独立配置时使用专用回答模型。"""
    return models.model if models.RESPONSE_MODEL_NAME == models.MODEL_NAME else models.get_response_model()

def build_agent_identity_prompt(profile: TenantProfile) -> str:
    """构建统一的动态企业客服助理身份提示词。"""
    scope_text = (
        "、".join(profile.business_scope)
        if profile.business_scope
        else "相关业务"
    )
    hours_text = profile.business_hours or "暂未配置"
    contact_text = profile.public_contact or "暂未配置"

    return (
        f"你是{profile.company_name}的{profile.assistant_name}。\n"
        f"企业简介：{profile.short_description or '暂未配置'}\n"
        f"业务范围：{scope_text}\n"
        f"营业时间：{hours_text}\n"
        f"联系方式：{contact_text}\n"
        f"回复风格要求：{profile.tone}\n"
    )

def answer_from_profile(state: CustomerServiceState):
    """根据结构化企业配置资料直接回答，不经过知识库检索。"""
    tenant_id = state.get("tenant_id", "default")
    profile = get_tenant_profile(tenant_id)
    resolved_query = state.get("resolved_query", "")

    scope_str = (
        "、".join(profile.business_scope)
        if profile.business_scope
        else "未配置具体业务范围"
    )
    prompt_content = f"""
{build_agent_identity_prompt(profile)}

你正在回答用户关于企业基本信息或客服服务指南的咨询（例如“这是干啥的”“你们是谁”“介绍一下”“营业时间是什么”“怎么联系你们”等）。

企业结构化资料如下：
- 企业名称：{profile.company_name}
- 英文品牌名：{profile.brand_name_en or '未配置'}
- 客服助理名称：{profile.assistant_name}
- 企业简介：{profile.short_description or '企业暂未配置简介'}
- 业务范围：{scope_str}
- 营业与服务时间：{profile.business_hours or '企业暂未配置营业时间'}
- 公开联系方式：{profile.public_contact or '企业暂未配置公开联系方式'}

回答规则：
1. 只能使用上述企业结构化资料进行回答。
2. 若某项信息（如简介、营业时间、联系方式）为空或显示“暂未配置”，请明确说明“企业暂未配置对应信息”，切勿编造。
3. 不进入知识库检索，不要提示“知识库未查到”，不要转人工。
4. 使用{profile.tone}的中文进行回答。
"""

    messages = [
        SystemMessage(content=prompt_content),
        HumanMessage(content=resolved_query),
    ]

    response = _response_llm().invoke(messages)

    return {
        "active_issue": transition_issue(
            state.get("active_issue"),
            "answered",
        ),
        "messages": [response],
    }

def respond_with_grounded_knowledge(state: CustomerServiceState):
    """一次完成证据判断，并生成回答、追问或转人工消息。"""
    tenant_id = state.get("tenant_id", "default")
    profile = get_tenant_profile(tenant_id)
    resolved_query = state.get("resolved_query")
    intent = state.get("intent")
    documents = state.get("retrieved_documents", [])

    if not resolved_query:
        raise ValueError("状态中没有找到resolved_query")
    if not intent:
        raise ValueError("状态中没有找到intent")

    if documents:
        result = generate_grounded_response(
            llm=_response_llm(),
            resolved_query=resolved_query,
            intent=intent,
            documents=documents,
            company_name=profile.company_name,
            assistant_name=profile.assistant_name,
            tone=profile.tone,
        )
    else:
        result = GroundedResponse(
            evidence_status="not_found",
            answer=None,
            supporting_document_indexes=[],
            missing_information=[],
            clarifying_question=None,
            decision_reason="混合检索没有返回候选资料。",
        )

    supporting_documents = [
        documents[index - 1]
        for index in result.supporting_document_indexes
        if 1 <= index <= len(documents)
    ]

    if result.evidence_status == "sufficient":
        message = result.answer or profile.handoff_message
        issue_status = "answered"
    elif result.evidence_status == "insufficient":
        message = result.clarifying_question or "可以再说明一下当前页面显示的具体状态吗？"
        issue_status = "awaiting_user"
    elif result.evidence_status == "conflict":
        message = "抱歉，目前检索到的资料存在不一致，我无法安全地给出确定结论，建议联系人工客服核实。"
        issue_status = "handed_off"
    else:
        message = (
            profile.handoff_message
            or "抱歉，目前知识库中暂未查到能够明确回答该问题的资料，建议联系人工客服进一步处理。"
        )
        issue_status = "handed_off"

    return {
        "active_issue": transition_issue(
            state.get("active_issue"),
            issue_status,
            last_clarifying_question=(
                result.clarifying_question
                if issue_status == "awaiting_user"
                else None
            ),
        ),
        "evidence_status": result.evidence_status,
        "supporting_document_indexes": result.supporting_document_indexes,
        "supporting_documents": supporting_documents,
        "evidence_missing_information": result.missing_information,
        "evidence_clarifying_question": result.clarifying_question,
        "evidence_reason": result.decision_reason,
        "messages": [AIMessage(content=message)],
    }

def respond_out_of_scope(state: CustomerServiceState):
    """对明确不属于客服业务的问题进行固定拒答。"""
    tenant_id = state.get("tenant_id", "default")
    profile = get_tenant_profile(tenant_id)
    current_query = ""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            current_query = str(message.content).casefold()
            break
    unsafe_terms = (
        "绕过",
        "规避风控",
        "避开风控",
        "批量注册",
        "伪造ip",
        "伪造位置",
        "盗号",
        "盗回",
    )
    if any(term in current_query for term in unsafe_terms):
        return {
            "messages": [
                AIMessage(
                    content=(
                        "抱歉，我不能帮助绕过平台验证、规避风控、伪造位置、"
                        "批量注册或获取他人账号。请使用平台官方验证与账号恢复流程。"
                    )
                )
            ]
        }
    scope_text = (
        "、".join(profile.business_scope)
        if profile.business_scope
        else "相关业务"
    )
    return {
        "messages": [
            AIMessage(
                content=f"抱歉，我目前只能处理{profile.company_name}相关的{scope_text}等客服咨询。"
            )
        ]
    }

def respond_scope_uncertain(_state: CustomerServiceState):
    """要求用户补充具体的业务问题。"""
    return {
        "messages": [
            AIMessage(content="把问题说的具体一点，并且附上问题截图")
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
        "messages": [AIMessage(content=question)],
    }

def respond_chitchat(state: CustomerServiceState):
    """处理简单寒暄，但不提供知识库工具。"""
    tenant_id = state.get("tenant_id", "default")
    profile = get_tenant_profile(tenant_id)

    current_message = None
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            current_message = message
            break

    if current_message is None:
        raise ValueError("消息列表中没有找到用户消息")

    response = _response_llm().invoke(
        [
            SystemMessage(
                content=(
                    f"你是{profile.company_name}的{profile.assistant_name}。"
                    "只回应用户当前的简单问候、感谢或告别，使用自然、友好、简洁的中文。"
                    "不要查询知识库，也不要继续之前的业务问题。"
                )
            ),
            current_message,
        ]
    )

    return {
        "messages": [response],
    }
