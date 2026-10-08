"""在一次模型调用中完成证据判断与知识库回答。"""

import json
import re

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from back.domain.conversation import EvidenceStatus, IntentType


ANSWER_STATUSES = ("sufficient", "partial")

# 客服内部快捷回复（如“改邮箱”快捷键）和内部群不能作为用户步骤；模型常把它改写成
# “快捷入口/快捷流程”。只匹配带引号的名称，避免误伤“Ctrl+C 快捷键”这类软件操作说明。
INTERNAL_OPERATION_PATTERN = re.compile(
    r"[“\"'‘「『][^”\"'’」』\n]{1,20}[”\"'’」』]\s*快捷(?:键|回复|入口|流程|方式|指令)|工作群"
)


class GroundedResponse(BaseModel):
    """知识库回答及其证据状态。"""

    evidence_status: EvidenceStatus = Field(description="证据状态")
    supporting_document_indexes: list[int] = Field(
        default_factory=list,
        description="直接支持结论的资料编号",
    )
    # 状态在回答之前输出，流式接口确认可回答且存在候选资料后发送 Token。
    answer: str | None = Field(default=None, description="最多4句的中文客服答复")
    needs_human: bool = Field(default=False, description="当前请求是否需要站长人工办理或确认")
    missing_information: list[str] = Field(default_factory=list)
    clarifying_question: str | None = None
    decision_reason: str = Field(description="不超过20字的内部判断依据")


def sanitize_grounded_answer(answer: str) -> str:
    """移除模型偶尔泄露的内部资料编号，不改写业务内容。"""
    citation = r"(?:参考)?资料\s*\d+(?:\s*[、,，和及]\s*\d+)*"
    cleaned = re.sub(
        rf"^\s*(?:根据\s*)?{citation}\s*[：:,，]\s*",
        "",
        answer,
    )
    cleaned = re.sub(
        rf"\s*[（(\[]\s*{citation}\s*[）)\]]\s*$",
        "",
        cleaned,
    )
    cleaned = re.sub(
        rf"\s+{citation}\s*[。.]?\s*$",
        "",
        cleaned,
    )
    return cleaned.strip()


def normalize_grounded_response(
    response: GroundedResponse,
    *,
    document_count: int,
) -> GroundedResponse:
    """修正无效文档编号和互相矛盾的模型字段。"""
    valid_indexes: list[int] = []
    for index in response.supporting_document_indexes:
        if 1 <= index <= document_count and index not in valid_indexes:
            valid_indexes.append(index)

    status = response.evidence_status
    reason = response.decision_reason
    answer = sanitize_grounded_answer(response.answer) if response.answer else None

    if status in ANSWER_STATUSES and (not answer or document_count <= 0):
        status = "not_found"
        answer = None
        valid_indexes = []
        reason = f"{reason} 回答为空或无候选资料，按未找到处理。".strip()
    elif status in ANSWER_STATUSES and INTERNAL_OPERATION_PATTERN.search(answer):
        status = "not_found"
        answer = None
        valid_indexes = []
        reason = f"{reason} 回答含客服内部操作，按未找到处理。".strip()
    elif status in ANSWER_STATUSES and not valid_indexes:
        valid_indexes = [1]
        reason = f"{reason} 未给有效编号，回填第1名。".strip()
    elif status == "conflict" and len(valid_indexes) < 2:
        status = "not_found"
        answer = None
        valid_indexes = []
        reason = f"{reason} 冲突资料编号不足，系统按未找到可靠答案处理。".strip()

    if status == "insufficient":
        missing = response.missing_information or ["确定处理方式所需的关键情况"]
        question = response.clarifying_question or "可以再说明一下当前页面显示的具体状态吗？"
        answer = None
        valid_indexes = []
    else:
        missing = []
        question = None
        if status not in ANSWER_STATUSES:
            answer = None
        if status == "not_found":
            valid_indexes = []

    return response.model_copy(
        update={
            "evidence_status": status,
            "answer": answer,
            "needs_human": status == "partial" or (status in ANSWER_STATUSES and response.needs_human),
            "supporting_document_indexes": valid_indexes,
            "missing_information": missing,
            "clarifying_question": question,
            "decision_reason": reason,
        }
    )


def generate_grounded_response(
    *,
    llm: BaseChatModel,
    resolved_query: str,
    intent: IntentType,
    documents: list[Document],
    company_name: str,
    assistant_name: str,
    tone: str,
) -> GroundedResponse:
    """根据候选资料一次生成证据状态和最终答复。"""
    structured_llm = llm.with_structured_output(GroundedResponse)
    formatted_documents = []
    for index, document in enumerate(documents, start=1):
        formatted_documents.append(
            {
                "index": index,
                "title": document.metadata.get("section_title", ""),
                "content": document.page_content,
            }
        )

    system_prompt = f"""
你是{company_name}的{assistant_name}，只依据 documents 判断并回答。

状态：
- sufficient：资料能直接回答，或能从资料正文直接推出一步结论。填写支持资料编号，用{tone}中文在4句内回答。
- partial：资料只能回答问题的一部分。先回答有依据的部分，最后一句说明哪部分需要人工确认。填写支持资料编号，用{tone}中文在4句内回答。
- insufficient：资料里有几种处理方式，要看用户还没说的情况。只提一个用户能自己看到、确认的追问，answer=null。
- not_found：资料确实不涉及这个问题，answer=null。
- conflict：同样条件下，两份以上资料的规则互相矛盾。填写冲突资料的编号，answer=null。

规则：
- "能回答"不要求和资料逐字一致。用户的口语、疑问句，和资料里的陈述句、场景描述说的是同一件事就算。
- 可以做资料正文能直接推出的一步推理，但不能加资料里没有的金额、时限、条件或处理方式。
- 不能只因为"资料里没有原话"就判 not_found。在 sufficient 和 not_found 之间拿不准时，优先考虑 partial；partial 也必须有正文支持可回答的部分，完全无依据仍判 not_found。
- 用户自己描述的情况不算证据；检索排名和相似度也不代表证据充分。
- 资料可能是写给客服人员的内部手册。其中的快捷键或快捷回复名称、内部工作群、"发给后台/转给某人"等，是客服的内部操作，不是用户能执行的步骤：不得让用户使用、查找或点击它们，也不得改写成"快捷入口""菜单""流程"等说法。某个问题的资料只有这类内部操作、没有能直接告诉用户的内容时，判 not_found。
- 你无法查询或修改用户的账户、订单、工单和办理进度，也不能替用户提交、申请、反馈或转交后台。需要人工办理或查询进度的事项，说明需要联系人工客服，并告诉用户要准备什么，不得承诺由你或"我们"代为处理。
- needs_human：用户要求实际办理账户变更、核查个人订单或办理进度，或回答中的未知部分需要人工确认时为 true；仅询问退款条件、续费规则、使用步骤等且资料足以回答时为 false。不要仅因为资料提到人工客服就把政策咨询判为需要人工。
- 需要人工处理时说明用户可以点击“提交人工处理”填写事项。聊天答复尚未创建申请，不得说已提交、已转交或已发邮件。
- 回答和追问都不得向用户索要账号、邮箱、订单号或密码。
- 回答里不要出现资料编号或内部判断。按结构顺序输出，decision_reason 不超过20字。

示例：
- 用户问"续费后不会有流量吗"，资料写着"流量用完后本应购买重置却买成了续费，所以还是没流量，引导购买重置" → sufficient
- 用户问"续费送多少GB流量"，资料没写具体数量 → not_found
- 用户问"怎么改绑定邮箱"，资料只写着"改邮箱：'改邮箱'快捷键" → not_found
"""
    payload = {
        "question": resolved_query,
        "intent": intent,
        "documents": formatted_documents,
    }
    response = structured_llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ]
    )
    return normalize_grounded_response(response, document_count=len(documents))
