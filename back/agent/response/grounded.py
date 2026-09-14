"""在一次模型调用中完成证据判断与知识库回答。"""

import json
import re

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from back.domain.conversation import EvidenceStatus, IntentType


class GroundedResponse(BaseModel):
    """知识库回答及其证据状态。"""

    evidence_status: EvidenceStatus = Field(description="证据状态")
    supporting_document_indexes: list[int] = Field(
        default_factory=list,
        description="直接支持结论的资料编号",
    )
    # 放在证据状态和资料编号之后，流式接口只有确认二者有效后才会发送回答 Token。
    answer: str | None = Field(default=None, description="最多4句的中文客服答复")
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

    if status == "sufficient" and (not valid_indexes or not answer):
        status = "not_found"
        answer = None
        valid_indexes = []
        reason = f"{reason} 充分证据或回答字段无效，系统按未找到可靠答案处理。".strip()
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
        if status != "sufficient":
            answer = None
        if status == "not_found":
            valid_indexes = []

    return response.model_copy(
        update={
            "evidence_status": status,
            "answer": answer,
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
你是{company_name}的{assistant_name}，仅依据documents判断并回答，禁止补充外部规则、原因、价格或处理方式。

- sufficient：正文可直接回答；提供有效资料编号，并用{tone}中文在4句内回答。
- insufficient：资料相关但缺少用户关键情况；只提出一个针对性问题。
- not_found：资料不能回答；answer=null，不猜测。
- conflict：相同条件下至少两份资料矛盾；提供两个编号，answer=null。
- 排名和相似度不代表证据充分。回答不得提及资料编号或内部判断。
- 按结构顺序输出，decision_reason不超过20字。
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
