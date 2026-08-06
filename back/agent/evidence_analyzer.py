"""
检索证据充分性分析模块。

本模块负责判断混合检索返回的候选资料是否足以支持回答用户问题。
它只进行证据判断，不生成最终客服回答。
"""

import json

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from back.agent.state import EvidenceStatus, IntentType


class EvidenceAnalysis(BaseModel):
    evidence_status: EvidenceStatus = Field(
        description=(
            "sufficient表示资料足以回答；"
            "insufficient表示还需要用户补充信息；"
            "not_found表示资料无法回答；"
            "conflict表示资料之间存在冲突。"
        )
    )

    supporting_document_indexes: list[int] = Field(
        default_factory=list,
        description=(
            "能够直接支持判断的参考资料编号，从1开始。"
        ),
    )

    missing_information: list[str] = Field(
        default_factory=list,
        description=(
            "evidence_status为insufficient时仍需用户提供的信息。"
        ),
    )

    clarifying_question: str | None = Field(
        default=None,
        description=(
            "需要用户补充信息时提出的一个针对性问题。"
        ),
    )

    decision_reason: str = Field(
        description="简短说明证据判断依据，不超过两句话。"
    )


def evaluate_evidence(
    llm: BaseChatModel,
    resolved_query: str,
    intent: IntentType,
    documents: list[Document],
) -> EvidenceAnalysis:
    """判断检索资料能否支持回答完整用户问题。"""

    structured_llm = llm.with_structured_output(
        EvidenceAnalysis
    )

    system_prompt = """
你是可乐云AI客服系统中的检索证据判断节点。

你的任务不是回答用户问题，而是判断给出的知识库候选资料是否足以支持回答。
只能使用输入中的用户问题和候选资料进行判断，不能使用自己的外部知识。

证据状态包括：

1. sufficient
至少一份资料明确覆盖用户问题的核心事实、规则或处理方法，
可以直接根据资料回答。

2. insufficient
资料与问题方向相关，但资料中的不同处理方式取决于用户尚未提供的关键信息，
需要先向用户提出一个针对性问题。

3. not_found
候选资料只是词语或主题相似，没有明确回答用户的核心问题，
或者资料中完全没有相关规则和处理方法。

4. conflict
两份或多份资料在相同条件下给出了互相矛盾的规则或处理方法，
不能安全地自行选择。

判断规则：

- 检索排名、RRF分数和语义相似不等于证据充分。
- 必须检查资料正文是否明确支持回答。
- 不得把用户描述的事实当成知识库证据。
- 不得根据常识补全资料中没有的信息。
- supporting_document_indexes只填写直接支持判断的资料编号。
- sufficient时supporting_document_indexes不能为空。
- not_found时supporting_document_indexes必须为空。
- insufficient时missing_information不能为空，并生成一个具体追问。
- conflict时填写产生冲突的资料编号。
- 追问用户能够观察或确认的信息，不要要求用户判断技术原因。
- decision_reason只说明判断依据，不要生成最终客服回答。
"""

    formatted_documents = []

    for index, document in enumerate(
        documents,
        start=1,
    ):
        formatted_documents.append(
            {
                "document_index": index,
                "section_id": document.metadata.get(
                    "section_id"
                ),
                "section_title": document.metadata.get(
                    "section_title"
                ),
                "content": document.page_content,
            }
        )

    input_data = {
        "resolved_query": resolved_query,
        "intent": intent,
        "candidate_documents": formatted_documents,
    }

    result = structured_llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=json.dumps(
                    input_data,
                    ensure_ascii=False,
                    indent=2,
                )
            ),
        ]
    )

    return result
