"""
检索证据充分性分析模块。

本模块负责判断混合检索返回的候选资料是否足以支持回答用户问题。
它只进行证据判断，不生成最终客服回答。
"""

# Python 自带的 JSON 工具。
# 这里主要用来把问题、意图和检索资料整理成一段格式清楚的文字，
# 然后一起交给大模型进行判断。
import json


# Document 是 LangChain 用来保存文档内容的格式。
# 我们从知识库检索出来的每一个 chunk，都是一个 Document。
# 它里面主要有：
# page_content：当前文本块的正文内容。
# metadata：章节编号、章节标题、检索分数等额外信息。
from langchain_core.documents import Document


# BaseChatModel 表示一个可以聊天的大模型。
# 这里不是创建模型，而是规定 analyze_evidence 接收的 llm
# 必须是一个 LangChain 支持的聊天模型。
# 因此以后更换模型时，这个函数通常不需要跟着修改。
from langchain_core.language_models.chat_models import BaseChatModel


# SystemMessage 用来告诉模型需要完成什么任务、遵守什么规则。
# HumanMessage 用来把本次需要分析的问题和资料发送给模型。
# 它们相当于给不同内容标明“系统要求”和“用户输入”两种身份。
from langchain_core.messages import HumanMessage, SystemMessage


# BaseModel 用来规定大模型必须返回什么样的结构。
# Field 用来解释每个字段的作用，并添加必要的格式限制。
# 例如要求模型必须返回 evidence_status、判断理由和支持资料编号。
from pydantic import BaseModel, Field


# 从 state.py 导入项目已经统一规定好的字段类型。
# EvidenceStatus：证据是否充分，例如 sufficient、insufficient。
# IntentType：用户问题属于哪种业务，例如流量、套餐、退款。
# 这样可以避免不同文件使用不一致的状态名称。
from back.domain.conversation import EvidenceStatus, IntentType


# 这个类用来规定“证据判断节点”必须返回哪些内容。
#
# 大模型拿到用户问题和检索出的多个文档后，需要按照这个类的格式返回结果。
# 这样后面的 LangGraph 节点就能通过固定字段判断：
# 1. 资料能不能回答问题。
# 2. 哪几份资料真正有用。
# 3. 是否还需要向用户追问。
# 4. 是否应该拒答或转人工。
class EvidenceAnalysis(BaseModel):

    # 记录本次检索资料是否足够支持回答。
    #
    # 只能使用 EvidenceStatus 中规定的四种结果：
    # sufficient：资料足够，可以根据知识库回答。
    # insufficient：资料有一定帮助，但缺少用户的关键信息，需要追问。
    # not_found：检索出的资料无法回答这个问题。
    # conflict：不同资料的说法互相冲突，不能直接选择其中一个回答。
    evidence_status: EvidenceStatus = Field(
        description=(
            "sufficient表示资料足以回答；"
            "insufficient表示还需要用户补充信息；"
            "not_found表示资料无法回答；"
            "conflict表示资料之间存在冲突。"
        )
    )

    # 记录真正能够支持回答的参考资料编号。
    #
    # 例如模型认为第1份和第3份资料有用，就返回：
    # [1, 3]
    #
    # 编号从1开始，对应我们发送给模型的“参考资料1、参考资料2……”。
    # default_factory=list 表示没有支持资料时，默认返回空列表 []。
    supporting_document_indexes: list[int] = Field(
        default_factory=list,
        description=(
            "能够直接支持判断的参考资料编号，从1开始。"
        ),
    )

    # 记录当前还缺少哪些用户信息。
    #
    # 这个字段主要在 evidence_status 为 insufficient 时使用。
    # 例如：
    # ["套餐是否已经到账", "使用的客户端名称"]
    #
    # 如果不缺少信息，默认返回空列表 []。
    missing_information: list[str] = Field(
        default_factory=list,
        description=(
            "evidence_status为insufficient时仍需用户提供的信息。"
        ),
    )

    # 根据缺少的信息，生成一句需要向用户提出的追问。
    #
    # 例如：
    # "请问套餐现在已经显示到账了吗？"
    #
    # str | None 表示它可以是一段文字，也可以没有内容。
    # 不需要追问时返回 None。
    clarifying_question: str | None = Field(
        default=None,
        description=(
            "需要用户补充信息时提出的一个针对性问题。"
        ),
    )

    # 简单说明为什么作出当前的证据判断。
    #
    # 这个字段主要用于后端日志、测试和排查问题，
    # 不会直接作为客服答案发送给用户。
    decision_reason: str = Field(
        description="简短说明证据判断依据，不超过两句话。"
    )


# 这个函数负责判断：
# “检索出来的这些知识库资料，到底能不能回答用户的问题？”
#
# 它不会生成最终客服答案，只负责检查证据是否充分，
# 然后返回前面定义好的 EvidenceAnalysis 结构。
def evaluate_evidence(
    llm: BaseChatModel,        # 项目当前使用的大模型。
    resolved_query: str,       # 经过上下文整理后的完整用户问题。
    intent: IntentType,        # 用户问题所属的业务类型。
    documents: list[Document], # Reranker 精排后返回的候选资料。
) -> EvidenceAnalysis:
    """判断检索资料能否支持回答完整用户问题。"""

    # 要求大模型严格按照 EvidenceAnalysis 的字段格式返回结果。
    #
    # 如果不使用结构化输出，模型可能只返回一段普通文字，
    # 后面的代码就很难准确获得 evidence_status、资料编号和追问内容。
    structured_llm = llm.with_structured_output(
        EvidenceAnalysis
    )

    # 给证据判断模型设置工作规则。
    #
    # 这里重点告诉模型：
    # 1. 它只负责判断证据，不负责回答用户。
    # 2. 只能使用检索资料，不能依靠自己的知识。
    # 3. 什么情况下算资料充分、资料不足、没有答案或资料冲突。
    system_prompt = """
你是智能客服系统中的检索证据判断节点。

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
- “明确支持”不要求候选资料与用户问题逐字一致。
- 如果资料描述的业务场景、因果关系或处理方法能够直接推出答案，可以判断为 sufficient。
- 允许使用资料正文能够直接推出的单步结论，但不得加入资料之外的新条件、金额、时限或处理方式。
- 如果资料说明用户本应购买操作B，却误购操作A，并因此没有实现目标，则该资料能够支持“操作A不能实现操作B的效果”。
- 不得仅因为资料使用陈述句、场景描述，而用户使用疑问句或口语表达，就判断为 not_found。
- not_found只用于候选资料确实无法提供答案的情况，不能仅以“没有逐字定义”为理由。
- 不得把用户描述的事实当成知识库证据。
- 不得根据常识补全资料中没有的信息。
- supporting_document_indexes只填写直接支持判断的资料编号。
- sufficient时supporting_document_indexes不能为空。
- not_found时supporting_document_indexes必须为空。
- insufficient时missing_information不能为空，并生成一个具体追问。
- conflict时填写产生冲突的资料编号。
- 追问用户能够观察或确认的信息，不要要求用户判断技术原因。
- decision_reason只说明判断依据，不要生成最终客服回答。

示例：

正例：
用户问“续费后不会有流量吗”，资料说明“流量用尽后本应购买重置却买成续费，因此还是没流量，并引导购买重置”，应判断为 sufficient。

反例：
用户问“续费会赠送多少GB流量”，资料没有提供赠送流量数量，应判断为 not_found，不能自行推测。
"""

    # 用来保存整理后的候选资料。
    #
    # 原来的 documents 是 LangChain 的 Document 对象，
    # 不适合直接放进 JSON，因此需要先转换成普通字典。
    formatted_documents = []

    # 遍历 Reranker 返回的所有候选资料。
    #
    # start=1 表示资料编号从1开始，
    # 这样模型返回的编号可以直接对应“参考资料1、参考资料2……”。
    for index, document in enumerate(
        documents,
        start=1,
    ):
        # 从 Document 中提取证据判断真正需要的信息。
        formatted_documents.append(
            {
                # 当前资料的编号。
                "document_index": index,

                # 当前资料在知识库中的章节编号。
                "section_id": document.metadata.get(
                    "section_id"
                ),

                # 当前资料的章节标题。
                "section_title": document.metadata.get(
                    "section_title"
                ),

                # 当前资料的正文内容，这是模型判断证据的主要依据。
                "content": document.page_content,
            }
        )

    # 将用户问题、业务意图和候选资料整理到一个字典中。
    # 这就是本次证据判断需要交给模型的全部数据。
    input_data = {
        "resolved_query": resolved_query,
        "intent": intent,
        "candidate_documents": formatted_documents,
    }

    # 正式调用大模型进行证据判断。
    result = structured_llm.invoke(
        [
            # SystemMessage 放证据判断的规则。
            SystemMessage(content=system_prompt),

            # HumanMessage 放本次真正需要判断的问题和候选资料。
            HumanMessage(
                content=json.dumps(
                    input_data,

                    # 保留中文，避免转换成 Unicode 编码形式。
                    ensure_ascii=False,

                    # 添加缩进，让发送给模型的 JSON 更清楚。
                    indent=2,
                )
            ),
        ]
    )

    # 返回结构化判断结果。
    #
    # result 中包含：
    # evidence_status、supporting_document_indexes、
    # missing_information、clarifying_question 和 decision_reason。
    return result
