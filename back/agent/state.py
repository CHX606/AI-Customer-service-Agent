"""
该文件规范了系统内部状态，LangGraph节点与节点之间的信息传递都需要按照该文件所规范的去做。
"""


from typing import Literal, NotRequired, TypedDict

from langchain_core.documents import Document
from langgraph.graph import MessagesState


# 上下文关系类型：约束 relation 字段，用来表示当前消息与上一问题的关系。
RelationType = Literal[
    "continue",    # 继续当前问题，例如补充故障信息或回答上一轮追问。
    "new_issue",   # 开始一个新的、可以独立理解的问题。
    "correction",  # 否定或修正之前提供的信息。
    "uncertain",   # 现有信息不足，暂时无法确定新旧问题关系。
]

# 业务范围类型：约束 scope 字段，用来决定问题应该进入哪个处理分支。
ScopeType = Literal[
    "in_scope",      # 属于可乐云客服业务范围，需要继续识别意图或检索知识库。
    "chitchat",      # 简单问候、感谢或告别，不需要检索知识库。
    "out_of_scope",  # 明确不属于可乐云业务范围，例如天气、新闻等问题。
    "uncertain",     # 信息太少，无法判断是否属于客服业务范围。
]

# 业务意图类型：约束 intent 字段，用来表示用户具体在咨询哪类业务问题。
IntentType = Literal[
    "account",              # 账号登录、注册、账号状态或账号信息相关问题。
    "order_payment",        # 订单状态、支付失败、支付成功未生效或掉单问题。
    "package",              # 套餐购买、套餐到账、套餐状态或更换套餐问题。
    "renewal",              # 套餐续费、续费规则或续费结果相关问题。
    "traffic",              # 流量查询、流量消耗、流量显示或流量重置问题。
    "subscription_import",  # 订阅链接获取、导入、更新或订阅未生效问题。
    "node_connection",      # 节点不可用、连接失败或连接后无法上网的问题。
    "software_usage",       # 客户端下载、安装、配置或操作方法问题。
    "refund",               # 退款条件、退款规则或退款处理流程问题。
    "other_business",       # 属于可乐云业务，但不属于以上已定义类别的问题。
    "unknown",              # 无法根据当前信息可靠识别具体业务意图。
]

# 问题生命周期类型：约束 active_issue.status，记录当前业务问题处理到哪一步。
IssueStatus = Literal[
    "open",           # 问题正在分析、检索或处理中。
    "awaiting_user",  # 系统已经追问，正在等待用户补充关键信息。
    "answered",       # 系统已经给出答案，但用户尚未确认问题真正解决。
    "resolved",       # 用户明确表示问题已经解决。
    "handed_off",     # 知识库无法安全处理，已经建议或执行转人工。
]


class ActiveIssue(TypedDict):
    """当前业务问题的结构化状态。"""

    summary: str
    status: IssueStatus
    intent: NotRequired[IntentType | None]
    last_clarifying_question: NotRequired[str | None]

IntentAction = Literal[
    "retrieve",
    "clarify",
]

EvidenceStatus = Literal[
    "sufficient",
    "insufficient",
    "not_found",
    "conflict",
]


class CustomerServiceState(MessagesState):
    active_issue: NotRequired[ActiveIssue | None]

    relation: NotRequired[RelationType | None]

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
