"""业务意图和信息充分性分析模块。

本模块负责：
1. 识别用户当前业务问题的主要意图（含企业介绍与固定资料咨询）。
2. 提取用户已经提供的信息。
3. 判断当前信息是否足以进入知识库检索或企业资料回答。
4. 信息不足时生成针对性追问。
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from back.domain.conversation import IntentAction, IntentType


class IntentAnalysis(BaseModel):
    intent: IntentType = Field(
        description="用户当前业务问题的主要意图。"
    )

    action: IntentAction = Field(
        description=(
            "profile表示企业介绍/营业时间/联系方式等固定资料问题，直接根据企业配置回答；"
            "retrieve表示具体业务信息足够，可以进入检索；"
            "clarify表示业务信息不足，需要追问。"
        )
    )

    known_information: list[str] = Field(
        default_factory=list,
        description="用户已经明确提供的有效信息。",
    )

    missing_information: list[str] = Field(
        default_factory=list,
        description="确定检索方向仍然缺少的关键信息。",
    )

    clarifying_question: str | None = Field(
        default=None,
        description=(
            "需要追问用户的问题；"
            "action为retrieve或profile时必须为None。"
        ),
    )

    decision_reason: str = Field(
        description="简短说明为什么选择当前action。"
    )


def analyze_intent(
    llm: BaseChatModel,
    resolved_query: str,
    company_name: str = "企业",
) -> IntentAnalysis:
    """识别业务意图并判断动作（profile / retrieve / clarify）。"""

    structured_llm = llm.with_structured_output(IntentAnalysis)

    system_prompt = f"""
你是{company_name}AI客服的业务意图和信息充分性分析节点。

输入的问题已经被判断为客服业务范围内的问题。

你的任务是：

1. 识别用户的主要业务意图。
2. 提取用户已经明确提供的信息。
3. 判断当前信息属于哪种处理动作。
4. 信息不足时，生成一个针对性追问。

意图包括：

- company_info：企业介绍、企业定位、这是做什么的、产品用途、业务范围、营业时间、联系方式、人工服务时间等企业基本资料咨询。
- account：账号相关问题（注册、登录、修改密码等）。
- order_payment：订单、购买和支付问题。
- package：套餐购买、到账和使用问题。
- renewal：续费相关问题。
- traffic：流量查询、重置和异常问题。
- subscription_import：订阅导入相关问题。
- node_connection：节点、连接和网络异常。
- software_usage：客户端安装、配置和使用问题。
- refund：退款和售后问题。
- other_business：其他明确的业务问题。
- unknown：暂时无法识别具体业务意图。

动作包括：

- profile：属于 company_info（企业介绍、做什么的、营业时间、联系方式等），直接根据结构化企业资料回答，不进入知识库检索。
- retrieve：业务信息足够，可以进入知识库检索。
- clarify：业务信息不足，需要向用户追问。

判断规则：

- 凡是询问“这是干啥的”“你们是谁”“介绍一下”“营业时间是什么”“怎么联系你们”“人工客服电话/邮箱”等问题，意图必须为 company_info，action 必须为 profile。
- 当 action 为 profile 或 retrieve 时，missing_information 必须为空，clarifying_question 必须为 null。
- 如果用户已经描述了明确的业务故障场景或可观察现象，应选择 retrieve。
- “不能用”“有问题”“连不上”等笼统故障，如果没有说明发生阶段或现象，选择 clarify。
- clarify 时 missing_information 不能为空，clarifying_question 必须是明确问题。
- clarify 时只询问最关键的一个问题。

示例：

用户说“你们是做什么的”：
意图为 company_info，action 为 profile。

用户说“营业时间是什么时候”：
意图为 company_info，action 为 profile。

用户说“续费成功后流量没有重置”：
意图为 traffic，信息足够，action 为 retrieve。

用户说“我购买了套餐，但是不能用”：
意图为 package，信息不足，action 为 clarify。
"""

    result = structured_llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"待分析的问题：{resolved_query}"),
        ]
    )

    return result
