"""
业务意图和信息充分性分析模块。

本模块负责：
1. 识别用户当前业务问题的主要意图。
2. 提取用户已经提供的信息。
3. 判断当前信息是否足以进入知识库检索。
4. 信息不足时生成针对性追问。

本模块只处理已经被判断为in_scope的业务问题。
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from back.agent.state import IntentAction, IntentType


class IntentAnalysis(BaseModel):
    intent: IntentType = Field(
        description="用户当前业务问题的主要意图。"
    )

    action: IntentAction = Field(
        description=(
            "retrieve表示信息足够，可以进入检索；"
            "clarify表示信息不足，需要追问。"
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
            "action为retrieve时必须为None。"
        ),
    )

    decision_reason: str = Field(
        description="简短说明为什么选择当前action。"
    )


def analyze_intent(
    llm: BaseChatModel,
    resolved_query: str,
) -> IntentAnalysis:
    """识别业务意图并判断信息是否足够。"""

    structured_llm = llm.with_structured_output(
        IntentAnalysis
    )

    system_prompt = """
你是可乐云AI客服的业务意图和信息充分性分析节点。

输入的问题已经被判断为可乐云业务范围内的问题。

你的任务是：

1. 识别用户的主要业务意图。
2. 提取用户已经明确提供的信息。
3. 判断当前信息是否足以进入知识库检索。
4. 信息不足时，生成一个针对性追问。

意图包括：

- account：账号相关问题。
- order_payment：订单、购买和支付问题。
- package：套餐购买、到账和使用问题。
- renewal：续费相关问题。
- traffic：流量查询、重置和异常问题。
- subscription_import：订阅导入相关问题。
- node_connection：节点、连接和网络异常。
- software_usage：客户端安装、配置和使用问题。
- refund：退款和售后问题。
- other_business：其他明确的可乐云业务问题。
- unknown：暂时无法识别具体业务意图。

动作包括：

- retrieve：信息足够，可以进入知识库检索。
- clarify：信息不足，需要向用户追问。

判断规则：

- 判断的是“是否足以进入检索”，不是“是否已经确定最终故障原因”。
- 问题表达口语化不等于信息不足。
- 如果用户已经描述了明确的故障场景或可观察现象，
  应选择retrieve，即使该现象背后可能存在多种原因。
- “不能用”“有问题”“连不上”等笼统结论本身不算具体的可观察现象。
  如果没有说明发生在哪个业务阶段、客户端状态或错误表现，应选择clarify。
- 只有问题可能对应多个完全不同的业务阶段，
  无法确定检索方向时，才选择clarify。
- 不要为了确定最终根因，在第一次检索前过度追问技术细节。
- 不要把助手提出的问题当成用户已经确认的信息。
- 不要编造用户没有提供的事实。
- 不要根据知识库是否存在答案判断action。
- clarify时只询问最关键的一个问题。
- 追问用户能够观察到的现象，不要要求用户判断故障原因。
- 不要使用“请提供更多信息”这种宽泛追问。
- retrieve时missing_information必须为空，
  clarifying_question必须为null。
- clarify时missing_information不能为空，
  clarifying_question必须是明确问题。

示例：

用户说“续费成功后流量没有重置”：
意图为traffic，信息足够，action为retrieve。

用户说“我购买了套餐，但是不能用”：
意图为package，但可能处于未到账、未导入或连接失败等不同阶段，
信息不足，action为clarify。

用户说“节点不能用”：
没有说明是无法连接、连接后无法上网还是客户端报错，
信息不足，action为clarify。

用户说“Windows上的Clash Verge显示连接成功，但无法打开网页”：
已经描述了平台、客户端状态和故障现象，
足以确定连接故障的检索方向，action为retrieve。
"""

    result = structured_llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"待分析的问题：{resolved_query}"
            ),
        ]
    )

    return result
