"""
业务范围判断模块。

本模块负责判断整理后的用户问题是否属于可乐云客服业务范围。

判断结果包括：

- in_scope：属于可乐云客服业务。
- chitchat：简单问候、感谢或告别。
- out_of_scope：明确与可乐云业务无关。
- uncertain：信息不足，暂时无法判断。

本模块不负责上下文边界判断、知识库检索或回答生成。
"""

from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


class ScopeAnalysis(BaseModel):
    scope: Literal[
        "in_scope",
        "chitchat",
        "out_of_scope",
        "uncertain",
    ] = Field(
        description="用户问题所属的业务范围类型。"
    )

    scope_reason: str = Field(
        description="简短说明范围判断依据，不超过两句话。"
    )


def analyze_scope(
    llm: BaseChatModel,
    resolved_query: str,
) -> ScopeAnalysis:
    """判断完整用户问题所属的业务范围。"""

    structured_llm = llm.with_structured_output(
        ScopeAnalysis
    )

    system_prompt = """
你是可乐云AI客服的业务范围判断节点。

你只负责判断问题属于哪一种范围，不要回答用户问题。

范围包括：

1. in_scope
与可乐云产品或服务有关的问题，例如账号、套餐、购买、支付、
订单、续费、退款、流量、订阅、节点、网络连接、客户端、
软件安装和配置。

2. chitchat
问候、感谢、告别等简单对话，例如“你好”“谢谢”“再见”。

3. out_of_scope
明确与可乐云业务无关的问题，例如天气、新闻、做饭、数学和编程。

4. uncertain
信息太少，无法判断是否属于可乐云业务。

判断规则：

- 业务问题即使知识库可能没有答案，也属于in_scope。
- 模糊的业务问题仍然可以属于in_scope。
- 不要根据知识库是否能检索到资料判断业务范围。
- 不要回答问题。
- scope_reason只说明判断依据，不超过两句话。
"""

    result = structured_llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"待判断的问题：{resolved_query}"
            ),
        ]
    )

    return result