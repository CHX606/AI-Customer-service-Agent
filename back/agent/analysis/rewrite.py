"""
用户查询改写模块。

本模块负责将已经明确的口语化业务问题，改写成适合知识库检索的标准查询。

本模块只负责改写问题，不负责：
1. 判断问题是否明确。
2. 检索知识库。
3. 生成最终回答。
"""

import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from back.domain.conversation import IntentType


class QueryRewriteResult(BaseModel):
    rewritten_queries: list[str] = Field(
        min_length=1,
        max_length=3,
        description=(
            "根据用户原意生成的1到3个标准检索问题。"
        ),
    )

    rewrite_reason: str = Field(
        description=(
            "简短说明改写时标准化了哪些表达。"
        )
    )


def rewrite_query(
    llm: BaseChatModel,
    resolved_query: str,
    intent: IntentType,
    known_information: list[str],
) -> QueryRewriteResult:
    """将明确的用户问题改写成标准检索问题。"""

    structured_llm = llm.with_structured_output(
        QueryRewriteResult
    )

    system_prompt = """
你是智能客服系统中的查询改写节点。

输入的问题已经完成了上下文整理、业务范围判断和信息充分性判断，
可以进入知识库检索。

你的任务是把用户问题改写成1到3个适合知识库检索的标准问题。

改写规则：

1. 只改写问题，不回答问题。
2. 使用准确、简洁的客服业务术语。
3. 保留用户已经明确提供的事实。
4. 不得添加用户没有提供的信息。
5. 不得猜测故障原因或解决方案。
6. 保留否定关系，例如“没有到账”“无法连接”“没有重置”。
7. 保留事件顺序，例如“续费成功后流量没有重置”。
8. 保留Windows、MacOS、Clash Verge等平台和软件名称。
9. 第一条必须是最主要、最准确的标准检索问题。
10. 其他查询可以使用不同表达补充召回，但不能改变用户意图。
11. 不要生成意思完全相同的重复问题。
12. 不要因为出现“购买”“充值”等词，就擅自判断为续费。
13. 不要把“流量”“套餐”“订阅”“节点”等不同概念随意替换。

示例一：

原问题：
我又买了一个月，日期变了，但是剩下的流量还是没回来。

已知信息：
用户进行了续费；
套餐有效期已经延长；
流量没有恢复。

改写结果：
- 续费成功后套餐有效期已延长但流量没有重置。
- 续费后剩余流量未刷新怎么办。

示例二：

原问题：
Windows上的Clash Verge显示连接成功，但是网页打不开。

已知信息：
用户使用Windows；
用户使用Clash Verge；
客户端显示连接成功；
无法打开网页。

改写结果：
- Windows版Clash Verge连接成功后无法打开网页怎么办。
- Clash Verge显示已连接但无法访问网络。
"""

    input_data = {
        "resolved_query": resolved_query,
        "intent": intent,
        "known_information": known_information,
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