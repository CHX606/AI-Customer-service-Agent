"""
此文件为对话上下文边界判断，由于AI客服所面对的用户可能多次咨询，且咨询内容和上次咨询内容关系需要区分，
因此该文件为完善上下文的边界判断，判断和上次咨询的内容关系，当前关系有：
continue：继续当前问题。
new_issue：开始新的问题。
correction：修正之前的信息。
uncertain：暂时无法判断新旧问题关系。
"""

import json
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


class ContextAnalysis(BaseModel):
    relation: Literal[
        "continue",
        "new_issue",
        "correction",
        "uncertain",
    ] = Field(
        description=(
            "当前消息与正在处理的问题之间的关系："
            "continue 表示继续当前问题；"
            "new_issue 表示开始新问题；"
            "correction 表示修正之前的信息；"
            "uncertain 表示无法确定。"
        )
    )

    resolved_query: str = Field(
        description=(
            "结合有效上下文后得到的完整、可独立理解的用户问题。"
            "如果是新问题，则只整理当前新问题，不能混入旧问题。"
        )
    )

    decision_reason: str = Field(
        description="简短说明为什么判断为这种关系，不超过两句话。"
    )


def analyze_context(
    llm: BaseChatModel,
    current_query: str,
    recent_messages: list[dict[str, str]],
    active_issue: dict[str, Any] | None,
) -> ContextAnalysis:
    """分析用户当前消息与正在处理的问题之间的关系。"""

    structured_llm = llm.with_structured_output(ContextAnalysis)

    system_prompt = """
你是客服系统中的上下文分析节点。

你的任务不是回答用户问题，而是判断用户当前消息与正在处理的问题之间的关系，
并生成一个脱离聊天记录也能独立理解的完整问题。

关系只能是以下四种之一：

1. continue
当前消息是在回答上一轮问题、补充信息，或者继续描述当前问题。

2. new_issue
当前消息表达了一个可以独立理解的新问题，并且与当前正在处理的问题不同。

3. correction
当前消息明确否定或修正了之前的信息。

4. uncertain
根据现有信息无法可靠判断当前消息属于旧问题还是新问题。

判断规则：

- 如果没有正在处理的问题，判断为 new_issue。
- “是的”“没有”“已经到账了”“还是不行”等不能独立理解的回答，
  通常属于 continue。
- 使用“这个”“它”“刚才的”等指代当前问题的消息，通常属于 continue。
- “另外”“还有一个问题”“换个问题”等表达通常表示 new_issue。
- “不是续费，是第一次购买”等明确推翻旧信息的表达属于 correction。
- 不要把助手提出的问题当成用户已经确认的事实。
- 不要编造用户没有提供的信息。
- 如果是 new_issue，resolved_query 不能混入旧问题。
- 如果是 continue 或 correction，resolved_query 应合并当前问题和用户新提供的信息。
- 如果是 uncertain，resolved_query 应保留用户原意，不要强行补充无法确定的信息。
- resolved_query 只整理问题，不要回答问题。
- decision_reason 只需要简短说明判断依据，不要回答客服问题。
"""

    input_data = {
        "active_issue": active_issue,
        "recent_messages": recent_messages,
        "current_query": current_query,
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