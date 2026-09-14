"""
此文件为对话上下文边界判断，由于AI客服所面对的用户可能多次咨询，且咨询内容和上次咨询内容关系需要区分，
因此该文件为完善上下文的边界判断，判断和上次咨询的内容关系，当前关系有：
continue：继续当前问题。
new_issue：开始新的问题。
correction：修正之前的信息。
uncertain：暂时无法判断新旧问题关系。
"""

import json
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from back.agent.workflow.lifecycle import decide_relation_with_lifecycle
from back.domain.conversation import ActiveIssue


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

    is_self_contained: bool = Field(
        description="只看当前消息，是否已经能独立理解为一个完整问题。"
    )

    references_active_issue: bool = Field(
        description=(
            "当前消息是否通过这个、它、刚才的问题、还是不行等表达，"
            "明确指向正在处理的问题。"
        )
    )

    answers_last_question: bool = Field(
        description="当前消息是否在直接回答助手上一轮提出的问题。"
    )

    explicit_new_issue: bool = Field(
        description=(
            "当前消息是否使用另外、还有一个问题、换个问题等表达，"
            "明确开始新问题。"
        )
    )


def analyze_context(
    llm: BaseChatModel,
    current_query: str,
    recent_messages: list[dict[str, str]],
    active_issue: ActiveIssue | None,
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
- is_self_contained 只能根据 current_query 本身判断，不能把旧问题补进去后再判断。
- references_active_issue 表示当前消息确实指向旧问题；仅仅业务类别相近不算指向。
- answers_last_question 只在当前消息确实回答了助手最后一个问题时为 true。
- explicit_new_issue 只在用户明确表达切换问题时为 true。
- active_issue.status 为 awaiting_user 时，直接回答上一轮追问通常属于 continue。
- active_issue.status 为 answered 时，用户仍可能继续追问、确认或改写同一问题。
  如果当前消息是上一问题的同义改写、确认性追问或更具体的追问，应判断为 continue。
- active_issue.status 为 resolved 或 handed_off 时，新的完整问题默认属于 new_issue；
  但如果用户明确指向之前的问题，仍可判断为 continue。
- 不要把助手提出的问题当成用户已经确认的事实。
- 不要编造用户没有提供的信息。
- 如果是 new_issue，resolved_query 不能混入旧问题。
- 如果是 continue 或 correction，resolved_query 应合并当前问题和用户新提供的信息。
- 如果是 uncertain，resolved_query 应保留用户原意，不要强行补充无法确定的信息。
- resolved_query 只整理问题，不要回答问题。
- decision_reason 只需要简短说明判断依据，不要回答客服问题。

示例：

上一问题：“续费以后流量为什么没有重置？”
当前消息：“续费之后不会有流量吗？”
二者是同一规则的确认性追问，应判断为 continue。
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

    final_relation = decide_relation_with_lifecycle(
        model_relation=result.relation,
        active_issue=active_issue,
        is_self_contained=result.is_self_contained,
        references_active_issue=result.references_active_issue,
        answers_last_question=result.answers_last_question,
        explicit_new_issue=result.explicit_new_issue,
    )

    if final_relation != result.relation:
        final_resolved_query = result.resolved_query

        if final_relation == "new_issue":
            final_resolved_query = current_query.strip()

        result = result.model_copy(
            update={
                "relation": final_relation,
                "resolved_query": final_resolved_query,
                "decision_reason": (
                    f"{result.decision_reason} "
                    "系统结合当前问题生命周期修正了上下文关系。"
                ),
            }
        )

    return result
