"""
LG做流程图
"""


from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from back.agent.query_analyzer import analyze_context
from back.agent.state import CustomerServiceState
from back.agent.tools import search_knowledge_base
from back.llm import model


SYSTEM_PROMPT = """
你是可乐云的 AI 客服助手。

请遵守以下规则：

1. 用户询问账号、套餐、续费、流量、节点、订单、软件使用
   或其他可乐云业务问题时，必须先调用 search_knowledge_base 工具。
2. 回答业务问题时，只能使用知识库工具返回的资料。
3. 不要编造知识库中不存在的规则、价格或处理方式。
4. 如果知识库没有明确答案，请说明暂未查到，并建议联系人工客服。
5. 使用简洁、友好、容易理解的中文回答。
6. 普通问候或与业务无关的简单对话，可以直接回答。
"""


TOOLS = [
    search_knowledge_base,
]


model_with_tools = model.bind_tools(TOOLS)


def build_recent_messages(
    messages: list[BaseMessage],
    limit: int = 6,
) -> list[dict[str, str]]:
    """把 LangChain 消息转换成上下文分析节点需要的简单格式。"""

    recent_messages = []

    for message in messages:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue

        if not message.content:
            continue

        recent_messages.append(
            {
                "role": role,
                "content": str(message.content),
            }
        )

    return recent_messages[-limit:]


def analyze_context_node(
    state: CustomerServiceState,
):
    """判断当前用户消息与正在处理的问题之间的关系。"""

    messages = state["messages"]

    current_user_index = None

    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            current_user_index = index
            break

    if current_user_index is None:
        raise ValueError("消息列表中没有找到用户消息")

    current_query = str(
        messages[current_user_index].content
    )

    previous_messages = messages[:current_user_index]

    recent_messages = build_recent_messages(
        previous_messages,
    )

    active_issue = state.get("active_issue")

    analysis = analyze_context(
        llm=model,
        current_query=current_query,
        recent_messages=recent_messages,
        active_issue=active_issue,
    )

    if analysis.relation in {
        "continue",
        "new_issue",
        "correction",
    }:
        updated_active_issue = {
            "summary": analysis.resolved_query,
        }
    else:
        updated_active_issue = active_issue

    return {
        "active_issue": updated_active_issue,
        "relation": analysis.relation,
        "resolved_query": analysis.resolved_query,
        "context_reason": analysis.decision_reason,
    }


def call_model(
    state: CustomerServiceState,
):
    """调用大模型，并让模型判断是否需要使用工具。"""

    resolved_query = state.get("resolved_query")
    relation = state.get("relation")

    context_prompt = f"""
上下文分析结果：

当前需要处理的完整问题：
{resolved_query}

当前消息与上一问题的关系：
{relation}

请根据完整问题理解用户本轮意图。
这段上下文分析只用于理解问题，不能作为业务知识或回答依据。
"""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=context_prompt),
        *state["messages"],
    ]

    response = model_with_tools.invoke(messages)

    return {
        "messages": [response],
    }


graph_builder = StateGraph(CustomerServiceState)

graph_builder.add_node(
    "analyze_context",
    analyze_context_node,
)

graph_builder.add_node(
    "agent",
    call_model,
)

graph_builder.add_node(
    "tools",
    ToolNode(TOOLS),
)

graph_builder.add_edge(
    START,
    "analyze_context",
)

graph_builder.add_edge(
    "analyze_context",
    "agent",
)

graph_builder.add_conditional_edges(
    "agent",
    tools_condition,
)

graph_builder.add_edge(
    "tools",
    "agent",
)


customer_service_graph = graph_builder.compile()


if __name__ == "__main__":
    question = "续费之后为什么流量没有重置？"

    result = customer_service_graph.invoke(
        {
            "messages": [
                HumanMessage(content=question),
            ],
        }
    )

    for message in result["messages"]:
        message.pretty_print()