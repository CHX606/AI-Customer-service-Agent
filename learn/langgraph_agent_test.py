import os

from dotenv import load_dotenv

from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition


load_dotenv()


# ======================
# 1. 创建模型
# ======================

llm = ChatOpenAI(
    model=os.getenv("MODEL"),
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)


# ======================
# 2. 创建工具
# ======================

@tool
def get_weather(city: str):
    """
    查询指定城市的天气
    """
    return f"{city}今天晴天，温度25℃"



tools = [
    get_weather
]


# ======================
# 3. 定义State
# ======================

class State(TypedDict):
    messages: list


# ======================
# 4. 绑定工具
# ======================

llm_with_tools = llm.bind_tools(
    tools
)


# ======================
# 5. Agent节点
# ======================

def call_model(state: State):

    response = llm_with_tools.invoke(
        state["messages"]
    )

    return {
        "messages": [
            response
        ]
    }


# ======================
# 6. Tool节点
# ======================

tool_node = ToolNode(
    tools
)


# ======================
# 7. 创建Graph
# ======================

builder = StateGraph(State)


# 添加节点

builder.add_node(
    "agent",
    call_model
)


builder.add_node(
    "tools",
    tool_node
)



# ======================
# 8. 连接边
# ======================

builder.add_edge(
    START,
    "agent"
)


builder.add_conditional_edges(
    "agent",
    tools_condition
)


builder.add_edge(
    "tools",
    "agent"
)


# ======================
# 9. 编译
# ======================

graph = builder.compile()



# ======================
# 10. 运行
# ======================

result = graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "广州天气怎么样？"
            }
        ]
    }
)


print(
    result["messages"][-1].content
)