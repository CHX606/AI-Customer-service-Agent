from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


# 1. 定义状态
class State(TypedDict):
    number: int


# 2. 定义节点
def add_one(state: State):
    return {
        "number": state["number"] + 1
    }


def multiply_two(state: State):
    return {
        "number": state["number"] * 2
    }


# 3. 创建图
builder = StateGraph(State)


# 4. 添加节点
builder.add_node(
    "add_one",
    add_one
)

builder.add_node(
    "multiply_two",
    multiply_two
)


# 5. 连接节点
builder.add_edge(
    START,
    "add_one"
)

builder.add_edge(
    "add_one",
    "multiply_two"
)

builder.add_edge(
    "multiply_two",
    END
)


# 6. 编译
graph = builder.compile()


# 7. 执行
result = graph.invoke(
    {
        "number": 10
    }
)


print(result)