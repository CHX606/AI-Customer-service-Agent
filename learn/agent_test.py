import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


load_dotenv()


llm = ChatOpenAI(
    model=os.getenv("MODEL"),
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。"""
    return f"{city}今天晴天，温度25℃"


@tool
def add_numbers(a: float, b: float) -> float:
    """计算两个数字相加。"""
    return a + b


# 创建短期记忆
memory = InMemorySaver()


# 创建Agent
agent = create_agent(
    model=llm,
    tools=[
        get_weather,
        add_numbers
    ],
    checkpointer=memory,
    system_prompt="你是一个有帮助的AI助手。"
)


# 同一个thread_id表示同一段对话
config = {
    "configurable": {
        "thread_id": "user_1"
    }
}


while True:

    question = input("你：")

    if question == "退出":
        break


    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        },
        config=config
    )


    print(
        "AI：",
        result["messages"][-1].content
    )