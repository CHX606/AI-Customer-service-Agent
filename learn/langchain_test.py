#下载
#pip install langchain
#pip install langchain-openai


import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


load_dotenv()


llm = ChatOpenAI(
    model=os.getenv("MODEL"),
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)

messages = [
    SystemMessage(content = "你是一名专业Python老师，请用简单易懂的方式回答。"),
    HumanMessage(content = "你好，请介绍一下自己")
]


response = llm.invoke(messages)


print(response.content)