import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("MODEL"),
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)


question = input("你: ")

prompt = PromptTemplate.from_template(
    """
你是一名专业Python老师。

请回答用户的问题：

{question}
"""
)

chain = prompt | llm

response = chain.invoke(
    {
        "question":question
    }
)

print("AI: ",response.content)
