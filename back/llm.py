"""
该文件是统一的LLM配置文件,
它负责：
读取.env中的模型配置
→ 检查配置是否存在
→ 创建ChatOpenAI模型对象
→ 供graph.py等文件统一导入使用


这里主要是“创建模型对象”,真正调用LLM是在执行: model.invoke(...)或: model_with_tools.invoke(...)
"""


import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL_NAME = os.getenv("MODEL")


if not API_KEY:
    raise ValueError("没有找到环境变量 API_KEY")

if not BASE_URL:
    raise ValueError("没有找到环境变量 BASE_URL")

if not MODEL_NAME:
    raise ValueError("没有找到环境变量 MODEL")


model = ChatOpenAI(
    model=MODEL_NAME,
    api_key=API_KEY,
    base_url=BASE_URL,
)