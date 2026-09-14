"""
该文件是统一的LLM配置文件,
它负责：
读取.env中的模型配置
→ 检查配置是否存在
→ 创建ChatOpenAI模型对象
→ 供graph.py等文件统一导入使用


这里主要是“创建模型对象”,真正调用LLM是在执行: model.invoke(...)或: model_with_tools.invoke(...)
"""


from functools import lru_cache
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL_NAME = os.getenv("MODEL")
ROUTER_MODEL_NAME = os.getenv("ROUTER_MODEL") or MODEL_NAME
CONTEXT_MODEL_NAME = os.getenv("CONTEXT_MODEL") or MODEL_NAME
RESPONSE_MODEL_NAME = os.getenv("RESPONSE_MODEL") or MODEL_NAME
ROUTER_TIMEOUT_SECONDS = float(os.getenv("ROUTER_TIMEOUT_SECONDS", "5"))
ROUTER_MAX_RETRIES = int(os.getenv("ROUTER_MAX_RETRIES", "0"))
CONTEXT_TIMEOUT_SECONDS = float(os.getenv("CONTEXT_TIMEOUT_SECONDS", "15"))
CONTEXT_MAX_RETRIES = int(os.getenv("CONTEXT_MAX_RETRIES", "2"))
RESPONSE_TIMEOUT_SECONDS = float(os.getenv("RESPONSE_TIMEOUT_SECONDS", "45"))
RESPONSE_MAX_RETRIES = int(os.getenv("RESPONSE_MAX_RETRIES", "1"))
RESPONSE_REASONING_EFFORT = os.getenv(
    "RESPONSE_REASONING_EFFORT",
    "low",
).strip()
IMAGE_UNDERSTANDING_MODEL_NAME = os.getenv("IMAGE_UNDERSTANDING_MODEL") or MODEL_NAME


@lru_cache(maxsize=1)
def get_default_model() -> ChatOpenAI:
    if not API_KEY:
        raise ValueError("没有找到环境变量 API_KEY")

    if not BASE_URL:
        raise ValueError("没有找到环境变量 BASE_URL")

    if not MODEL_NAME:
        raise ValueError("没有找到环境变量 MODEL")


    return ChatOpenAI(
        model=MODEL_NAME, api_key=API_KEY, base_url=BASE_URL, temperature=0,
        timeout=float(os.getenv("MODEL_TIMEOUT_SECONDS", "45")),
        max_retries=int(os.getenv("MODEL_MAX_RETRIES", "1")),
    )


class LazyChatModel:
    """请求真正使用模型时才创建客户端，配置故障不阻断管理接口。"""
    def __getattr__(self, name):
        return getattr(get_default_model(), name)


model = LazyChatModel()


@lru_cache(maxsize=1)
def get_router_model() -> ChatOpenAI:
    """创建低输出、短超时的独立路由模型客户端。"""

    if not ROUTER_MODEL_NAME:
        raise ValueError("没有找到环境变量 ROUTER_MODEL 或 MODEL")

    return ChatOpenAI(
        model=ROUTER_MODEL_NAME,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0,
        timeout=ROUTER_TIMEOUT_SECONDS,
        max_retries=ROUTER_MAX_RETRIES,
    )


@lru_cache(maxsize=1)
def get_context_model() -> ChatOpenAI:
    """创建复杂多轮上下文分析模型客户端。"""

    if not CONTEXT_MODEL_NAME:
        raise ValueError("没有找到环境变量 CONTEXT_MODEL 或 MODEL")

    return ChatOpenAI(
        model=CONTEXT_MODEL_NAME,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0,
        timeout=CONTEXT_TIMEOUT_SECONDS,
        max_retries=CONTEXT_MAX_RETRIES,
    )


@lru_cache(maxsize=1)
def get_response_model() -> ChatOpenAI:
    """创建证据判断与最终客服回答模型客户端。"""

    if not RESPONSE_MODEL_NAME:
        raise ValueError("没有找到环境变量 RESPONSE_MODEL 或 MODEL")

    model_kwargs = {}
    if RESPONSE_REASONING_EFFORT:
        model_kwargs["reasoning_effort"] = RESPONSE_REASONING_EFFORT
    return ChatOpenAI(
        model=RESPONSE_MODEL_NAME,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0,
        timeout=RESPONSE_TIMEOUT_SECONDS,
        max_retries=RESPONSE_MAX_RETRIES,
        **model_kwargs,
    )


@lru_cache(maxsize=1)
def get_image_understanding_model() -> ChatOpenAI:
    """创建独立的强视觉模型，避免图片预处理受聊天小模型限制。"""

    if not IMAGE_UNDERSTANDING_MODEL_NAME:
        raise ValueError("没有找到环境变量 IMAGE_UNDERSTANDING_MODEL")

    return ChatOpenAI(
        model=IMAGE_UNDERSTANDING_MODEL_NAME,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0,
        timeout=45,
        max_retries=1,
    )
