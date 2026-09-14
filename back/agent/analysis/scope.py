"""业务范围判断模块。

本模块负责判断整理后的用户问题是否属于客服业务范围。

判断结果包括：
- in_scope：属于企业客服业务（含企业介绍、服务范围、营业时间、联系方式与具体业务咨询）。
- chitchat：简单问候、感谢或告别。
- out_of_scope：明确与企业业务无关。
- uncertain：信息不足，暂时无法判断。
"""

import re
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


class ScopeAnalysis(BaseModel):
    scope: Literal[
        "in_scope",
        "chitchat",
        "out_of_scope",
        "uncertain",
    ] = Field(description="用户问题所属的业务范围类型。")

    scope_reason: str = Field(
        description="简短说明范围判断依据，不超过两句话。"
    )


_BUSINESS_HINTS = {
    "账号",
    "登录",
    "密码",
    "套餐",
    "订单",
    "支付",
    "续费",
    "退款",
    "流量",
    "订阅",
    "节点",
    "网络",
    "连接",
    "客户端",
    "软件",
    "安装",
    "配置",
    "企业",
    "公司",
    "营业时间",
    "联系",
}

_UNDERSPECIFIED_QUERIES = {
    "不行",
    "还是不行",
    "怎么还是不行",
    "不能用",
    "还是不能用",
    "用不了",
    "有问题",
    "出问题了",
    "怎么办",
    "怎么弄",
}


def _normalize_query(text: str) -> str:
    """移除空白与常见标点，便于执行稳定的短句规则。"""
    return re.sub(r"[\s，。！？、,.!?；;：:]+", "", text)


def _is_underspecified_query(text: str) -> bool:
    """识别没有业务对象、没有故障现象的独立模糊问题。"""
    normalized = _normalize_query(text)
    if any(hint in normalized for hint in _BUSINESS_HINTS):
        return False
    return normalized in _UNDERSPECIFIED_QUERIES


def analyze_scope(
    llm: BaseChatModel,
    resolved_query: str,
    company_name: str = "当前企业",
) -> ScopeAnalysis:
    """判断完整用户问题所属的业务范围。"""

    # 这类短句交给模型容易在 in_scope/uncertain 之间漂移。只有上下文
    # 节点补全了业务对象后才应继续进入业务流程，否则稳定地要求用户补充。
    if _is_underspecified_query(resolved_query):
        return ScopeAnalysis(
            scope="uncertain",
            scope_reason="问题未包含具体业务对象或可判断的故障现象，需要补充信息。",
        )

    structured_llm = llm.with_structured_output(ScopeAnalysis)

    system_prompt = f"""
你是{company_name}AI客服的业务范围判断节点。

你只负责判断问题属于哪一种范围，不要回答用户问题。

范围包括：

1. in_scope
- 企业基本资料咨询：例如“你们是谁”“这是做什么的”“介绍一下企业/产品”“营业时间是什么”“怎么联系你们/联系方式”“人工客服在吗”等，均属于 in_scope。
- 与企业产品、服务、功能或操作有关的问题，例如账号、套餐、购买、支付、订单、续费、退款、流量、订阅、节点、网络连接、客户端、软件安装与配置。

2. chitchat
问候、感谢、告别等简单对话，例如“你好”“谢谢”“再见”。

3. out_of_scope
明确与企业产品、服务或企业资料无关的问题，例如天气、新闻、做饭、数学、通用代码编写等。

4. uncertain
信息太少，无法判断是否属于客服业务。

判断规则：
- “你们是谁”“这是干啥的”“营业时间”“联系电话/邮箱”等企业资料咨询问题必须判定为 in_scope。
- 业务问题即使知识库可能没有答案，也属于 in_scope。
- 已说明账号、套餐、续费、节点、客户端等业务对象的模糊问题可以判定为 in_scope。
- 完全没有业务对象和故障现象的短句（如“怎么还是不行”“有问题”）必须判定为 uncertain。
- 不要根据知识库是否能检索到资料判断业务范围。
- 不要回答问题。
- scope_reason只说明判断依据，不超过两句话。
"""

    result = structured_llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"待判断的问题：{resolved_query}"),
        ]
    )

    return result
