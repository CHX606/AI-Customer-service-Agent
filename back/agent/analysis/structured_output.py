"""客服请求分析职责模块：structured_output。"""
import json
from typing import Any, TypeVar
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


def _structured_message_text(raw: Any) -> str:
    """提取 Responses API 返回的文本内容，兼容字符串和内容块列表。"""
    content = getattr(raw, "content", raw)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return ""


def _validate_structured_text(
    schema: type[StructuredModel], raw: Any
) -> StructuredModel:
    text = _structured_message_text(raw)
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    return schema.model_validate(payload)


def _invoke_structured_output(
    llm: BaseChatModel,
    schema: type[StructuredModel],
    messages: list[SystemMessage | HumanMessage],
) -> StructuredModel:
    """优先使用原生结构化结果，并兼容部分模型返回文本 JSON。"""
    try:
        runner = llm.with_structured_output(schema, include_raw=True)
    except TypeError:
        # 离线假模型和部分旧版 provider 不接受 include_raw。
        result = llm.with_structured_output(schema).invoke(messages)
        if isinstance(result, schema):
            return result
        return schema.model_validate(result)

    result = runner.invoke(messages)
    if isinstance(result, schema):
        return result
    if isinstance(result, dict):
        parsed = result.get("parsed")
        if isinstance(parsed, schema):
            return parsed
        if parsed is not None:
            return schema.model_validate(parsed)
        return _validate_structured_text(schema, result.get("raw"))
    return _validate_structured_text(schema, result)
