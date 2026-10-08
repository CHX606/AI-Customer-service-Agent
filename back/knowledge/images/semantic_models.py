"""图片语义结果、可观察事实和指纹数据契约。"""
from dataclasses import dataclass
from typing import Literal
from langchain_core.documents import Document
from pydantic import BaseModel, Field, field_validator

MAX_CONTEXT_CHARS = 5000

def _clean_text(value: str, max_length: int) -> str:
    return "\n".join(
        line.strip()
        for line in str(value).replace("\x00", "").splitlines()
        if line.strip()
    )[:max_length]

def _clean_text_list(values: list[str], limit: int = 12) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = _clean_text(value, 500)
        key = item.casefold()
        if not item or key in seen:
            continue
        seen.add(key)
        cleaned.append(item)
        if len(cleaned) >= limit:
            break
    return cleaned

class KnowledgeImageSemanticCard(BaseModel):
    """知识库图片结合相邻正文后的稳定语义。"""

    should_index: bool = True
    image_type: Literal[
        "故障截图",
        "操作步骤",
        "设置示例",
        "公告或说明",
        "二维码或链接",
        "装饰或无关图片",
        "其他",
    ] = "其他"
    summary: str = Field(default="", max_length=600)
    context_role: str = Field(default="", max_length=600)
    visible_evidence: list[str] = Field(default_factory=list, max_length=12)
    problem_meaning: str = Field(default="", max_length=1200)
    recommended_action: str = Field(default="", max_length=1800)
    search_queries: list[str] = Field(default_factory=list, max_length=12)
    uncertainty: str = Field(default="", max_length=600)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator(
        "summary",
        "context_role",
        "problem_meaning",
        "recommended_action",
        "uncertainty",
        mode="before",
    )
    @classmethod
    def clean_strings(cls, value: object) -> str:
        return _clean_text(str(value or ""), 1800)

    @field_validator("visible_evidence", "search_queries", mode="before")
    @classmethod
    def clean_lists(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return _clean_text_list([str(item) for item in value])

class CustomerImageUnderstanding(BaseModel):
    """未知客户图片的可观察事实和检索表达，不直接生成业务答案。"""

    summary: str = Field(min_length=1, max_length=600)
    visible_evidence: list[str] = Field(default_factory=list, max_length=12)
    likely_problem: str = Field(default="", max_length=900)
    search_queries: list[str] = Field(default_factory=list, max_length=8)
    uncertainty: str = Field(default="", max_length=600)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator(
        "summary",
        "likely_problem",
        "uncertainty",
        mode="before",
    )
    @classmethod
    def clean_strings(cls, value: object) -> str:
        return _clean_text(str(value or ""), 900)

    @field_validator("visible_evidence", "search_queries", mode="before")
    @classmethod
    def clean_lists(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return _clean_text_list([str(item) for item in value], limit=8)

@dataclass(frozen=True)
class ImageFingerprint:
    sha256: str
    dhash: str
    detail_dhash: str
    width: int
    height: int

@dataclass(frozen=True)
class ImageSemanticMatch:
    strategy: Literal["exact_sha256", "perceptual_dhash"]
    distance: int
    documents: tuple[Document, ...]
