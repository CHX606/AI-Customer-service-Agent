"""租户与企业配置数据模型。"""

import re
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field, field_validator


TENANT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


def validate_tenant_id(tenant_id: str) -> str:
    """校验 tenant_id 格式，只允许英文字母、数字、下划线和短横线。"""
    if not isinstance(tenant_id, str):
        raise ValueError("tenant_id 必须为字符串")
    if not tenant_id:
        raise ValueError("tenant_id 不能为空")
    if len(tenant_id) > 64:
        raise ValueError("tenant_id 长度不能超过 64 个字符")
    if not TENANT_ID_PATTERN.match(tenant_id):
        raise ValueError(
            "tenant_id 只允许包含英文字母、数字、下划线(_)和短横线(-)，不得包含空格或特殊字符"
        )
    return tenant_id


KnowledgeSourceStatus = Literal[
    "pending",
    "processing",
    "ready",
    "failed",
]


class TenantProfile(BaseModel):
    """企业完整配置信息。"""

    tenant_id: str = Field(description="租户唯一标识")
    company_name: str = Field(min_length=1, max_length=200, description="企业名称")
    brand_name_en: str | None = Field(
        default=None, max_length=100, description="企业英文品牌名"
    )
    assistant_name: str = Field(
        min_length=1, max_length=100, description="客服助理名称"
    )
    short_description: str = Field(
        max_length=4000, description="企业简介与产品定位"
    )
    business_scope: list[str] = Field(
        default_factory=list,
        max_length=50,
        description="业务与服务范围列表",
    )
    business_hours: str | None = Field(
        default=None, max_length=500, description="营业或客服工作时间"
    )
    public_contact: str | None = Field(
        default=None, max_length=1000, description="对外公开联系方式"
    )
    welcome_title: str = Field(
        min_length=1, max_length=200, description="聊天欢迎语主标题"
    )
    welcome_description: str = Field(
        min_length=1, max_length=2000, description="聊天欢迎语详细描述"
    )
    tone: str = Field(
        default="简洁、友好、专业",
        min_length=1,
        max_length=200,
        description="客服回复语气风格",
    )
    handoff_message: str = Field(
        default="抱歉，知识库暂未查到明确资料，建议联系人工客服核实。",
        min_length=1,
        max_length=2000,
        description="转人工提示语",
    )
    suggested_questions: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="前端展示的推荐问题列表",
    )
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @field_validator("tenant_id")
    @classmethod
    def check_tenant_id(cls, value: str) -> str:
        return validate_tenant_id(value)

    @field_validator("business_scope", "suggested_questions")
    @classmethod
    def validate_text_list(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            item = value.strip()
            if not item:
                continue
            if len(item) > 500:
                raise ValueError("列表中的单项内容不能超过 500 个字符")
            cleaned.append(item)
        return cleaned


class TenantPublicProfile(BaseModel):
    """对外公开的企业资料（不包含内部运维配置）。"""

    tenant_id: str
    company_name: str
    brand_name_en: str | None = None
    assistant_name: str
    short_description: str
    business_scope: list[str] = Field(default_factory=list)
    business_hours: str | None = None
    public_contact: str | None = None
    welcome_title: str
    welcome_description: str
    tone: str = "简洁、友好、专业"
    handoff_message: str
    suggested_questions: list[str] = Field(default_factory=list)
    updated_at: datetime


class TenantProfileUpdate(BaseModel):
    """企业配置更新请求体。"""

    tenant_id: str = "default"
    company_name: str | None = Field(default=None, min_length=1, max_length=200)
    brand_name_en: str | None = Field(default=None, max_length=100)
    assistant_name: str | None = Field(default=None, min_length=1, max_length=100)
    short_description: str | None = Field(default=None, max_length=4000)
    business_scope: list[str] | None = Field(default=None, max_length=50)
    business_hours: str | None = Field(default=None, max_length=500)
    public_contact: str | None = Field(default=None, max_length=1000)
    welcome_title: str | None = Field(default=None, min_length=1, max_length=200)
    welcome_description: str | None = Field(
        default=None, min_length=1, max_length=2000
    )
    tone: str | None = Field(default=None, min_length=1, max_length=200)
    handoff_message: str | None = Field(default=None, min_length=1, max_length=2000)
    suggested_questions: list[str] | None = Field(default=None, max_length=20)

    @field_validator("tenant_id")
    @classmethod
    def check_tenant_id(cls, value: str) -> str:
        return validate_tenant_id(value)

    @field_validator(
        "company_name",
        "assistant_name",
        "short_description",
        "welcome_title",
        "welcome_description",
        "tone",
        "handoff_message",
        mode="before",
    )
    @classmethod
    def reject_null_required_fields(cls, value: str | None) -> str:
        if value is None:
            raise ValueError("该字段不能设置为 null")
        return value

    @field_validator("business_scope", "suggested_questions")
    @classmethod
    def validate_optional_text_list(
        cls, values: list[str] | None
    ) -> list[str] | None:
        if values is None:
            return None
        return TenantProfile.validate_text_list(values)


class KnowledgeSource(BaseModel):
    """知识库文档数据源元数据。"""

    source_id: str = Field(description="数据源唯一ID")
    tenant_id: str = Field(description="所属租户ID")
    original_filename: str = Field(description="上传时的原始文件名")
    stored_filename: str = Field(description="服务器存储的文件名")
    file_type: str = Field(description="文件类型扩展名，如 docx/pdf/txt/md")
    content_hash: str = Field(description="文件内容 SHA-256 哈希")
    status: KnowledgeSourceStatus = Field(
        default="pending", description="知识库索引状态"
    )
    error_message: str | None = Field(
        default=None, description="处理失败时的错误原因"
    )
    chunk_count: int = Field(default=0, description="切块分片数量")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @field_validator("tenant_id")
    @classmethod
    def check_tenant_id(cls, value: str) -> str:
        return validate_tenant_id(value)
