"""前后端接口数据格式定义。"""

from pydantic import BaseModel, Field, field_validator
from back.domain.tenant import validate_tenant_id


class ChatRequest(BaseModel):
    """聊天接口请求数据。"""

    tenant_id: str = Field(
        default="default",
        description="租户标识",
    )
    message: str = Field(
        min_length=1,
        max_length=2000,
        description="用户发送的问题内容",
    )
    session_id: str = Field(
        min_length=1,
        max_length=100,
        description="会话唯一标识",
    )
    turn_id: str | None = Field(default=None, min_length=1, max_length=200)
    regenerate: bool = False
    previous_answer_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @field_validator("tenant_id")
    @classmethod
    def check_tenant_id(cls, value: str) -> str:
        return validate_tenant_id(value)


class ChatResponse(BaseModel):
    """聊天接口响应数据。"""

    answer: str = Field(description="客服回答内容")
    session_id: str = Field(description="会话唯一标识")
    tenant_id: str = Field(default="default", description="租户标识")
