"""
这是前后端聊天接口的数据格式定义。
ChatRequest
规定前端发送：
message
session_id
并限制字符长度。


ChatResponse
规定后端返回：
answer
session_id

"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """聊天接口的请求数据。"""

    message: str = Field(
        min_length=1,
        max_length=2000,
    )

    session_id: str = Field(
        min_length=1,
        max_length=100,
    )


class ChatResponse(BaseModel):
    """聊天接口的响应数据。"""

    answer: str
    session_id: str