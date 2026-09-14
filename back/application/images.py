"""图片聊天用例：图像解析完成后复用聊天服务。"""
from back.application.chat import ChatService
from back.application.ports import ImageInterpreter
from back.domain.chat import ChatCommand, ChatResult
from back.domain.errors import InvalidRequest, NotFound

MAX_IMAGE_BYTES = 10 * 1024 * 1024


class ImageChatService:
    def __init__(self, chat: ChatService, interpreter: ImageInterpreter):
        self.chat = chat
        self.interpreter = interpreter

    def execute(self, content: bytes, message: str, session_id: str, tenant_id: str, *, turn_id: str | None = None) -> ChatResult:
        if not content or len(content) > MAX_IMAGE_BYTES:
            raise InvalidRequest("截图不能为空，且大小不能超过 10MB")
        if self.chat.tenants.get(tenant_id) is None:
            raise NotFound("未找到租户配置资料")
        description = self.interpreter.describe(content, message, tenant_id)
        return self.chat.execute(ChatCommand(description, session_id, tenant_id, turn_id), allow_semantic_cache=False)
