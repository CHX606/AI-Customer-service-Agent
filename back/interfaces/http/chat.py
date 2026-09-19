"""聊天网关：认证、HTTP 数据转换和流式编码。"""
import json
from dataclasses import asdict
from typing import Annotated
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from back.bootstrap import get_chat_service, get_image_chat_service
from back.core.features import image_features_enabled
from back.application.images import MAX_IMAGE_BYTES
from back.domain.chat import ChatCommand as InternalChatRequest, PreparedChat
from back.domain.errors import NotFound
from back.interfaces.http.auth import verify_chat_tenant_authorization
from back.interfaces.http.schemas import ChatRequest, ChatResponse
from back.domain.tenant import validate_tenant_id

router = APIRouter(tags=["Chat"])


def require_image_chat():
    if not image_features_enabled():
        raise HTTPException(status_code=503, detail="图片问答暂未开放，请直接输入文字问题。")


def _command(request, token=None):
    if get_chat_service().tenants.get(request.tenant_id) is None:
        raise NotFound(f"未找到租户 '{request.tenant_id}' 的配置资料")
    tenant_id = verify_chat_tenant_authorization(request.tenant_id, token)
    return InternalChatRequest(request.message, request.session_id, tenant_id,
                               request.turn_id, request.regenerate, request.previous_answer_hash)


def _prepare_chat(request, x_tenant_token=None):
    return get_chat_service().prepare(_command(request, x_tenant_token))


def _execute_chat(request, x_tenant_token=None, *, allow_semantic_cache=True):
    result = get_chat_service().execute(_command(request, x_tenant_token), allow_semantic_cache=allow_semantic_cache)
    return ChatResponse(**asdict(result))


def _stream_line(event: dict[str, object]) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"


def _stream_prepared_chat(prepared):
    for event in get_chat_service().stream(prepared):
        yield _stream_line(event)


@router.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
    x_tenant_token: str | None = Header(default=None, alias="X-Tenant-Token"),
):
    """接收纯文字问题并返回 AI 客服回答。"""
    return _execute_chat(request, x_tenant_token=x_tenant_token)


@router.post("/chat/stream")
def chat_stream(
    request: ChatRequest,
    x_tenant_token: str | None = Header(default=None, alias="X-Tenant-Token"),
):
    """流式返回处理进度与最终回答。"""
    prepared = _prepare_chat(request, x_tenant_token=x_tenant_token)
    return StreamingResponse(
        _stream_prepared_chat(prepared),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/chat/image",
    response_model=ChatResponse,
    dependencies=[Depends(require_image_chat)],
)
def chat_with_image(
    session_id: Annotated[
        str,
        Form(min_length=1, max_length=100),
    ],
    image: Annotated[
        UploadFile,
        File(description="用户上传的故障截图"),
    ],
    message: Annotated[
        str,
        Form(max_length=2000),
    ] = "",
    tenant_id: Annotated[
        str,
        Form(description="租户ID"),
    ] = "default",
    turn_id: Annotated[str | None, Form(min_length=1, max_length=200)] = None,
    x_tenant_token: str | None = Header(default=None, alias="X-Tenant-Token"),
):
    """临时理解用户截图，结合知识库回答；不将客户图片自动写入知识库。"""
    try:
        safe_tenant_id = validate_tenant_id(tenant_id)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err
    verify_chat_tenant_authorization(safe_tenant_id, x_tenant_token)
    result = get_image_chat_service().execute(
        image.file.read(MAX_IMAGE_BYTES + 1), message, session_id, safe_tenant_id,
        turn_id=turn_id,
    )
    return ChatResponse(**asdict(result))
