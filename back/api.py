"""
该文件后端API接口文件，负责连接前端和Agent。

前端POST /chat 或 /chat/image
→ 接收文字，或者文字与故障截图
→ 读取该会话历史状态
→ 调用LangGraph Agent
→ 保存更新后的状态
→ 将最终回答返回前端

另外还负责：
/health：检查后端是否正常。
CORS：允许前端地址访问后端。
session_states：临时保存不同用户的会话状态。
"""


from dataclasses import dataclass
from typing import Annotated

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage

from back.agent.graph import customer_service_graph
from back.agent.state import CustomerServiceState
from back.schemas import ChatRequest, ChatResponse
from back.rag.user_image_query import (
    MAX_UPLOAD_BYTES,
    UserImageValidationError,
    analyze_user_image,
    build_image_agent_message,
)


app = FastAPI(
    title="AI Customer Service Agent",
    version="0.1.0",
)


FRONTEND_ORIGINS = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


session_states: dict[str, CustomerServiceState] = {}


@dataclass(frozen=True)
class InternalChatRequest:
    """图片 OCR 完成后调用 Agent 使用的内部请求。"""

    message: str
    session_id: str


@app.get("/health")
def health_check():
    """检查后端服务是否正常运行。"""

    return {
        "status": "ok",
    }


def _execute_chat(
    request: ChatRequest | InternalChatRequest,
) -> ChatResponse:
    """运行文字和图片请求共用的 LangGraph 会话流程。"""

    previous_state = session_states.get(
        request.session_id,
        {},
    )

    history = previous_state.get(
        "messages",
        [],
    )

    active_issue = previous_state.get(
        "active_issue",
    )

    result = customer_service_graph.invoke(
        {
            "messages": [
                *history,
                HumanMessage(content=request.message),
            ],
            "active_issue": active_issue,
        }
    )

    updated_state: CustomerServiceState = {
        "messages": result["messages"],
        "active_issue": result.get("active_issue"),
        "relation": result.get("relation"),
        "resolved_query": result.get("resolved_query"),
        "context_reason": result.get("context_reason"),
        "scope": result.get("scope"),
        "scope_reason": result.get("scope_reason"),
        "intent": result.get("intent"),
        "action": result.get("action"),
        "known_information": result.get("known_information", []),
        "missing_information": result.get("missing_information", []),
        "clarifying_question": result.get("clarifying_question"),
        "intent_reason": result.get("intent_reason"),
        "rewritten_queries": result.get("rewritten_queries", []),
        "rewrite_reason": result.get("rewrite_reason"),
        "search_queries": result.get("search_queries", []),
        "retrieval_candidates": result.get("retrieval_candidates", []),
        "retrieved_documents": result.get("retrieved_documents", []),
        "evidence_status": result.get("evidence_status"),
        "supporting_document_indexes": result.get(
            "supporting_document_indexes",
            [],
        ),
        "supporting_documents": result.get("supporting_documents", []),
        "evidence_missing_information": result.get(
            "evidence_missing_information",
            [],
        ),
        "evidence_clarifying_question": result.get(
            "evidence_clarifying_question"
        ),
        "evidence_reason": result.get("evidence_reason"),
    }

    session_states[request.session_id] = updated_state

    print(
        {
            "session_id": request.session_id,
            "relation": result.get("relation"),
            "resolved_query": result.get("resolved_query"),
            "active_issue": result.get("active_issue"),
            "context_reason": result.get("context_reason"),
            "scope": result.get("scope"),
            "scope_reason": result.get("scope_reason"),
            "intent": result.get("intent"),
            "action": result.get("action"),
            "known_information": result.get("known_information", []),
            "missing_information": result.get("missing_information", []),
            "clarifying_question": result.get("clarifying_question"),
            "intent_reason": result.get("intent_reason"),
            "rewritten_queries": result.get("rewritten_queries", []),
            "rewrite_reason": result.get("rewrite_reason"),
            "search_queries": result.get("search_queries", []),
            "retrieval_candidate_count": len(
                result.get("retrieval_candidates", [])
            ),
            "rrf_top_sections": [
                {
                    "section_id": document.metadata.get("section_id"),
                    "section_title": document.metadata.get("section_title"),
                    "rrf_rank": document.metadata.get("rrf_rank"),
                    "rrf_score": document.metadata.get("rrf_score"),
                }
                for document in result.get(
                    "retrieval_candidates",
                    [],
                )[:10]
            ],
            "retrieved_document_count": len(
                result.get("retrieved_documents", [])
            ),
            "retrieved_sections": [
                {
                    "section_id": document.metadata.get("section_id"),
                    "section_title": document.metadata.get("section_title"),
                    "rrf_rank": document.metadata.get("rrf_rank"),
                    "rrf_score": document.metadata.get("rrf_score"),
                    "reranker_rank": document.metadata.get(
                        "reranker_rank"
                    ),
                    "reranker_score": document.metadata.get(
                        "reranker_score"
                    ),
                    "matched_queries": document.metadata.get(
                        "matched_queries",
                        [],
                    ),
                }
                for document in result.get("retrieved_documents", [])
            ],
            "evidence_status": result.get("evidence_status"),
            "supporting_document_indexes": result.get(
                "supporting_document_indexes",
                [],
            ),
            "evidence_missing_information": result.get(
                "evidence_missing_information",
                [],
            ),
            "evidence_clarifying_question": result.get(
                "evidence_clarifying_question"
            ),
            "evidence_reason": result.get("evidence_reason"),
        }
    )

    final_answer = result["messages"][-1].content

    return ChatResponse(
        answer=final_answer,
        session_id=request.session_id,
    )


@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):
    """接收纯文字问题并返回 AI 客服回答。"""

    return _execute_chat(request)


@app.post(
    "/chat/image",
    response_model=ChatResponse,
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
):
    """
    临时识别用户截图，再把 OCR 文字作为本次查询交给 Agent。

    图片和中间结果只存在于临时目录，不写入知识库数据库。
    """

    image_bytes = image.file.read(
        MAX_UPLOAD_BYTES + 1
    )

    try:
        analysis = analyze_user_image(image_bytes)
    except UserImageValidationError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error
    except RuntimeError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    agent_message = build_image_agent_message(
        user_message=message,
        analysis=analysis,
    )

    print(
        {
            "session_id": session_id,
            "input_type": "image",
            "image_filename": image.filename,
            "image_size": (
                f"{analysis.width}x{analysis.height}"
            ),
            "image_layout": analysis.layout,
            "image_region_count": analysis.region_count,
            "image_ocr_length": len(analysis.ocr_text),
        }
    )

    return _execute_chat(
        InternalChatRequest(
            message=agent_message,
            session_id=session_id,
        )
    )
