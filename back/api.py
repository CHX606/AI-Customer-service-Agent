"""
该文件后端API接口文件，负责连接前端和Agent。

前端POST /chat
→ 接收message和session_id
→ 读取该会话历史状态
→ 调用LangGraph Agent
→ 保存更新后的状态
→ 将最终回答返回前端

另外还负责：
/health：检查后端是否正常。
CORS：允许前端地址访问后端。
session_states：临时保存不同用户的会话状态。
"""


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage

from back.agent.graph import customer_service_graph
from back.agent.state import CustomerServiceState
from back.schemas import ChatRequest, ChatResponse


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


@app.get("/health")
def health_check():
    """检查后端服务是否正常运行。"""

    return {
        "status": "ok",
    }


@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):
    """接收用户问题并返回 AI 客服回答。"""

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
    }

    session_states[request.session_id] = updated_state

    print(
        {
            "session_id": request.session_id,
            "relation": result.get("relation"),
            "resolved_query": result.get("resolved_query"),
            "active_issue": result.get("active_issue"),
            "context_reason": result.get("context_reason"),
        }
    )

    final_answer = result["messages"][-1].content

    return ChatResponse(
        answer=final_answer,
        session_id=request.session_id,
    )