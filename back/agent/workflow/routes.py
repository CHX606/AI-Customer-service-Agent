"""客服工作流职责模块：routes"""
from back.agent.workflow.state import CustomerServiceState

def route_after_request_analysis(state: CustomerServiceState) -> str:
    """根据合并分析结果直接选择后续分支。"""
    scope = state.get("scope")
    if scope == "chitchat":
        return "chitchat"
    if scope == "out_of_scope":
        return "out_of_scope"
    if scope == "uncertain":
        return "scope_uncertain"

    action = state.get("action")
    if action == "profile":
        return "answer_profile"
    if action == "retrieve":
        return "hybrid_search"
    return "clarify"
