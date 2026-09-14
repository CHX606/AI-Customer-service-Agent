"""当前客服流程：仅负责注册节点与连线。"""
from langgraph.graph import END, START, StateGraph
from back.agent.workflow.state import CustomerServiceState
from back.agent.workflow.context_nodes import analyze_request_node
from back.agent.workflow.response_nodes import answer_from_profile, respond_with_grounded_knowledge, respond_out_of_scope, respond_scope_uncertain, respond_clarify, respond_chitchat
from back.agent.workflow.retrieval_nodes import hybrid_search_node, rerank_node
from back.agent.workflow.routes import route_after_request_analysis

graph_builder = StateGraph(CustomerServiceState)

graph_builder.add_node("analyze_request", analyze_request_node)
graph_builder.add_node("answer_profile", answer_from_profile)
graph_builder.add_node("hybrid_search", hybrid_search_node)
graph_builder.add_node("rerank", rerank_node)
graph_builder.add_node("grounded_answer", respond_with_grounded_knowledge)
graph_builder.add_node("out_of_scope", respond_out_of_scope)
graph_builder.add_node("scope_uncertain", respond_scope_uncertain)
graph_builder.add_node("clarify", respond_clarify)
graph_builder.add_node("chitchat", respond_chitchat)

graph_builder.add_edge(START, "analyze_request")
graph_builder.add_conditional_edges(
    "analyze_request",
    route_after_request_analysis,
    {
        "answer_profile": "answer_profile",
        "hybrid_search": "hybrid_search",
        "clarify": "clarify",
        "chitchat": "chitchat",
        "out_of_scope": "out_of_scope",
        "scope_uncertain": "scope_uncertain",
    },
)

graph_builder.add_edge("answer_profile", END)
graph_builder.add_edge("hybrid_search", "rerank")
graph_builder.add_edge("rerank", "grounded_answer")
graph_builder.add_edge("grounded_answer", END)

graph_builder.add_edge("chitchat", END)
graph_builder.add_edge("out_of_scope", END)
graph_builder.add_edge("scope_uncertain", END)
graph_builder.add_edge("clarify", END)

customer_service_graph = graph_builder.compile()
