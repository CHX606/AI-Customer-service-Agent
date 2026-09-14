"""客服工作流职责模块：retrieval_nodes"""
from back.agent.workflow.state import CustomerServiceState
from back.knowledge.retrieval.hybrid import RRF_CANDIDATES, retrieve_documents_multi_query
from back.knowledge.retrieval.reranker import rerank_documents

def hybrid_search_node(state: CustomerServiceState):
    """对原问题和改写问题在当前租户集合中执行多查询宽召回。"""
    search_queries = state.get("search_queries", [])
    resolved_query = state.get("resolved_query")
    tenant_id = state.get("tenant_id", "default")

    if not search_queries and resolved_query:
        search_queries = [resolved_query]

    if not search_queries:
        raise ValueError("状态中没有找到可用的检索问题")

    documents = retrieve_documents_multi_query(
        queries=search_queries,
        tenant_id=tenant_id,
        limit=RRF_CANDIDATES,
    )

    return {
        "retrieval_candidates": documents,
    }

def rerank_node(state: CustomerServiceState):
    """使用 CrossEncoder 对 RRF 候选资料进行精排。"""
    resolved_query = state.get("resolved_query")
    candidates = state.get("retrieval_candidates", [])

    if not resolved_query:
        raise ValueError("状态中没有找到resolved_query")

    documents = rerank_documents(
        query=resolved_query,
        documents=candidates,
    )

    return {
        "retrieved_documents": documents,
    }
