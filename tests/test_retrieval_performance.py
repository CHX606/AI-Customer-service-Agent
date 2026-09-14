"""OpenSearch 混合检索查询数量、候选规模与跨查询 RRF 测试。"""

from unittest.mock import patch

from langchain_core.documents import Document

import back.knowledge.retrieval.hybrid as hybrid
from back.agent.workflow.retrieval_nodes import hybrid_search_node


def doc(name: str) -> Document:
    return Document(
        page_content=f"{name}正文",
        metadata={
            "tenant_id": "default",
            "source_id": "source",
            "section_id": name,
            "chunk_index": "0",
        },
    )


def test_retrieval_limits_are_bounded_for_cpu_inference():
    assert hybrid.MAX_SEARCH_QUERIES == 3
    assert hybrid.VECTOR_CANDIDATES <= 12
    assert hybrid.KEYWORD_CANDIDATES <= 12
    assert hybrid.RRF_CANDIDATES <= 12


def test_multi_query_caps_distinct_queries_to_three():
    seen = []

    def fake_hybrid(question, tenant_id, limit=hybrid.RRF_CANDIDATES):
        seen.append((question, tenant_id, limit))
        return []

    with patch.object(hybrid, "retrieve_hybrid_candidates", fake_hybrid):
        hybrid.retrieve_documents_multi_query(
            ["查询一", "查询二", "查询三", "查询四", "查询五"],
            tenant_id="tenant_a",
        )

    assert [item[0] for item in seen] == ["查询一", "查询二", "查询三"]
    assert all(item[1] == "tenant_a" for item in seen)


def test_duplicates_do_not_consume_query_budget():
    seen = []

    def fake_hybrid(question, tenant_id, limit=hybrid.RRF_CANDIDATES):
        seen.append(question)
        return []

    with patch.object(hybrid, "retrieve_hybrid_candidates", fake_hybrid):
        hybrid.retrieve_documents_multi_query(
            [" 查询一 ", "查询一", "查询二", "查询三", "查询四"]
        )

    assert seen == ["查询一", "查询二", "查询三"]


def test_multi_query_result_never_exceeds_requested_limit():
    candidates = [(doc(f"d{i}"), float(20 - i)) for i in range(20)]
    with patch.object(
        hybrid, "retrieve_hybrid_candidates", return_value=candidates
    ):
        results = hybrid.retrieve_documents_multi_query(["问题"], limit=7)
    assert len(results) == 7


def test_single_query_uses_opensearch_hybrid_result():
    shared = doc("shared")
    with patch.object(
        hybrid,
        "retrieve_hybrid_candidates",
        return_value=[(shared, 0.032)],
    ) as mocked:
        results = hybrid.retrieve_documents_hybrid(
            "问题", tenant_id="tenant_a", limit=3
        )

    mocked.assert_called_once_with("问题", "tenant_a", 3)
    assert results[0].metadata["retrieval_method"] == "opensearch_hybrid_rrf"
    assert results[0].metadata["opensearch_hybrid_rank"] == 1


def test_rrf_tracks_all_queries_that_matched_a_document():
    shared = doc("shared")

    def fake_hybrid(question, tenant_id, limit=hybrid.RRF_CANDIDATES):
        return [(shared, 0.03)]

    with patch.object(hybrid, "retrieve_hybrid_candidates", fake_hybrid):
        results = hybrid.retrieve_documents_multi_query(["问题一", "问题二"])

    assert results[0].metadata["matched_queries"] == ["问题一", "问题二"]
    assert len(results[0].metadata["query_ranks"]) == 2


def test_graph_requests_only_bounded_rrf_candidate_count():
    with patch(
        "back.agent.workflow.retrieval_nodes.retrieve_documents_multi_query",
        return_value=[],
    ) as mocked:
        result = hybrid_search_node(
            {
                "tenant_id": "default",
                "resolved_query": "流量没有重置",
                "search_queries": ["流量没有重置"],
            }
        )

    assert result["retrieval_candidates"] == []
    assert mocked.call_args.kwargs["limit"] <= 12


def test_empty_and_whitespace_queries_do_not_search():
    with patch.object(hybrid, "retrieve_hybrid_candidates") as mocked:
        results = hybrid.retrieve_documents_multi_query(["", "  ", "\n"])

    assert results == []
    mocked.assert_not_called()
