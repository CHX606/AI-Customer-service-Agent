"""相邻文本块扩展的窗口、隔离、去重和失败回退测试。"""

from unittest.mock import Mock

from langchain_core.documents import Document
import pytest

import back.agent.workflow.retrieval_nodes as retrieval_nodes
import back.knowledge.retrieval.expansion as expansion
from back.infrastructure.search.opensearch import OpenSearchKnowledgeStore


def doc(index=10, text="中心正文", **metadata):
    return Document(page_content=text, metadata={
        "tenant_id": "tenant_a",
        "source_id": "source_a",
        "section_id": "4.2",
        "section_title": "续费规则",
        "content_hash": "version_a",
        "chunk_index": str(index),
        "reranker_score": 0.92,
        **metadata,
    })


def install_store(monkeypatch, documents):
    store = Mock()
    store.get.return_value = {
        "documents": [item.page_content for item in documents],
        "metadatas": [item.metadata for item in documents],
    }
    factory = Mock(return_value=store)
    monkeypatch.setattr(expansion, "get_vector_store", factory)
    return store, factory


def test_expands_only_same_section_window_and_preserves_ranking_metadata(monkeypatch):
    center = doc(text="4.2 续费规则\n续费延长有效期")
    store, factory = install_store(monkeypatch, [
        doc(8, "不相邻"),
        doc(9, "4.2 续费规则\n流量已用完"),
        doc(10, "同版本但已变化的中心"),
        doc(11, "4.2 续费规则\n需要购买重置"),
        doc(12, "不相邻"),
    ])

    result = expansion.expand_neighbor_chunks([center], "tenant_a")

    assert result[0].page_content == "4.2 续费规则\n流量已用完\n续费延长有效期\n需要购买重置"
    assert result[0].metadata == {**center.metadata, "expanded_chunks": [9, 10, 11]}
    assert center.page_content == "4.2 续费规则\n续费延长有效期"
    factory.assert_called_once_with("tenant_a")
    assert store.get.call_args.kwargs["where"] == {
        "source_id": {"$in": ["source_a"]},
        "section_id": {"$in": ["4.2"]},
        "content_hash": {"$in": ["version_a"]},
        "chunk_index": {"$in": ["10", "11", "9"]},
    }


@pytest.mark.parametrize("changed", [
    {"tenant_id": "tenant_b"},
    {"source_id": "source_b"},
    {"section_id": "4.3"},
    {"content_hash": "version_b"},
    {"content_type": "image"},
    {"content_type": "image_diagnostic"},
    {"content_type": "image_semantic"},
])
def test_excludes_foreign_or_image_neighbors(monkeypatch, changed):
    center = doc()
    install_store(monkeypatch, [doc(9, "不应混入", **changed)])

    result = expansion.expand_neighbor_chunks([center], "tenant_a")

    assert result == [center]
    assert result[0] is center


@pytest.mark.parametrize("changed", [
    {"tenant_id": "tenant_b"},
    {"source_id": ""},
    {"section_id": ""},
    {"chunk_index": "invalid"},
    {"chunk_index": None},
    {"chunk_index": -1},
    {"chunk_index": 1.5},
    {"content_type": "image"},
    {"content_type": "image_diagnostic"},
    {"content_type": "image_semantic"},
])
def test_ineligible_targets_do_not_query(monkeypatch, changed):
    original = [doc(**changed)]
    _, factory = install_store(monkeypatch, [])

    assert expansion.expand_neighbor_chunks(original, "tenant_a") is original
    factory.assert_not_called()


def test_empty_documents_do_not_query(monkeypatch):
    _, factory = install_store(monkeypatch, [])
    assert expansion.expand_neighbor_chunks([], "tenant_a") == []
    factory.assert_not_called()


def test_first_chunk_uses_only_nonnegative_window(monkeypatch):
    center = doc(0)
    store, _ = install_store(monkeypatch, [doc(1, "后文")])

    result = expansion.expand_neighbor_chunks([center], "tenant_a")

    assert result[0].metadata["expanded_chunks"] == [0, 1]
    assert store.get.call_args.kwargs["where"]["chunk_index"] == {"$in": ["0", "1"]}


def test_combines_once_and_preserves_result_count_and_order(monkeypatch):
    first, second, third = doc(10), doc(11, "后文"), doc(20, "另一章节", section_id="4.3")
    original = [first, second, third]
    install_store(monkeypatch, [doc(9, "前文"), second])

    result = expansion.expand_neighbor_chunks(original, "tenant_a")

    assert len(result) == len(original)
    assert [item.metadata["chunk_index"] for item in result] == ["10", "11", "20"]
    assert result[0].metadata["expanded_chunks"] == [9, 10, 11]
    assert result[1] is second
    assert result[2] is third


def test_removes_exact_heading_and_suffix_prefix_overlap(monkeypatch):
    center = doc(text="4.2 续费规则\n这是切块共享的段落。后文")
    install_store(monkeypatch, [doc(9, "4.2 续费规则\n前文。这是切块共享的段落。")])

    result = expansion.expand_neighbor_chunks([center], "tenant_a")

    assert result[0].page_content == "4.2 续费规则\n前文。这是切块共享的段落。后文"


def test_short_matching_text_is_not_removed_as_overlap(monkeypatch):
    center = doc(text="30天内可使用")
    install_store(monkeypatch, [doc(9, "有效期为30")])

    result = expansion.expand_neighbor_chunks([center], "tenant_a")

    assert result[0].page_content == "有效期为30\n30天内可使用"


def test_generic_chunks_preserve_first_line_as_body(monkeypatch):
    center = doc(text="正文开头的延续\n第二段", section_id="通用", section_title="手册")
    install_store(monkeypatch, [doc(9, "正文开头\n第一段", section_id="通用", section_title="手册")])

    result = expansion.expand_neighbor_chunks([center], "tenant_a")

    assert result[0].page_content == "正文开头\n第一段\n正文开头的延续\n第二段"


@pytest.mark.parametrize("total, expected_expanded", [(1400, True), (1401, False)])
def test_enforces_total_character_limit(monkeypatch, total, expected_expanded):
    center = doc(text="中" * 400)
    install_store(monkeypatch, [doc(9, "前" * (total - 401))])

    result = expansion.expand_neighbor_chunks([center], "tenant_a")

    assert ("expanded_chunks" in result[0].metadata) is expected_expanded
    if expected_expanded:
        assert len(result[0].page_content) == expansion.MAX_CHARS
    else:
        assert result[0] is center


def test_missing_versions_do_not_mix_with_versioned_neighbors(monkeypatch):
    center = doc(content_hash="")
    install_store(monkeypatch, [doc(9, "另一版本")])

    assert expansion.expand_neighbor_chunks([center], "tenant_a")[0] is center


def test_store_query_keeps_mandatory_tenant_filter(monkeypatch):
    client = Mock()
    client.search.return_value = {"hits": {"hits": []}}
    store = OpenSearchKnowledgeStore(
        "tenant_a",
        client=client,
        settings=Mock(index_alias="knowledge", request_timeout=3),
    )
    store._ready = True
    monkeypatch.setattr(expansion, "get_vector_store", lambda tenant_id: store)

    expansion.expand_neighbor_chunks([doc()], "tenant_a")

    filters = client.search.call_args.kwargs["body"]["query"]["bool"]["filter"]
    assert {"term": {"tenant_id": "tenant_a"}} in filters
    assert {"terms": {"content_hash": ["version_a"]}} in filters
    assert {"terms": {"chunk_index": ["10", "11", "9"]}} in filters


def test_rerank_node_falls_back_when_expansion_fails(monkeypatch, caplog):
    original = [doc()]
    monkeypatch.setattr(retrieval_nodes, "rerank_documents", Mock(return_value=original))
    expand = Mock(side_effect=RuntimeError("搜索不可用"))
    monkeypatch.setattr(retrieval_nodes, "expand_neighbor_chunks", expand)

    result = retrieval_nodes.rerank_node({
        "tenant_id": "tenant_a", "resolved_query": "续费后没有流量", "retrieval_candidates": original,
    })

    assert result["retrieved_documents"] is original
    expand.assert_called_once_with(original, "tenant_a")
    assert "相邻块扩展失败，使用原始结果" in caplog.text


def test_rerank_node_passes_expanded_documents(monkeypatch):
    original, expanded = [doc()], [doc(text="补齐后的正文")]
    monkeypatch.setattr(retrieval_nodes, "rerank_documents", Mock(return_value=original))
    expand = Mock(return_value=expanded)
    monkeypatch.setattr(retrieval_nodes, "expand_neighbor_chunks", expand)

    result = retrieval_nodes.rerank_node({"resolved_query": "续费后没有流量"})

    assert result["retrieved_documents"] is expanded
    expand.assert_called_once_with(original, "default")
