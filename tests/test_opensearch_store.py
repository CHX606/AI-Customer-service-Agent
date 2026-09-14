"""OpenSearch 索引映射与持久化文档结构测试。"""

from unittest.mock import Mock

from langchain_core.documents import Document

import back.infrastructure.search.opensearch as vectorstore


def settings() -> vectorstore.OpenSearchSettings:
    return vectorstore.OpenSearchSettings(
        url="http://localhost:9200",
        username="",
        password="",
        verify_certs=False,
        ca_certs=None,
        request_timeout=3,
        index_alias="customer-service-knowledge",
        index_version="test",
        search_pipeline="test-rrf",
        vector_dimension=3,
        shards=2,
        replicas=1,
    )


def test_bootstrap_creates_hybrid_index_and_rrf_pipeline():
    client = Mock()
    client.indices.exists_alias.return_value = False
    client.indices.exists.return_value = False
    store = vectorstore.OpenSearchKnowledgeStore(
        "default", client=client, settings=settings()
    )

    store.ensure_ready()

    create = client.indices.create.call_args.kwargs
    assert create["index"] == "customer-service-knowledge-test"
    properties = create["body"]["mappings"]["properties"]
    assert properties["content_terms"]["similarity"] == "BM25"
    assert properties["embedding"]["type"] == "knn_vector"
    assert properties["embedding"]["dimension"] == 3
    pipeline = client.transport.perform_request.call_args.kwargs["body"]
    processor = pipeline["phase_results_processors"][0]
    assert processor["score-ranker-processor"]["combination"]["technique"] == "rrf"


def test_indexed_document_contains_terms_vector_and_tenant(monkeypatch):
    client = Mock()
    store = vectorstore.OpenSearchKnowledgeStore(
        "tenant_a", client=client, settings=settings()
    )
    store._ready = True
    embedding = Mock()
    embedding.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    monkeypatch.setattr(vectorstore, "get_embedding_model", lambda: embedding)
    bulk = Mock(return_value=(1, []))
    monkeypatch.setattr(vectorstore.helpers, "bulk", bulk)

    store.add_documents(
        [
            Document(
                page_content="套餐退款时间为三天",
                metadata={
                    "tenant_id": "wrong_tenant",
                    "source_id": "src_01",
                    "chunk_index": 0,
                },
            )
        ],
        ["src_01_hash_0"],
    )

    action = bulk.call_args.args[1][0]
    source = action["_source"]
    assert source["tenant_id"] == "tenant_a"
    assert source["metadata"]["tenant_id"] == "tenant_a"
    assert source["embedding"] == [0.1, 0.2, 0.3]
    assert "退款" in source["content_terms"]
