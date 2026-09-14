"""OpenSearch 查询和删除必须强制使用 tenant_id 隔离。"""

from unittest.mock import Mock

from back.infrastructure.search.opensearch import OpenSearchKnowledgeStore, OpenSearchSettings


def settings() -> OpenSearchSettings:
    return OpenSearchSettings(
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
        shards=1,
        replicas=0,
    )


def test_keyword_query_contains_tenant_filter(monkeypatch):
    client = Mock()
    client.search.return_value = {"hits": {"hits": []}}
    store = OpenSearchKnowledgeStore(
        "tenant_a", client=client, settings=settings()
    )
    store._ready = True

    store.keyword_search_with_score("退款时间", k=5)

    body = client.search.call_args.kwargs["body"]
    assert {"term": {"tenant_id": "tenant_a"}} in body["query"]["bool"][
        "filter"
    ]


def test_hybrid_query_contains_top_level_tenant_filter(monkeypatch):
    client = Mock()
    client.search.return_value = {"hits": {"hits": []}}
    store = OpenSearchKnowledgeStore(
        "tenant_b", client=client, settings=settings()
    )
    store._ready = True
    embedding = Mock()
    embedding.embed_query.return_value = [0.1, 0.2, 0.3]
    monkeypatch.setattr("back.infrastructure.search.opensearch.get_embedding_model", lambda: embedding)

    store.hybrid_search_with_score("退款时间", k=5)

    body = client.search.call_args.kwargs["body"]
    assert body["query"]["hybrid"]["filter"] == {
        "term": {"tenant_id": "tenant_b"}
    }
    assert client.search.call_args.kwargs["params"]["search_pipeline"] == "test-rrf"


def test_delete_source_filters_by_tenant_and_source():
    client = Mock()
    client.delete_by_query.return_value = {"deleted": 4}
    store = OpenSearchKnowledgeStore(
        "tenant_a", client=client, settings=settings()
    )
    store._ready = True

    assert store.delete_source("src_01") == 4

    filters = client.delete_by_query.call_args.kwargs["body"]["query"]["bool"][
        "filter"
    ]
    assert {"term": {"tenant_id": "tenant_a"}} in filters
    assert {"term": {"source_id": "src_01"}} in filters
