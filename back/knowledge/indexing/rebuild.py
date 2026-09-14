"""从知识源重建索引的显式用例，仅由运维入口调用。"""
from langchain_core.documents import Document
from back.knowledge.ingestion.knowledge_loader import load_knowledge_documents
from back.infrastructure.search.opensearch import OpenSearchKnowledgeStore, get_vector_store, add_source_documents
from back.knowledge.retrieval.cache import semantic_cache_mutation

def rebuild_tenant_vector_store(
    tenant_id: str = "default",
) -> OpenSearchKnowledgeStore:
    with semantic_cache_mutation(tenant_id):
        return _rebuild_tenant_vector_store(tenant_id)


def _rebuild_tenant_vector_store(tenant_id: str) -> OpenSearchKnowledgeStore:
    """从事实来源重建指定租户的 OpenSearch 索引数据。"""
    knowledge_documents = load_knowledge_documents(tenant_id)
    store = get_vector_store(tenant_id)
    store.reset_collection()
    grouped: dict[str, list[Document]] = {}
    for document in knowledge_documents:
        source_id = str(document.metadata.get("source_id", "legacy"))
        grouped.setdefault(source_id, []).append(document)
    for source_id, documents in grouped.items():
        add_source_documents(tenant_id, source_id, documents)
    return store


def rebuild_vector_store() -> OpenSearchKnowledgeStore:
    """兼容旧接口：重建 default 租户。"""
    return rebuild_tenant_vector_store("default")
