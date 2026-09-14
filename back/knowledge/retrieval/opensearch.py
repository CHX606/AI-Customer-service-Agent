"""兼容旧导入路径；搜索引擎实现位于 infrastructure.search。"""
from back.infrastructure.search.opensearch import (
    OpenSearchKnowledgeStore, OpenSearchSettings, get_vector_store, get_opensearch_client,
    get_opensearch_settings, ensure_search_backend, should_bootstrap_search_backend,
    add_source_documents, delete_source_documents, rebuild_tenant_vector_store, rebuild_vector_store,
)
