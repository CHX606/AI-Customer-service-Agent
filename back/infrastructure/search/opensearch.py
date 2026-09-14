"""OpenSearch 知识检索存储。

同一份文档分块同时建立 BM25 关键词索引和 HNSW 向量索引，所有读写
都强制包含 tenant_id，替代原来的本地 Chroma 与进程内 BM25。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import logging
import os
from threading import Lock
from typing import Any

from langchain_core.documents import Document
from opensearchpy import OpenSearch, helpers

from back.knowledge.retrieval.embeddings import get_embedding_model
from back.knowledge.retrieval.text import build_keyword_terms
from back.domain.tenant import validate_tenant_id


logger = logging.getLogger(__name__)

DEFAULT_INDEX_ALIAS = "customer-service-knowledge"
DEFAULT_INDEX_VERSION = "v1"
DEFAULT_SEARCH_PIPELINE = "customer-service-hybrid-rrf-v1"
DEFAULT_VECTOR_DIMENSION = 512
DEFAULT_REQUEST_TIMEOUT = 30
MAX_GET_RESULTS = 10_000

_bootstrap_lock = Lock()

INDEXED_METADATA_FIELDS = {
    "tenant_id",
    "source_id",
    "content_hash",
    "section_id",
    "section_title",
    "chunk_index",
    "content_type",
    "filename",
    "image_sha256",
    "image_dhash",
    "image_detail_dhash",
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} 必须是整数") from error
    if value < minimum:
        raise ValueError(f"{name} 不能小于 {minimum}")
    return value


@dataclass(frozen=True)
class OpenSearchSettings:
    """OpenSearch 连接和索引配置。"""

    url: str
    username: str
    password: str
    verify_certs: bool
    ca_certs: str | None
    request_timeout: int
    index_alias: str
    index_version: str
    search_pipeline: str
    vector_dimension: int
    shards: int
    replicas: int

    @property
    def physical_index(self) -> str:
        return f"{self.index_alias}-{self.index_version}"


def get_opensearch_settings() -> OpenSearchSettings:
    """从环境变量读取 OpenSearch 设置。"""
    alias = os.getenv("OPENSEARCH_INDEX_ALIAS", DEFAULT_INDEX_ALIAS).strip()
    version = os.getenv(
        "OPENSEARCH_INDEX_VERSION", DEFAULT_INDEX_VERSION
    ).strip()
    pipeline = os.getenv(
        "OPENSEARCH_SEARCH_PIPELINE", DEFAULT_SEARCH_PIPELINE
    ).strip()
    if not alias or alias != alias.lower():
        raise ValueError("OPENSEARCH_INDEX_ALIAS 必须是非空小写名称")
    if not version or not pipeline:
        raise ValueError("OpenSearch 索引版本和搜索管道名称不能为空")

    return OpenSearchSettings(
        url=os.getenv("OPENSEARCH_URL", "https://localhost:9200").strip(),
        username=os.getenv("OPENSEARCH_USERNAME", "admin").strip(),
        password=os.getenv("OPENSEARCH_PASSWORD", "").strip(),
        verify_certs=_env_bool("OPENSEARCH_VERIFY_CERTS", True),
        ca_certs=os.getenv("OPENSEARCH_CA_CERTS") or None,
        request_timeout=_env_int(
            "OPENSEARCH_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT, minimum=1
        ),
        index_alias=alias,
        index_version=version,
        search_pipeline=pipeline,
        vector_dimension=_env_int(
            "OPENSEARCH_VECTOR_DIMENSION",
            DEFAULT_VECTOR_DIMENSION,
            minimum=1,
        ),
        shards=_env_int("OPENSEARCH_INDEX_SHARDS", 1, minimum=1),
        replicas=_env_int("OPENSEARCH_INDEX_REPLICAS", 1, minimum=0),
    )


@lru_cache(maxsize=1)
def get_opensearch_client() -> OpenSearch:
    """创建并缓存官方 OpenSearch Python 客户端。"""
    settings = get_opensearch_settings()
    kwargs: dict[str, Any] = {
        "hosts": [settings.url],
        "http_compress": True,
        "verify_certs": settings.verify_certs,
        "ssl_assert_hostname": settings.verify_certs,
        "ssl_show_warn": settings.verify_certs,
        "timeout": settings.request_timeout,
        "max_retries": 3,
        "retry_on_timeout": True,
    }
    if settings.username:
        if not settings.password:
            raise RuntimeError(
                "已配置 OPENSEARCH_USERNAME，但 OPENSEARCH_PASSWORD 为空"
            )
        kwargs["http_auth"] = (settings.username, settings.password)
    if settings.ca_certs:
        kwargs["ca_certs"] = settings.ca_certs
    return OpenSearch(**kwargs)


def _index_definition(settings: OpenSearchSettings) -> dict[str, Any]:
    """生成文本、关键词和向量共存的索引映射。"""
    keyword_field = {"type": "keyword", "ignore_above": 1024}
    return {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": settings.shards,
                "number_of_replicas": settings.replicas,
            },
            "analysis": {
                "analyzer": {
                    "pretokenized_zh": {
                        "type": "custom",
                        "tokenizer": "whitespace",
                        "filter": ["lowercase"],
                    }
                }
            },
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "document_id": keyword_field,
                "tenant_id": keyword_field,
                "source_id": keyword_field,
                "content_hash": keyword_field,
                "section_id": keyword_field,
                "section_title": {
                    "type": "text",
                    "fields": {"keyword": keyword_field},
                },
                "chunk_index": keyword_field,
                "content_type": keyword_field,
                "filename": keyword_field,
                "image_sha256": keyword_field,
                "image_dhash": keyword_field,
                "image_detail_dhash": keyword_field,
                "content": {"type": "text"},
                "content_terms": {
                    "type": "text",
                    "analyzer": "pretokenized_zh",
                    "search_analyzer": "pretokenized_zh",
                    "similarity": "BM25",
                },
                "embedding": {
                    "type": "knn_vector",
                    "dimension": settings.vector_dimension,
                    "method": {
                        "name": "hnsw",
                        "engine": "lucene",
                        "space_type": "cosinesimil",
                        "parameters": {
                            "m": 16,
                            "ef_construction": 100,
                        },
                    },
                },
                "metadata": {"type": "object", "enabled": False},
            },
        },
        "aliases": {
            settings.index_alias: {
                "is_write_index": True,
            }
        },
    }


def _search_pipeline_definition() -> dict[str, Any]:
    return {
        "description": "BM25 and vector hybrid search using RRF",
        "phase_results_processors": [
            {
                "score-ranker-processor": {
                    "combination": {
                        "technique": "rrf",
                        "rank_constant": 60,
                    }
                }
            }
        ],
    }


def should_bootstrap_search_backend() -> bool:
    """控制应用启动时是否检查并创建 OpenSearch 结构。"""
    return _env_bool("OPENSEARCH_BOOTSTRAP_ON_STARTUP", True)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _metadata_value(metadata: dict[str, Any], name: str) -> str:
    value = metadata.get(name, "")
    return "" if value is None else str(value)


def _source_to_document(hit: dict[str, Any]) -> Document:
    source = hit.get("_source") or {}
    metadata = dict(source.get("metadata") or {})
    for field in INDEXED_METADATA_FIELDS:
        value = source.get(field)
        if value not in {None, ""}:
            metadata[field] = value
    metadata["opensearch_id"] = str(hit.get("_id", ""))
    if hit.get("_score") is not None:
        metadata["opensearch_score"] = float(hit["_score"])
    return Document(
        page_content=str(source.get("content", "")),
        metadata=metadata,
    )


class OpenSearchKnowledgeStore:
    """面向单个租户的 OpenSearch 知识索引访问器。"""

    def __init__(
        self,
        tenant_id: str,
        *,
        client: OpenSearch | None = None,
        settings: OpenSearchSettings | None = None,
    ) -> None:
        self.tenant_id = validate_tenant_id(tenant_id)
        self.settings = settings or get_opensearch_settings()
        self.client = client or get_opensearch_client()
        self._ready = False

    def ensure_ready(self) -> None:
        if self._ready:
            return
        with _bootstrap_lock:
            alias_exists = bool(
                self.client.indices.exists_alias(
                    name=self.settings.index_alias
                )
            )
            if not alias_exists:
                physical_exists = bool(
                    self.client.indices.exists(
                        index=self.settings.physical_index
                    )
                )
                if not physical_exists:
                    self.client.indices.create(
                        index=self.settings.physical_index,
                        body=_index_definition(self.settings),
                    )
                else:
                    self.client.indices.put_alias(
                        index=self.settings.physical_index,
                        name=self.settings.index_alias,
                        body={"is_write_index": True},
                    )

            self.client.transport.perform_request(
                "PUT",
                f"/_search/pipeline/{self.settings.search_pipeline}",
                body=_search_pipeline_definition(),
            )
            self._ready = True

    def _tenant_filter(self) -> dict[str, Any]:
        return {"term": {"tenant_id": self.tenant_id}}

    def _upsert_documents(
        self,
        documents: list[Document],
        ids: list[str],
    ) -> None:
        if len(documents) != len(ids):
            raise ValueError("documents 与 ids 数量不一致")
        if not documents:
            return
        self.ensure_ready()

        embeddings = get_embedding_model().embed_documents(
            [document.page_content for document in documents]
        )
        expected_dimension = self.settings.vector_dimension
        for vector in embeddings:
            if len(vector) != expected_dimension:
                raise ValueError(
                    "Embedding 维度与 OpenSearch mapping 不一致："
                    f"实际 {len(vector)}，配置 {expected_dimension}"
                )

        actions = []
        for document_id, document, embedding in zip(
            ids, documents, embeddings, strict=True
        ):
            metadata = {
                **dict(document.metadata),
                "tenant_id": self.tenant_id,
            }
            source = {
                "document_id": document_id,
                "tenant_id": self.tenant_id,
                "source_id": _metadata_value(metadata, "source_id"),
                "content_hash": _metadata_value(metadata, "content_hash"),
                "section_id": _metadata_value(metadata, "section_id"),
                "section_title": _metadata_value(metadata, "section_title"),
                "chunk_index": _metadata_value(metadata, "chunk_index"),
                "content_type": _metadata_value(metadata, "content_type"),
                "filename": _metadata_value(metadata, "filename"),
                "image_sha256": _metadata_value(metadata, "image_sha256"),
                "image_dhash": _metadata_value(metadata, "image_dhash"),
                "image_detail_dhash": _metadata_value(
                    metadata, "image_detail_dhash"
                ),
                "content": document.page_content,
                "content_terms": build_keyword_terms(document.page_content),
                "embedding": [float(item) for item in embedding],
                "metadata": _json_safe(metadata),
            }
            actions.append(
                {
                    "_op_type": "index",
                    "_index": self.settings.index_alias,
                    "_id": document_id,
                    "_source": source,
                }
            )

        helpers.bulk(
            self.client,
            actions,
            refresh="wait_for",
            request_timeout=self.settings.request_timeout,
        )

    def add_documents(
        self, documents: list[Document], ids: list[str]
    ) -> None:
        self._upsert_documents(documents, ids)

    def update_documents(
        self, documents: list[Document], ids: list[str]
    ) -> None:
        self._upsert_documents(documents, ids)

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        self.ensure_ready()
        actions = [
            {
                "_op_type": "delete",
                "_index": self.settings.index_alias,
                "_id": document_id,
            }
            for document_id in ids
        ]
        helpers.bulk(
            self.client,
            actions,
            ignore_status=(404,),
            refresh="wait_for",
            request_timeout=self.settings.request_timeout,
        )

    def _where_filters(self, where: dict[str, Any] | None) -> list[dict]:
        filters: list[dict] = [self._tenant_filter()]
        for field, condition in (where or {}).items():
            if field not in INDEXED_METADATA_FIELDS:
                raise ValueError(f"OpenSearch 不支持过滤元数据字段：{field}")
            if isinstance(condition, dict):
                if "$eq" in condition:
                    filters.append({"term": {field: condition["$eq"]}})
                elif "$in" in condition:
                    filters.append({"terms": {field: condition["$in"]}})
                else:
                    raise ValueError(f"不支持的过滤条件：{condition}")
            else:
                filters.append({"term": {field: condition}})
        return filters

    def get(
        self,
        *,
        where: dict[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, list[Any]]:
        """兼容旧调用方式，按租户与元数据读取文档。"""
        self.ensure_ready()
        response = self.client.search(
            index=self.settings.index_alias,
            body={
                "size": MAX_GET_RESULTS,
                "query": {
                    "bool": {
                        "filter": self._where_filters(where),
                    }
                },
                "_source": {
                    "excludes": ["embedding", "content_terms"],
                },
            },
            params={"request_timeout": self.settings.request_timeout},
        )
        hits = response.get("hits", {}).get("hits", [])
        documents = [_source_to_document(hit) for hit in hits]
        requested = set(
            ["documents", "metadatas"] if include is None else include
        )
        result: dict[str, list[Any]] = {
            "ids": [str(hit.get("_id", "")) for hit in hits],
        }
        if "documents" in requested:
            result["documents"] = [item.page_content for item in documents]
        if "metadatas" in requested:
            result["metadatas"] = [item.metadata for item in documents]
        return result

    def get_source_ids(self, source_id: str) -> list[str]:
        """使用 scroll 读取完整数据源 ID 集，不受单次查询窗口限制。"""
        self.ensure_ready()
        query = {
            "query": {
                "bool": {
                    "filter": [
                        self._tenant_filter(),
                        {"term": {"source_id": source_id}},
                    ]
                }
            },
            "_source": False,
        }
        return [
            str(hit["_id"])
            for hit in helpers.scan(
                self.client,
                index=self.settings.index_alias,
                query=query,
                request_timeout=self.settings.request_timeout,
            )
        ]

    def count(self) -> int:
        self.ensure_ready()
        response = self.client.count(
            index=self.settings.index_alias,
            body={"query": self._tenant_filter()},
            params={"request_timeout": self.settings.request_timeout},
        )
        return int(response.get("count") or 0)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
    ) -> list[tuple[Document, float]]:
        """执行租户过滤后的向量检索，分数越大越相关。"""
        self.ensure_ready()
        vector = get_embedding_model().embed_query(query)
        if len(vector) != self.settings.vector_dimension:
            raise ValueError("查询向量维度与 OpenSearch mapping 不一致")
        response = self.client.search(
            index=self.settings.index_alias,
            body={
                "size": k,
                "_source": {"excludes": ["embedding", "content_terms"]},
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": [float(item) for item in vector],
                            "k": k,
                            "filter": self._tenant_filter(),
                        }
                    }
                },
            },
            params={"request_timeout": self.settings.request_timeout},
        )
        return [
            (_source_to_document(hit), float(hit.get("_score") or 0.0))
            for hit in response.get("hits", {}).get("hits", [])
        ]

    def keyword_search_with_score(
        self,
        query: str,
        k: int = 5,
    ) -> list[tuple[Document, float]]:
        """执行租户过滤后的持久化 BM25 关键词检索。"""
        terms = build_keyword_terms(query)
        if not terms:
            return []
        self.ensure_ready()
        response = self.client.search(
            index=self.settings.index_alias,
            body={
                "size": k,
                "_source": {"excludes": ["embedding", "content_terms"]},
                "query": {
                    "bool": {
                        "must": [
                            {
                                "match": {
                                    "content_terms": {
                                        "query": terms,
                                        "operator": "or",
                                    }
                                }
                            }
                        ],
                        "filter": [self._tenant_filter()],
                    }
                },
            },
            params={"request_timeout": self.settings.request_timeout},
        )
        return [
            (_source_to_document(hit), float(hit.get("_score") or 0.0))
            for hit in response.get("hits", {}).get("hits", [])
            if float(hit.get("_score") or 0.0) > 0
        ]

    def hybrid_search_with_score(
        self,
        query: str,
        *,
        k: int = 12,
        vector_candidates: int = 12,
        pagination_depth: int = 12,
    ) -> list[tuple[Document, float]]:
        """由 OpenSearch 搜索管道执行 BM25、向量召回与 RRF。"""
        terms = build_keyword_terms(query)
        if not terms:
            return self.similarity_search_with_score(query, k=k)
        self.ensure_ready()
        vector = get_embedding_model().embed_query(query)
        if len(vector) != self.settings.vector_dimension:
            raise ValueError("查询向量维度与 OpenSearch mapping 不一致")

        response = self.client.search(
            index=self.settings.index_alias,
            body={
                "size": k,
                "_source": {"excludes": ["embedding", "content_terms"]},
                "query": {
                    "hybrid": {
                        "pagination_depth": max(pagination_depth, k),
                        "filter": self._tenant_filter(),
                        "queries": [
                            {
                                "match": {
                                    "content_terms": {
                                        "query": terms,
                                        "operator": "or",
                                    }
                                }
                            },
                            {
                                "knn": {
                                    "embedding": {
                                        "vector": [
                                            float(item) for item in vector
                                        ],
                                        "k": max(vector_candidates, k),
                                    }
                                }
                            },
                        ],
                    }
                },
            },
            params={
                "search_pipeline": self.settings.search_pipeline,
                "request_timeout": self.settings.request_timeout,
            },
        )
        return [
            (_source_to_document(hit), float(hit.get("_score") or 0.0))
            for hit in response.get("hits", {}).get("hits", [])
        ]

    def delete_source(self, source_id: str) -> int:
        self.ensure_ready()
        response = self.client.delete_by_query(
            index=self.settings.index_alias,
            body={
                "query": {
                    "bool": {
                        "filter": [
                            self._tenant_filter(),
                            {"term": {"source_id": source_id}},
                        ]
                    }
                }
            },
            params={
                "conflicts": "proceed",
                "refresh": "true",
                "request_timeout": self.settings.request_timeout,
            },
        )
        return _confirmed_delete_count(response)

    def reset_collection(self) -> int:
        """仅清空当前租户的数据，不影响其他租户。"""
        self.ensure_ready()
        response = self.client.delete_by_query(
            index=self.settings.index_alias,
            body={"query": self._tenant_filter()},
            params={
                "conflicts": "proceed",
                "refresh": "true",
                "request_timeout": self.settings.request_timeout,
            },
        )
        return _confirmed_delete_count(response)


def _confirmed_delete_count(response: dict) -> int:
    if response.get("timed_out") or response.get("version_conflicts") or response.get("failures"):
        raise RuntimeError("搜索索引删除未完成，请重试；源文件和登记记录已保留。")
    return int(response.get("deleted") or 0)


@lru_cache(maxsize=128)
def get_vector_store(
    tenant_id: str = "default",
) -> OpenSearchKnowledgeStore:
    """兼容旧函数名：返回租户隔离的 OpenSearch 检索存储。"""
    return OpenSearchKnowledgeStore(tenant_id)


def ensure_search_backend() -> None:
    """启动时验证连接并幂等创建索引与 RRF 搜索管道。"""
    store = get_vector_store("default")
    store.client.cluster.health(
        params={"request_timeout": store.settings.request_timeout}
    )
    store.ensure_ready()


def add_source_documents(
    tenant_id: str,
    source_id: str,
    documents: list[Document],
) -> int:
    """为指定租户的数据源增量写入或更新 OpenSearch 文档。"""
    if not documents:
        return 0

    store = get_vector_store(tenant_id)
    existing_ids = set(store.get_source_ids(source_id))

    content_hash = str(documents[0].metadata.get("content_hash", "")).strip()
    if not content_hash:
        digest_source = "\n".join(doc.page_content for doc in documents)
        content_hash = hashlib.sha256(
            digest_source.encode("utf-8")
        ).hexdigest()
    version = content_hash[:12]
    new_ids = [
        f"{source_id}_{version}_{doc.metadata.get('chunk_index', index)}"
        for index, doc in enumerate(documents)
    ]
    if len(set(new_ids)) != len(new_ids):
        raise ValueError("同一数据源生成了重复的检索分片 ID")

    document_by_id = dict(zip(new_ids, documents, strict=True))
    added_ids = [item for item in new_ids if item not in existing_ids]
    retained_ids = [item for item in new_ids if item in existing_ids]
    try:
        if added_ids:
            store.add_documents(
                [document_by_id[item] for item in added_ids], added_ids
            )
        if retained_ids:
            store.update_documents(
                [document_by_id[item] for item in retained_ids], retained_ids
            )
    except Exception:
        if added_ids:
            try:
                store.delete(added_ids)
            except Exception:
                logger.exception(
                    "回滚失败的 OpenSearch 分片时发生异常",
                    extra={"tenant_id": tenant_id, "source_id": source_id},
                )
        raise

    stale_ids = sorted(existing_ids.difference(new_ids))
    if stale_ids:
        store.delete(stale_ids)
    return len(documents)


def delete_source_documents(tenant_id: str, source_id: str) -> int:
    """按 tenant_id 与 source_id 精准删除检索分片。"""
    return get_vector_store(tenant_id).delete_source(source_id)


def rebuild_tenant_vector_store(tenant_id: str = "default"):
    """旧调用入口；实际重建流程由索引服务负责。"""
    from back.knowledge.indexing.rebuild import rebuild_tenant_vector_store as rebuild
    return rebuild(tenant_id)


def rebuild_vector_store():
    return rebuild_tenant_vector_store("default")


if __name__ == "__main__":
    ensure_search_backend()
    store = rebuild_vector_store()
    stored_count = store.count()
    print("OpenSearch 知识索引构建成功")
    print("索引别名：", store.settings.index_alias)
    print("保存的文本块数量：", stored_count)
