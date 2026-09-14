"""OpenSearch 原生混合检索与多查询融合。"""

import os

from langchain_core.documents import Document

from back.infrastructure.search.opensearch import get_vector_store


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} 必须是整数") from error
    if value < minimum:
        raise ValueError(f"{name} 不能小于 {minimum}")
    return value


MAX_SEARCH_QUERIES = 3
VECTOR_CANDIDATES = 12
KEYWORD_CANDIDATES = 12
RRF_CANDIDATES = _env_int("RERANKER_CANDIDATES", 12)
FINAL_RESULTS = 5
RRF_CONSTANT = 60


def get_document_key(document: Document) -> tuple[str, str, str, str]:
    metadata = document.metadata
    return (
        str(metadata.get("tenant_id", "")),
        str(metadata.get("source_id", metadata.get("source", ""))),
        str(metadata.get("section_id", "")),
        str(metadata.get("chunk_index", "")),
    )


def retrieve_vector_candidates(
    question: str,
    tenant_id: str = "default",
    limit: int = VECTOR_CANDIDATES,
) -> list[tuple[Document, float]]:
    """OpenSearch HNSW 向量召回，分数越大越相关。"""
    return get_vector_store(tenant_id).similarity_search_with_score(
        query=question,
        k=limit,
    )


def retrieve_keyword_candidates(
    question: str,
    tenant_id: str = "default",
    limit: int = KEYWORD_CANDIDATES,
) -> list[tuple[Document, float]]:
    """OpenSearch 持久化 BM25 召回。"""
    return get_vector_store(tenant_id).keyword_search_with_score(
        query=question,
        k=limit,
    )


def retrieve_hybrid_candidates(
    question: str,
    tenant_id: str = "default",
    limit: int = RRF_CANDIDATES,
) -> list[tuple[Document, float]]:
    """由 OpenSearch 搜索管道执行 BM25、向量召回和 RRF。"""
    return get_vector_store(tenant_id).hybrid_search_with_score(
        question,
        k=limit,
        vector_candidates=VECTOR_CANDIDATES,
        pagination_depth=max(KEYWORD_CANDIDATES, VECTOR_CANDIDATES, limit),
    )


def retrieve_documents_hybrid(
    question: str,
    tenant_id: str = "default",
    limit: int = FINAL_RESULTS,
) -> list[Document]:
    results = retrieve_hybrid_candidates(question, tenant_id, limit)
    return [
        Document(
            page_content=document.page_content,
            metadata={
                **document.metadata,
                "retrieval_method": "opensearch_hybrid_rrf",
                "opensearch_hybrid_rank": rank,
                "opensearch_score": score,
            },
        )
        for rank, (document, score) in enumerate(results, start=1)
    ]


def _clean_queries(queries: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = " ".join(query.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)
        if len(cleaned) >= MAX_SEARCH_QUERIES:
            break
    return cleaned


def retrieve_documents_multi_query(
    queries: list[str],
    tenant_id: str = "default",
    limit: int = FINAL_RESULTS,
) -> list[Document]:
    """每条改写查询在 OpenSearch 内部混合检索，再跨查询做 RRF。"""
    cleaned_queries = _clean_queries(queries)
    if not cleaned_queries:
        return []

    fused: dict[tuple[str, str, str, str], dict] = {}
    for query in cleaned_queries:
        results = retrieve_hybrid_candidates(
            question=query,
            tenant_id=tenant_id,
            limit=RRF_CANDIDATES,
        )
        for rank, (document, score) in enumerate(results, start=1):
            key = get_document_key(document)
            item = fused.setdefault(
                key,
                {
                    "document": document,
                    "rrf_score": 0.0,
                    "matched_queries": set(),
                    "query_ranks": [],
                    "best_rank": 0,
                    "best_score": 0.0,
                },
            )
            item["rrf_score"] += 1.0 / (RRF_CONSTANT + rank)
            item["matched_queries"].add(query)
            item["query_ranks"].append(
                {
                    "query": query,
                    "method": "opensearch_hybrid",
                    "rank": rank,
                    "raw_score": score,
                }
            )
            if item["best_rank"] == 0 or rank < item["best_rank"]:
                item["best_rank"] = rank
                item["best_score"] = score

    ranked = sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )
    documents = []
    for rank, item in enumerate(ranked[:limit], start=1):
        document = item["document"]
        documents.append(
            Document(
                page_content=document.page_content,
                metadata={
                    **document.metadata,
                    "retrieval_method": "multi_query_opensearch_rrf",
                    "rrf_score": item["rrf_score"],
                    "rrf_rank": rank,
                    "matched_queries": sorted(item["matched_queries"]),
                    "query_ranks": item["query_ranks"],
                    "opensearch_hybrid_rank": item["best_rank"],
                    "opensearch_score": item["best_score"],
                },
            )
        )
    return documents
