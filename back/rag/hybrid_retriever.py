"""
根据用户提出的问题
向量检索结果+ BM25检索结果
→ RRF融合
→ 得到该问题最终的Top 5 chunk
"""


from functools import lru_cache

from langchain_core.documents import Document

from back.rag.keyword_retriever import (
    get_keyword_retriever,
    tokenize_chinese,
)
from back.rag.vectorstore import get_vector_store


VECTOR_CANDIDATES = 10
KEYWORD_CANDIDATES = 10
FINAL_RESULTS = 5

VECTOR_WEIGHT = 1.0
KEYWORD_WEIGHT = 1.0

RRF_CONSTANT = 60


@lru_cache(maxsize=1)
def get_cached_vector_store():
    """创建并缓存向量数据库连接。"""

    return get_vector_store()


def get_document_key(
    document: Document,
):
    """生成用于识别同一个文本块的唯一标识。"""

    metadata = document.metadata

    return (
        str(metadata.get("source", "")),
        str(metadata.get("section_id", "")),
        str(metadata.get("chunk_index", "")),
    )


def retrieve_vector_candidates(
    question: str,
    limit: int = VECTOR_CANDIDATES,
):
    """获取向量检索候选结果及原始距离。"""

    vector_store = get_cached_vector_store()

    results = (
        vector_store.similarity_search_with_score(
            query=question,
            k=limit,
        )
    )

    return [
        (
            document,
            float(distance),
        )
        for document, distance in results
    ]


def retrieve_keyword_candidates(
    question: str,
    limit: int = KEYWORD_CANDIDATES,
):
    """获取BM25候选结果及原始分数。"""

    retriever = get_keyword_retriever()

    query_tokens = tokenize_chinese(
        question
    )

    scores = retriever.vectorizer.get_scores(
        query_tokens
    )

    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )

    results = []

    for index in ranked_indexes:
        score = float(scores[index])

        # 完全没有关键词匹配的文本不加入候选。
        if score <= 0:
            continue

        results.append(
            (
                retriever.docs[index],
                score,
            )
        )

        if len(results) >= limit:
            break

    return results


def fuse_results_with_rrf(
    vector_results,
    keyword_results,
    limit: int = FINAL_RESULTS,
):
    """使用RRF融合向量和BM25的排名结果。"""

    fused_items = {}

    for rank, (
        document,
        distance,
    ) in enumerate(
        vector_results,
        start=1,
    ):
        document_key = get_document_key(
            document
        )

        if document_key not in fused_items:
            fused_items[document_key] = {
                "document": document,
                "rrf_score": 0.0,
                "vector_rank": 0,
                "keyword_rank": 0,
                "vector_distance": -1.0,
                "bm25_score": -1.0,
            }

        item = fused_items[document_key]

        item["rrf_score"] += (
            VECTOR_WEIGHT
            / (RRF_CONSTANT + rank)
        )

        item["vector_rank"] = rank
        item["vector_distance"] = distance

    for rank, (
        document,
        bm25_score,
    ) in enumerate(
        keyword_results,
        start=1,
    ):
        document_key = get_document_key(
            document
        )

        if document_key not in fused_items:
            fused_items[document_key] = {
                "document": document,
                "rrf_score": 0.0,
                "vector_rank": 0,
                "keyword_rank": 0,
                "vector_distance": -1.0,
                "bm25_score": -1.0,
            }

        item = fused_items[document_key]

        item["rrf_score"] += (
            KEYWORD_WEIGHT
            / (RRF_CONSTANT + rank)
        )

        item["keyword_rank"] = rank
        item["bm25_score"] = bm25_score

    ranked_items = sorted(
        fused_items.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    documents = []

    for item in ranked_items[:limit]:
        original_document = item["document"]

        documents.append(
            Document(
                page_content=(
                    original_document.page_content
                ),
                metadata={
                    **original_document.metadata,
                    "retrieval_method": (
                        "hybrid_rrf"
                    ),
                    "rrf_score": (
                        item["rrf_score"]
                    ),
                    "vector_rank": (
                        item["vector_rank"]
                    ),
                    "keyword_rank": (
                        item["keyword_rank"]
                    ),
                    "vector_distance": (
                        item["vector_distance"]
                    ),
                    "bm25_score": (
                        item["bm25_score"]
                    ),
                },
            )
        )

    return documents


def retrieve_documents_hybrid(
    question: str,
    limit: int = FINAL_RESULTS,
):
    """使用向量检索和BM25进行混合检索。"""

    vector_results = (
        retrieve_vector_candidates(
            question=question,
        )
    )

    keyword_results = (
        retrieve_keyword_candidates(
            question=question,
        )
    )

    documents = fuse_results_with_rrf(
        vector_results=vector_results,
        keyword_results=keyword_results,
        limit=limit,
    )

    return documents


def show_results(
    question: str,
):
    """打印混合检索测试结果。"""

    print("\n" + "=" * 70)
    print("用户问题：", question)

    documents = retrieve_documents_hybrid(
        question
    )

    for rank, document in enumerate(
        documents,
        start=1,
    ):
        metadata = document.metadata

        section_id = metadata.get(
            "section_id",
            "未知",
        )

        section_title = metadata.get(
            "section_title",
            "未知",
        )

        rrf_score = metadata.get(
            "rrf_score",
            0.0,
        )

        vector_rank = metadata.get(
            "vector_rank",
            0,
        )

        keyword_rank = metadata.get(
            "keyword_rank",
            0,
        )

        vector_distance = metadata.get(
            "vector_distance",
            -1.0,
        )

        bm25_score = metadata.get(
            "bm25_score",
            -1.0,
        )

        print(
            f"\n第 {rank} 名："
            f"{section_id} {section_title}"
        )

        print(
            f"RRF分数：{rrf_score:.6f}"
        )

        print(
            f"向量排名：{vector_rank}，"
            f"向量距离：{vector_distance:.4f}"
        )

        print(
            f"BM25排名：{keyword_rank}，"
            f"BM25分数：{bm25_score:.4f}"
        )

        text = " ".join(
            document.page_content.split()
        )

        print(text[:300])


def main():
    """运行混合检索测试。"""

    questions = [
        (
            "套餐时间已经延长了，"
            "但是剩余流量还是没有恢复。"
        ),
        (
            "Windows上的Clash Verge"
            "已经连接，但还是打不开网页。"
        ),
        "我已经买了，怎么还是用不了？",
        "北京明天会下雨吗？",
    ]

    for question in questions:
        show_results(question)


if __name__ == "__main__":
    main()