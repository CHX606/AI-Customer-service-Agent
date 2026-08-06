"""
混合检索候选结果精排模块。

Reranker同时读取用户问题和候选chunk，计算两者的真实相关程度，
再从RRF宽召回结果中选出最终Top K。
"""

from functools import lru_cache

import torch
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


MODEL_NAME = "BAAI/bge-reranker-base"
FINAL_RESULTS = 5
MAX_LENGTH = 512
BATCH_SIZE = 8


@lru_cache(maxsize=1)
def get_reranker_model() -> CrossEncoder:
    """创建并缓存CrossEncoder精排模型。"""

    return CrossEncoder(
        model_name=MODEL_NAME,
        max_length=MAX_LENGTH,
    )


def build_document_text(document: Document) -> str:
    """整理供Reranker判断的标题和正文。"""

    section_id = document.metadata.get(
        "section_id",
        "",
    )

    section_title = document.metadata.get(
        "section_title",
        "",
    )

    return (
        f"章节：{section_id} {section_title}\n"
        f"内容：{document.page_content}"
    )


def rerank_documents(
    query: str,
    documents: list[Document],
    limit: int = FINAL_RESULTS,
) -> list[Document]:
    """根据问题与chunk的匹配程度重新排序候选资料。"""

    if not documents:
        return []

    reranker = get_reranker_model()

    query_document_pairs = [
        (
            query,
            build_document_text(document),
        )
        for document in documents
    ]

    scores = reranker.predict(
        query_document_pairs,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        activation_fn=torch.nn.Sigmoid(),
    )

    scored_documents = [
        (
            document,
            float(score),
        )
        for document, score in zip(
            documents,
            scores,
            strict=True,
        )
    ]

    scored_documents.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    reranked_documents = []

    for rank, (
        document,
        score,
    ) in enumerate(
        scored_documents[:limit],
        start=1,
    ):
        reranked_documents.append(
            Document(
                page_content=document.page_content,
                metadata={
                    **document.metadata,
                    "retrieval_method": (
                        "hybrid_rrf_then_reranker"
                    ),
                    "reranker_rank": rank,
                    "reranker_score": score,
                },
            )
        )

    return reranked_documents
