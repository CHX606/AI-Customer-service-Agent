"""OpenSearch BM25 结果必须保留图片语义卡片。"""

from langchain_core.documents import Document

from back.knowledge.retrieval import keyword as keyword_retriever


class _OpenSearchStore:
    def keyword_search_with_score(self, question: str, k: int):
        assert question == "Clash TLS"
        assert k == keyword_retriever.MAX_RESULTS
        return [
            (
                Document(
                    page_content="图片含义 Clash TLS 导入失败",
                    metadata={
                        "content_type": "image_semantic",
                        "image_order": 9,
                    },
                ),
                8.5,
            )
        ]


def test_keyword_retriever_keeps_image_semantics(monkeypatch):
    monkeypatch.setattr(
        keyword_retriever,
        "get_vector_store",
        lambda _tenant_id: _OpenSearchStore(),
    )

    results = keyword_retriever.get_keyword_retriever("default").invoke(
        "Clash TLS"
    )
    assert results[0].metadata["content_type"] == "image_semantic"
