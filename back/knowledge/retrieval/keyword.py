"""OpenSearch 持久化 BM25 关键词检索兼容接口。"""

from dataclasses import dataclass

from langchain_core.documents import Document

from back.knowledge.retrieval.text import tokenize_chinese
from back.infrastructure.search.opensearch import get_vector_store
from back.domain.tenant import validate_tenant_id


MAX_RESULTS = 5


@dataclass(frozen=True)
class OpenSearchKeywordRetriever:
    tenant_id: str
    k: int = MAX_RESULTS

    def invoke(self, question: str) -> list[Document]:
        store = get_vector_store(self.tenant_id)
        return [
            document
            for document, _score in store.keyword_search_with_score(
                question, k=self.k
            )
        ]


def invalidate_keyword_retriever(tenant_id: str = "default") -> None:
    """兼容旧调用；OpenSearch 索引实时持久化，不再需要清内存缓存。"""
    validate_tenant_id(tenant_id)


def get_keyword_retriever(
    tenant_id: str = "default",
) -> OpenSearchKeywordRetriever:
    """返回基于 OpenSearch BM25 的检索器。"""
    return OpenSearchKeywordRetriever(validate_tenant_id(tenant_id))


def retrieve_documents_by_keyword(
    question: str,
    tenant_id: str = "default",
) -> list[Document]:
    return get_keyword_retriever(tenant_id).invoke(question)


if __name__ == "__main__":
    question = "续费之后为什么流量没有重置？"
    print("测试问题：", question)
    print("分词结果：", tokenize_chinese(question))
    for index, document in enumerate(
        retrieve_documents_by_keyword(question), start=1
    ):
        print(f"\n----- 第 {index} 个检索结果 -----")
        print(document.page_content)
        print("metadata：", document.metadata)
