"""默认知识库检索接口，使用 OpenSearch 原生混合检索。"""

from langchain_core.documents import Document

from back.knowledge.retrieval.hybrid import retrieve_documents_hybrid


MAX_RESULTS = 5


def retrieve_documents(
    question: str,
    tenant_id: str = "default",
) -> list[Document]:
    return retrieve_documents_hybrid(
        question=question,
        tenant_id=tenant_id,
        limit=MAX_RESULTS,
    )


if __name__ == "__main__":
    for index, document in enumerate(
        retrieve_documents("如何添加一个新的客服账号？"), start=1
    ):
        print(f"\n----- 第 {index} 个检索结果 -----")
        print(document.page_content)
        print("metadata：", document.metadata)
