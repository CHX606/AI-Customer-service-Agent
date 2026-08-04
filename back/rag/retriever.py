"""
这是向量检索文件，也是当前 tools.py 实际调用的检索器。

用户问题
→ 转成向量
→ 在向量数据库中查找相似chunk
→ 只保留相关度达到0.5的结果
→ 最多返回5个Document
"""


from back.rag.vectorstore import get_vector_store


MAX_RESULTS = 5
SCORE_THRESHOLD = 0.5


def get_retriever():
    """创建并返回知识库检索器。"""

    vector_store = get_vector_store()

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": MAX_RESULTS,
            "score_threshold": SCORE_THRESHOLD,
        },
    )

    return retriever


def retrieve_documents(question):
    """根据用户问题检索最相关的 Document。"""

    retriever = get_retriever()

    documents = retriever.invoke(question)

    return documents


if __name__ == "__main__":
    question = "如何添加一个新的客服账号？"

    documents = retrieve_documents(question)

    print("测试问题：", question)
    print("检索结果数量：", len(documents))

    for index, document in enumerate(documents, start=1):
        print(f"\n----- 第 {index} 个检索结果 -----")
        print(document.page_content)
        print("metadata：", document.metadata)