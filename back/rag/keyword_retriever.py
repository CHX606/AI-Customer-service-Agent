"""
  加载知识库
→ 加载splitter.py文件的切好的块
→ jieba中文分词
→ 建立BM25检索器
→ 根据关键词匹配分数返回Top 5 chunk
"""


from functools import lru_cache

import jieba
from langchain_community.retrievers import BM25Retriever

from back.rag.loader import load_documents
from back.rag.splitter import split_documents


MAX_RESULTS = 5


def tokenize_chinese(text: str) -> list[str]:
    """将中文文本切分成适合检索的关键词。"""

    words = jieba.lcut_for_search(
        text.lower(),
    )

    return [
        word.strip()
        for word in words
        if word.strip()
    ]


@lru_cache(maxsize=1)
def get_keyword_retriever():
    """创建并缓存 BM25 关键词检索器。"""

    documents = load_documents()
    chunks = split_documents(documents)

    retriever = BM25Retriever.from_documents(
        documents=chunks,
        preprocess_func=tokenize_chinese,
    )

    retriever.k = MAX_RESULTS

    return retriever


def retrieve_documents_by_keyword(question: str):
    """使用 BM25 检索与问题关键词最匹配的文本块。"""

    retriever = get_keyword_retriever()

    documents = retriever.invoke(question)

    return documents


if __name__ == "__main__":
    question = "续费之后为什么流量没有重置？"

    print("测试问题：", question)
    print("分词结果：", tokenize_chinese(question))

    documents = retrieve_documents_by_keyword(question)

    print("检索结果数量：", len(documents))

    for index, document in enumerate(
        documents,
        start=1,
    ):
        print(f"\n----- 第 {index} 个检索结果 -----")
        print(document.page_content)
        print("metadata：", document.metadata)