"""统一加载文本知识块和图片诊断知识块。"""


from langchain_core.documents import Document

from back.rag.image_document_loader import (
    load_image_documents,
)
from back.rag.loader import load_documents
from back.rag.splitter import split_documents


def load_knowledge_documents() -> list[Document]:
    """
    返回供 Chroma 和 BM25 共用的完整知识库。

    两种检索器必须调用同一个入口，否则图片资料可能只存在于
    向量检索，却无法参与 BM25 和后续 RRF 融合。
    """

    word_documents = load_documents()
    text_chunks = split_documents(word_documents)
    image_documents = load_image_documents()

    return text_chunks + image_documents
