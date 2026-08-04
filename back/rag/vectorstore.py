"""
get_vector_store():连接本地Chroma向量数据库，供检索器查询。

rebuild_vector_store():
重新构建数据库：
加载Word文档
→ 结构化切块
→ Embedding转成向量
→ 清空旧集合
→ 将新向量存入Chroma

通常修改知识库或切块规则后，才需要重新构建。
"""

from pathlib import Path

from langchain_chroma import Chroma

from back.rag.loader import load_documents
from back.rag.splitter import split_documents
from back.rag.embeddings import get_embedding_model


# 当前文件夹：back/rag
CURRENT_DIR = Path(__file__).resolve().parent

# 项目根目录：AI Customer service Agent
PROJECT_ROOT = CURRENT_DIR.parent.parent

# 向量数据库保存位置：data/chroma_db
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"

# Chroma 中的集合名称
COLLECTION_NAME = "customer_service_knowledge"


def get_vector_store():
    """连接本地 Chroma 向量数据库。"""

    embedding_model = get_embedding_model()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_PATH),
    )

    return vector_store


def rebuild_vector_store():
    """根据知识库文档重新构建向量数据库。"""

    documents = load_documents()
    chunks = split_documents(documents)

    vector_store = get_vector_store()

    # 清空旧集合，防止重复保存相同文本块
    vector_store.reset_collection()

    # 将文本块转换成向量并存入 Chroma
    vector_store.add_documents(
        documents=chunks
    )

    return vector_store


if __name__ == "__main__":
    vector_store = rebuild_vector_store()

    stored_data = vector_store.get()
    stored_count = len(stored_data["ids"])

    print("向量数据库构建成功")
    print("数据库位置：", CHROMA_PATH)
    print("保存的文本块数量：", stored_count)