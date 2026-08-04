"""
找到Word知识库文件
→ 读取其中的文字
→ 转换成LangChain的Document对象
→ 交给splitter.py进行切块

该文件只负责加载原始文档，不负责切块、向量化或检索。
"""


from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader


# 当前文件所在目录：back/rag
CURRENT_DIR = Path(__file__).resolve().parent

# Word 文件路径：back/knowledge/可乐云客服操作文档.docx
DOCUMENT_PATH = (
    CURRENT_DIR.parent
    / "knowledge"
    / "可乐云客服操作文档.docx"
)


def load_documents():
    """加载 Word 文档并返回 Document 列表。"""

    if not DOCUMENT_PATH.exists():
        raise FileNotFoundError(
            f"没有找到知识库文件：{DOCUMENT_PATH}"
        )

    loader = Docx2txtLoader(str(DOCUMENT_PATH))
    documents = loader.load()

    return documents


if __name__ == "__main__":
    documents = load_documents()

    print("文档加载成功")
    print("Document 数量：", len(documents))

    if documents:
        print("\n第一份 Document 的部分内容：")
        print(documents[0].page_content[:500])