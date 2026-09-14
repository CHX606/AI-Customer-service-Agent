"""通用多格式知识库文档加载器。

支持解析格式：
1. DOCX
2. TXT
3. Markdown (.md)
4. PDF

自动为加载的 Document 注入租户、数据源与哈希元数据。
"""

import hashlib
from pathlib import Path
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document
import pypdfium2 as pdfium

from back.core.paths import KNOWLEDGE_RESOURCES_DIR


DOCUMENT_PATH = KNOWLEDGE_RESOURCES_DIR / "可乐云客服操作文档.docx"


def compute_file_hash(path: Path | str) -> str:
    """计算文件的 SHA-256 哈希值。"""
    hasher = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _read_text_file(path: Path) -> str:
    """尝试以 UTF-8 或 GBK 编码读取纯文本内容。"""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="gbk", errors="ignore")


def _read_pdf_file(path: Path) -> str:
    """使用 pypdfium2 提取 PDF 页面正文，并确保资源完全释放。"""
    pdf = pdfium.PdfDocument(str(path))
    try:
        pages_text = []
        for page in pdf:
            try:
                textpage = page.get_textpage()
                try:
                    text = textpage.get_text_range()
                    if text.strip():
                        pages_text.append(text.strip())
                finally:
                    textpage.close()
            finally:
                page.close()
        return "\n\n".join(pages_text)
    finally:
        pdf.close()


def load_source_document(
    path: Path | str,
    *,
    tenant_id: str = "default",
    source_id: str = "legacy_kelecloud_docx",
    content_hash: str | None = None,
) -> list[Document]:
    """加载任意支持格式的知识库文档并统一包装为 LangChain Document。"""
    file_path = Path(path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"没有找到知识库文件：{file_path}")

    actual_hash = content_hash or compute_file_hash(file_path)
    extension = file_path.suffix.lower()

    if extension == ".docx":
        loader = Docx2txtLoader(str(file_path))
        raw_docs = loader.load()
        content = "\n\n".join(doc.page_content for doc in raw_docs)
    elif extension == ".pdf":
        content = _read_pdf_file(file_path)
    elif extension in {".txt", ".md", ".markdown"}:
        content = _read_text_file(file_path)
    else:
        raise ValueError(f"不支持的文档类型：{extension}")

    base_metadata = {
        "tenant_id": tenant_id,
        "source_id": source_id,
        "filename": file_path.name,
        "content_hash": actual_hash,
        "source": str(file_path),
    }

    return [
        Document(
            page_content=content,
            metadata=base_metadata,
        )
    ]


def load_documents() -> list[Document]:
    """加载默认的可乐云 Word 操作文档（向后兼容）。"""
    return load_source_document(
        DOCUMENT_PATH,
        tenant_id="default",
        source_id="legacy_kelecloud_docx",
    )


if __name__ == "__main__":
    documents = load_documents()
    print("文档加载成功")
    print("Document 数量：", len(documents))
    if documents:
        print("\n第一份 Document 的部分内容：")
        print(documents[0].page_content[:500])
