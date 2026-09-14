"""测试多格式文档加载（DOCX, PDF, TXT, MD）与自适应切块能力。"""

import shutil
from pathlib import Path
from langchain_core.documents import Document

from back.knowledge.ingestion.loader import DOCUMENT_PATH, load_documents, load_source_document
from back.knowledge.ingestion.splitter import split_documents


def _create_test_pdf_file(pdf_path: Path, text: str) -> None:
    """生成一个标准合规的单页 PDF 文件供自动化测试。"""
    raw_stream = f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET"
    stream_len = len(raw_stream.encode("latin-1"))
    pdf_text = (
        "%PDF-1.4\n"
        "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        f"4 0 obj\n<< /Length {stream_len} >>\nstream\n"
        f"{raw_stream}\n"
        "endstream\nendobj\n"
        "5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        "xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000244 00000 n \n0000000300 00000 n \n"
        "trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n380\n%%EOF\n"
    )
    pdf_path.write_bytes(pdf_text.encode("latin-1"))


def test_load_and_split_pdf(tmp_path: Path):
    """测试解析真实 PDF 二进制文档并自适应切块。"""
    pdf_path = tmp_path / "manual.pdf"
    expected_text = "KeleCloud Knowledge Base Guide for PDF"
    _create_test_pdf_file(pdf_path, expected_text)

    docs = load_source_document(
        pdf_path,
        tenant_id="tenant_pdf_test",
        source_id="src_pdf_01",
    )

    assert len(docs) == 1
    assert expected_text in docs[0].page_content
    assert docs[0].metadata["tenant_id"] == "tenant_pdf_test"
    assert docs[0].metadata["source_id"] == "src_pdf_01"

    chunks = split_documents(docs)
    assert len(chunks) >= 1
    assert chunks[0].metadata["tenant_id"] == "tenant_pdf_test"
    assert chunks[0].metadata["source_id"] == "src_pdf_01"
    assert expected_text in chunks[0].page_content


def test_load_real_production_docx():
    """测试读取工程内真实的可乐云 DOCX 操作文档。"""
    docs = load_documents()
    assert len(docs) == 1
    assert docs[0].metadata["tenant_id"] == "default"
    assert "可乐云" in docs[0].page_content

    chunks = split_documents(docs)
    assert len(chunks) >= 20
    assert all(chunk.metadata["tenant_id"] == "default" for chunk in chunks)


def test_load_docx_from_custom_path(tmp_path: Path):
    """测试从自定义路径加载 DOCX 文件。"""
    custom_docx = tmp_path / "custom_service.docx"
    shutil.copy(DOCUMENT_PATH, custom_docx)

    docs = load_source_document(
        custom_docx,
        tenant_id="tenant_custom",
        source_id="src_docx_01",
    )
    assert len(docs) == 1
    assert docs[0].metadata["tenant_id"] == "tenant_custom"
    assert docs[0].metadata["source_id"] == "src_docx_01"

    chunks = split_documents(docs)
    assert len(chunks) >= 20
    assert chunks[0].metadata["tenant_id"] == "tenant_custom"


def test_load_and_split_txt(tmp_path: Path):
    txt_file = tmp_path / "faq.txt"
    txt_file.write_text(
        "退款说明\n\n用户在购买后3天内未产生超过1GB流量的，支持全额原路退款。\n逾期或流量超出则不支持退款。",
        encoding="utf-8",
    )

    docs = load_source_document(
        txt_file,
        tenant_id="tenant_a",
        source_id="src_txt_01",
    )
    assert len(docs) == 1
    assert docs[0].metadata["tenant_id"] == "tenant_a"
    assert docs[0].metadata["source_id"] == "src_txt_01"

    chunks = split_documents(docs)
    assert len(chunks) >= 1
    assert chunks[0].metadata["tenant_id"] == "tenant_a"
    assert "退款" in chunks[0].page_content


def test_load_and_split_markdown(tmp_path: Path):
    md_file = tmp_path / "guide.md"
    md_file.write_text(
        "# 1.0 快速入门\n欢迎使用本平台。\n\n# 2.0 节点配置\n请在客户端中导入您的订阅链接。",
        encoding="utf-8",
    )

    docs = load_source_document(
        md_file,
        tenant_id="tenant_b",
        source_id="src_md_01",
    )
    assert len(docs) == 1

    chunks = split_documents(docs)
    assert len(chunks) >= 1
    assert chunks[0].metadata["tenant_id"] == "tenant_b"


def test_fallback_unstructured_document():
    doc = Document(
        page_content="这是一段没有任何章节编号或标题标记的自由格式文本内容。" * 20,
        metadata={"tenant_id": "test_tenant", "source_id": "src_unstructured"},
    )
    chunks = split_documents([doc])
    assert len(chunks) >= 1
    assert chunks[0].metadata["section_id"] == "通用"
    assert chunks[0].metadata["chunk_index"] == 0
