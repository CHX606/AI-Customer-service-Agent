"""当前参考 DOCX 的图片诊断文档随文字一起索引的测试。"""

from datetime import datetime
from unittest.mock import Mock

from langchain_core.documents import Document
import pytest

import back.knowledge.indexing.service as indexing_service
from back.domain.tenant import KnowledgeSource


REFERENCE_HASH = "reference-document-hash"


def source(**updates) -> KnowledgeSource:
    values = {
        "source_id": "src_reference",
        "tenant_id": "default",
        "original_filename": "参考资料.docx",
        "stored_filename": "参考资料.docx",
        "file_type": "docx",
        "content_hash": REFERENCE_HASH,
        "status": "processing",
        "chunk_count": 0,
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
    }
    values.update(updates)
    return KnowledgeSource(**values)


def test_matching_default_docx_loads_and_enriches_reference_images(monkeypatch):
    monkeypatch.setattr(
        indexing_service,
        "DOCUMENT_PATH",
        Mock(exists=Mock(return_value=True)),
    )
    monkeypatch.setattr(indexing_service, "compute_file_hash", lambda _path: REFERENCE_HASH)
    image_documents = [
        Document(page_content="图片诊断A", metadata={"image_order": 7}),
        Document(page_content="图片诊断B", metadata={"image_order": 8}),
    ]
    loader = Mock(return_value=image_documents)
    monkeypatch.setattr(indexing_service, "load_image_documents", loader)

    results = indexing_service._load_matching_reference_images(
        source(),
        starting_chunk_index=27,
    )

    assert [item.page_content for item in results] == ["图片诊断A", "图片诊断B"]
    assert [item.metadata["chunk_index"] for item in results] == [27, 28]
    assert all(item.metadata["tenant_id"] == "default" for item in results)
    assert all(item.metadata["source_id"] == "src_reference" for item in results)
    assert all(item.metadata["content_hash"] == REFERENCE_HASH for item in results)
    loader.assert_called_once_with()


@pytest.mark.parametrize(
    "candidate",
    [
        source(tenant_id="tenant_b"),
        source(file_type="pdf"),
        source(content_hash="different-hash"),
    ],
)
def test_nonmatching_source_does_not_load_reference_images(monkeypatch, candidate):
    monkeypatch.setattr(
        indexing_service,
        "DOCUMENT_PATH",
        Mock(exists=Mock(return_value=True)),
    )
    monkeypatch.setattr(indexing_service, "compute_file_hash", lambda _path: REFERENCE_HASH)
    loader = Mock()
    monkeypatch.setattr(indexing_service, "load_image_documents", loader)

    assert indexing_service._load_matching_reference_images(candidate, 3) == []
    loader.assert_not_called()


def test_indexing_writes_text_and_reference_images_in_one_source(monkeypatch, tmp_path):
    current = source()
    ready = current.model_copy(update={"status": "ready", "chunk_count": 3})
    text_chunk = Document(
        page_content="文字块",
        metadata={"source_id": current.source_id, "chunk_index": 0},
    )
    image_chunks = [
        Document(page_content="图片A", metadata={"chunk_index": 1}),
        Document(page_content="图片B", metadata={"chunk_index": 2}),
    ]
    monkeypatch.setattr(
        indexing_service,
        "load_source_document",
        Mock(return_value=[Document(page_content="原文")]),
    )
    monkeypatch.setattr(indexing_service, "split_documents", Mock(return_value=[text_chunk]))
    image_loader = Mock(return_value=image_chunks)
    monkeypatch.setattr(indexing_service, "_load_matching_reference_images", image_loader)
    add = Mock(return_value=3)
    monkeypatch.setattr(indexing_service, "add_source_documents", add)
    monkeypatch.setattr(indexing_service, "invalidate_semantic_cache", Mock())
    monkeypatch.setattr(
        indexing_service,
        "update_knowledge_source_status",
        Mock(side_effect=[current, ready]),
    )

    result = indexing_service.index_knowledge_source(current, tmp_path / "source.docx")

    assert result.status == "ready"
    image_loader.assert_called_once_with(current, starting_chunk_index=1)
    add.assert_called_once_with(
        tenant_id="default",
        source_id="src_reference",
        documents=[text_chunk, *image_chunks],
    )


def test_contextual_loader_prefers_semantics_and_keeps_missing_ocr_fallback(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("IMAGE_SEMANTIC_ENABLED", "1")
    current = source()
    fallbacks = [
        Document(page_content="OCR 7", metadata={"image_order": 7}),
        Document(page_content="OCR 8", metadata={"image_order": 8}),
    ]
    semantics = [
        Document(
            page_content="语义 7 + 上下文",
            metadata={"image_order": 7, "content_type": "image_semantic"},
        )
    ]
    monkeypatch.setattr(
        indexing_service,
        "_load_matching_reference_images",
        Mock(return_value=fallbacks),
    )
    semantic_loader = Mock(return_value=semantics)
    monkeypatch.setattr(
        indexing_service,
        "build_docx_image_semantic_documents",
        semantic_loader,
    )

    results = indexing_service._load_contextual_image_documents(
        current,
        tmp_path / "source.docx",
        starting_chunk_index=27,
    )

    assert [document.page_content for document in results] == [
        "语义 7 + 上下文",
        "OCR 8",
    ]
    assert [document.metadata["chunk_index"] for document in results] == [27, 28]
    semantic_loader.assert_called_once()
    assert semantic_loader.call_args.kwargs["ocr_by_image_order"] == {
        7: "OCR 7",
        8: "OCR 8",
    }


def test_contextual_loader_uses_all_fallbacks_when_visual_model_returns_none(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("IMAGE_SEMANTIC_ENABLED", "1")
    fallbacks = [
        Document(page_content="OCR 7", metadata={"image_order": 7}),
        Document(page_content="OCR 8", metadata={"image_order": 8}),
    ]
    monkeypatch.setattr(
        indexing_service,
        "_load_matching_reference_images",
        Mock(return_value=fallbacks),
    )
    monkeypatch.setattr(
        indexing_service,
        "build_docx_image_semantic_documents",
        Mock(return_value=[]),
    )
    results = indexing_service._load_contextual_image_documents(
        source(), tmp_path / "source.docx", 10
    )
    assert [item.page_content for item in results] == ["OCR 7", "OCR 8"]
    assert [item.metadata["chunk_index"] for item in results] == [10, 11]


def test_contextual_loader_non_docx_never_runs_image_pipeline(monkeypatch, tmp_path):
    monkeypatch.setenv("IMAGE_SEMANTIC_ENABLED", "1")
    visual_loader = Mock()
    fallback_loader = Mock()
    monkeypatch.setattr(
        indexing_service,
        "build_docx_image_semantic_documents",
        visual_loader,
    )
    monkeypatch.setattr(
        indexing_service,
        "_load_matching_reference_images",
        fallback_loader,
    )
    assert indexing_service._load_contextual_image_documents(
        source(file_type="pdf"), tmp_path / "source.pdf", 4
    ) == []
    visual_loader.assert_not_called()
    fallback_loader.assert_not_called()
