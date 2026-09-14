"""向量数据源新增、同版本更新、过期删除与失败回滚测试。"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.documents import Document
import pytest

import back.knowledge.ingestion.knowledge_loader as knowledge_loader
import back.infrastructure.search.opensearch as vectorstore


SOURCE_ID = "src_update"
CONTENT_HASH = "abcdef1234567890"
VERSION = CONTENT_HASH[:12]


def document(index: int, text: str | None = None) -> Document:
    return Document(
        page_content=text or f"正文{index}",
        metadata={
            "source_id": SOURCE_ID,
            "content_hash": CONTENT_HASH,
            "chunk_index": index,
        },
    )


def chunk_id(index: int) -> str:
    return f"{SOURCE_ID}_{VERSION}_{index}"


def fake_store(existing_ids):
    store = Mock()
    store.get_source_ids.return_value = list(existing_ids)
    return store


def test_new_source_adds_all_documents(monkeypatch):
    store = fake_store([])
    docs = [document(0), document(1)]

    with patch.object(vectorstore, "get_vector_store", return_value=store):
        assert vectorstore.add_source_documents("default", SOURCE_ID, docs) == 2

    store.add_documents.assert_called_once_with(
        docs,
        [chunk_id(0), chunk_id(1)],
    )
    store.update_documents.assert_not_called()
    store.delete.assert_not_called()


def test_same_hash_reindex_updates_existing_documents_in_place(monkeypatch):
    store = fake_store([chunk_id(0), chunk_id(1)])
    docs = [document(0, "新正文0"), document(1, "新正文1")]

    with patch.object(vectorstore, "get_vector_store", return_value=store):
        vectorstore.add_source_documents("default", SOURCE_ID, docs)

    store.add_documents.assert_not_called()
    store.update_documents.assert_called_once_with(
        docs,
        [chunk_id(0), chunk_id(1)],
    )
    store.delete.assert_not_called()


def test_reindex_adds_new_updates_retained_and_deletes_stale(monkeypatch):
    stale_id = chunk_id(99)
    store = fake_store([chunk_id(0), stale_id])
    docs = [document(0, "更新正文"), document(1, "新增图片诊断")]

    with patch.object(vectorstore, "get_vector_store", return_value=store):
        vectorstore.add_source_documents("default", SOURCE_ID, docs)

    store.add_documents.assert_called_once_with(
        [docs[1]],
        [chunk_id(1)],
    )
    store.update_documents.assert_called_once_with(
        [docs[0]],
        [chunk_id(0)],
    )
    store.delete.assert_called_once_with([stale_id])


def test_duplicate_chunk_ids_are_rejected_before_writing(monkeypatch):
    store = fake_store([])

    with patch.object(vectorstore, "get_vector_store", return_value=store):
        with pytest.raises(ValueError, match="重复"):
            vectorstore.add_source_documents(
                "default",
                SOURCE_ID,
                [document(0, "A"), document(0, "B")],
            )

    store.add_documents.assert_not_called()
    store.update_documents.assert_not_called()


def test_update_failure_rolls_back_newly_added_ids(monkeypatch):
    store = fake_store([chunk_id(0)])
    store.update_documents.side_effect = RuntimeError("update failed")

    with patch.object(vectorstore, "get_vector_store", return_value=store):
        with pytest.raises(RuntimeError, match="update failed"):
            vectorstore.add_source_documents(
                "default",
                SOURCE_ID,
                [document(0), document(1)],
            )

    store.delete.assert_called_once_with([chunk_id(1)])


def test_add_failure_attempts_to_rollback_new_ids(monkeypatch):
    store = fake_store([])
    store.add_documents.side_effect = RuntimeError("add failed")

    with patch.object(vectorstore, "get_vector_store", return_value=store):
        with pytest.raises(RuntimeError, match="add failed"):
            vectorstore.add_source_documents(
                "default",
                SOURCE_ID,
                [document(0), document(1)],
            )

    store.delete.assert_called_once_with([chunk_id(0), chunk_id(1)])


def test_empty_document_list_does_not_open_vector_store(monkeypatch):
    factory = Mock()
    with patch.object(vectorstore, "get_vector_store", factory):
        assert vectorstore.add_source_documents("default", SOURCE_ID, []) == 0
    factory.assert_not_called()


def test_legacy_loader_assigns_unique_indexes_and_source_metadata_to_images():
    raw = Document(
        page_content="原文",
        metadata={
            "tenant_id": "default",
            "source_id": "legacy_kelecloud_docx",
            "content_hash": "hash-v1",
            "filename": "knowledge.docx",
        },
    )
    text_chunks = [document(0, "文本0"), document(1, "文本1")]
    images = [
        Document(page_content="图片0", metadata={"chunk_index": 0}),
        Document(page_content="图片1", metadata={"chunk_index": 0}),
    ]

    with (
        patch.object(knowledge_loader, "load_documents", return_value=[raw]),
        patch.object(knowledge_loader, "split_documents", return_value=text_chunks),
        patch.object(knowledge_loader, "load_image_documents", return_value=images),
    ):
        loaded = knowledge_loader._load_legacy_documents("default")

    assert [item.metadata["chunk_index"] for item in loaded] == [0, 1, 2, 3]
    assert all(
        item.metadata["source_id"] == "legacy_kelecloud_docx"
        for item in loaded[2:]
    )
    assert all(item.metadata["content_hash"] == "hash-v1" for item in loaded[2:])


def test_non_default_legacy_loader_keeps_tenant_isolation_and_skips_images():
    raw = Document(page_content="原文", metadata={"tenant_id": "default"})
    chunks = [Document(page_content="文本", metadata={"chunk_index": 0})]

    with (
        patch.object(knowledge_loader, "load_documents", return_value=[raw]),
        patch.object(knowledge_loader, "split_documents", return_value=chunks),
        patch.object(knowledge_loader, "load_image_documents") as image_loader,
    ):
        loaded = knowledge_loader._load_legacy_documents("tenant_a")

    assert loaded[0].metadata["tenant_id"] == "tenant_a"
    image_loader.assert_not_called()


def test_knowledge_loader_deduplicates_sources_with_same_content_hash():
    sources = [
        SimpleNamespace(
            source_id="src_uploaded_copy",
            status="ready",
            content_hash="same-hash",
            stored_filename="knowledge.docx",
        ),
        SimpleNamespace(
            source_id="legacy_kelecloud_docx",
            status="ready",
            content_hash="same-hash",
            stored_filename="knowledge.docx",
        ),
    ]
    expected = [document(0, "唯一正文")]

    with (
        patch.object(
            knowledge_loader,
            "get_all_knowledge_sources",
            return_value=sources,
        ),
        patch.object(
            knowledge_loader,
            "_load_legacy_documents",
            return_value=expected,
        ) as legacy_loader,
        patch.object(knowledge_loader, "load_source_document") as uploaded_loader,
    ):
        loaded = knowledge_loader.load_knowledge_documents("default")

    assert loaded == expected
    legacy_loader.assert_called_once_with("default")
    uploaded_loader.assert_not_called()


def test_prebuilt_image_documents_are_split_below_reranker_budget():
    image = Document(
        page_content="图片诊断内容。" * 100,
        metadata={
            "section_id": "4.3",
            "section_title": "导入",
            "chunk_index": 27,
        },
    )

    chunks = knowledge_loader.split_prebuilt_documents(
        [image],
        starting_chunk_index=10,
    )

    assert len(chunks) > 1
    assert [item.metadata["chunk_index"] for item in chunks] == list(
        range(10, 10 + len(chunks))
    )
    assert all(len(item.page_content) <= 420 for item in chunks)
    assert all(item.metadata["parent_chunk_index"] == 27 for item in chunks)
