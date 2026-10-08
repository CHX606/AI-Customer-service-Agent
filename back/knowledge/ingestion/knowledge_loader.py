"""按租户加载统一知识库文档列表（供 OpenSearch 重建索引使用）。"""

from pathlib import Path
from langchain_core.documents import Document
from back.core.features import image_features_enabled, local_ocr_enabled

from back.knowledge.images.semantics import load_cached_image_semantic_documents, load_required_docx_image_semantic_documents
from back.knowledge.ingestion.loader import PUBLIC_SOURCE_ID, load_documents, load_source_document
from back.knowledge.ingestion.splitter import split_documents, split_prebuilt_documents
from back.tenant.service import (
    get_all_knowledge_sources,
    get_tenant_upload_dir,
)


def _load_legacy_documents(tenant_id: str) -> list[Document]:
    """兼容旧调用名，空库只加载公开 Markdown 文本，不合并内部 DOCX 图片。"""
    return [
        Document(
            page_content=document.page_content,
            metadata={**document.metadata, "tenant_id": tenant_id},
        )
        for document in split_documents(load_documents())
    ]


def load_knowledge_documents(tenant_id: str = "default") -> list[Document]:
    """返回指定租户的完整可用知识库文档分块。"""
    sources = get_all_knowledge_sources(tenant_id)

    # 默认租户空库时只加载安全用户文档，其他租户不共享默认知识。
    if not sources and tenant_id == "default":
        return _load_legacy_documents(tenant_id)

    all_chunks: list[Document] = []
    ready_sources = [source for source in sources if source.status == "ready"]
    ready_sources.sort(
        key=lambda source: source.source_id != PUBLIC_SOURCE_ID
    )
    unique_sources = []
    seen_content_hashes: set[str] = set()
    for source in ready_sources:
        content_hash = source.content_hash.strip()
        if content_hash and content_hash in seen_content_hashes:
            continue
        if content_hash:
            seen_content_hashes.add(content_hash)
        unique_sources.append(source)

    for source in unique_sources:
        upload_dir = get_tenant_upload_dir(tenant_id, source.source_id)
        file_path = upload_dir / source.stored_filename
        if not file_path.exists():
            continue

        raw_docs = load_source_document(
            file_path,
            tenant_id=tenant_id,
            source_id=source.source_id,
            content_hash=source.content_hash,
        )
        file_chunks = split_documents(raw_docs)
        cache_loader = load_cached_image_semantic_documents
        cache_options = {}
        if image_features_enabled() and not local_ocr_enabled() and file_path.suffix.lower() == ".docx":
            cache_loader = load_required_docx_image_semantic_documents
            cache_options["document_path"] = file_path
        cached_images = cache_loader(
            tenant_id=tenant_id,
            source_id=source.source_id,
            content_hash=source.content_hash,
            starting_chunk_index=len(file_chunks),
            **cache_options,
        )
        cached_image_chunks = split_prebuilt_documents(
            cached_images,
            starting_chunk_index=len(file_chunks),
        )
        all_chunks.extend(file_chunks + cached_image_chunks)
    return all_chunks
