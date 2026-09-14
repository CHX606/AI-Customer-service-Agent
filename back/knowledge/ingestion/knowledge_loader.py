"""按租户加载统一知识库文档列表（供 OpenSearch 重建索引使用）。"""

from pathlib import Path
from langchain_core.documents import Document

from back.knowledge.images.documents import load_image_documents
from back.knowledge.images.semantics import load_cached_image_semantic_documents
from back.knowledge.ingestion.loader import load_documents, load_source_document
from back.knowledge.ingestion.splitter import split_documents, split_prebuilt_documents
from back.tenant.service import (
    get_all_knowledge_sources,
    get_tenant_upload_dir,
)


def _load_legacy_documents(tenant_id: str) -> list[Document]:
    """加载旧版 DOCX，并为图片块分配与文本块不冲突的稳定编号。"""
    word_documents = load_documents()
    text_chunks = [
        Document(
            page_content=document.page_content,
            metadata={**document.metadata, "tenant_id": tenant_id},
        )
        for document in split_documents(word_documents)
    ]
    if tenant_id != "default":
        return text_chunks

    base_metadata = word_documents[0].metadata if word_documents else {}
    raw_image_documents = []
    for document in load_image_documents():
        raw_image_documents.append(
            Document(
                page_content=document.page_content,
                metadata={
                    **document.metadata,
                    "tenant_id": tenant_id,
                    "source_id": str(
                        base_metadata.get("source_id", "legacy_kelecloud_docx")
                    ),
                    "content_hash": str(base_metadata.get("content_hash", "")),
                    "filename": str(base_metadata.get("filename", "")),
                    "chunk_index": document.metadata.get("chunk_index"),
                },
            )
        )
    image_documents = split_prebuilt_documents(
        raw_image_documents,
        starting_chunk_index=len(text_chunks),
    )
    return text_chunks + image_documents


def load_knowledge_documents(tenant_id: str = "default") -> list[Document]:
    """返回指定租户的完整可用知识库文档分块。"""
    sources = get_all_knowledge_sources(tenant_id)

    # 如果租户没有任何注册数据源且是 default 租户，则回退加载旧版可乐云知识库
    if not sources and tenant_id == "default":
        return _load_legacy_documents(tenant_id)

    all_chunks: list[Document] = []
    ready_sources = [source for source in sources if source.status == "ready"]
    ready_sources.sort(
        key=lambda source: source.source_id != "legacy_kelecloud_docx"
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
        if source.source_id == "legacy_kelecloud_docx":
            all_chunks.extend(_load_legacy_documents(tenant_id))
        else:
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
            cached_images = load_cached_image_semantic_documents(
                tenant_id=tenant_id,
                source_id=source.source_id,
                content_hash=source.content_hash,
                starting_chunk_index=len(file_chunks),
            )
            cached_image_chunks = split_prebuilt_documents(
                cached_images,
                starting_chunk_index=len(file_chunks),
            )
            all_chunks.extend(file_chunks + cached_image_chunks)

    return all_chunks
