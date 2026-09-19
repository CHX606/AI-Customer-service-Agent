"""知识库数据源索引编排。

集中处理文档解析、切块、OpenSearch 混合索引写入与状态更新，避免
上传和重新索引接口各自维护一套容易漂移的流程。
"""

from pathlib import Path

from langchain_core.documents import Document
from back.core.features import image_features_enabled, local_ocr_enabled

from back.knowledge.images.documents import load_image_documents
from back.knowledge.images.semantics import (
    build_docx_image_semantic_documents,
    image_semantics_enabled,
)
from back.knowledge.ingestion.loader import DOCUMENT_PATH, compute_file_hash, load_source_document
from back.knowledge.retrieval.cache import invalidate_semantic_cache
from back.knowledge.ingestion.splitter import split_documents, split_prebuilt_documents
from back.infrastructure.search.opensearch import add_source_documents
from back.domain.tenant import KnowledgeSource
from back.domain.errors import NotFound
from back.infrastructure.operation_locks import source_operation, cache_mutation_lock
from back.tenant.service import update_knowledge_source_status


class KnowledgeIndexError(RuntimeError):
    """文档解析或索引失败。"""


def _load_matching_reference_images(
    source: KnowledgeSource,
    starting_chunk_index: int,
) -> list[Document]:
    """当前默认参考 DOCX 重索时，把已验证的图片诊断结果一并入库。"""
    if (
        not local_ocr_enabled()
        or source.tenant_id != "default"
        or source.file_type.lower() != "docx"
        or not DOCUMENT_PATH.exists()
        or source.content_hash != compute_file_hash(DOCUMENT_PATH)
    ):
        return []

    image_documents = load_image_documents()
    return [
        Document(
            page_content=document.page_content,
            metadata={
                **document.metadata,
                "tenant_id": source.tenant_id,
                "source_id": source.source_id,
                "content_hash": source.content_hash,
                "chunk_index": starting_chunk_index + index,
            },
        )
        for index, document in enumerate(image_documents)
    ]


def _load_contextual_image_documents(
    source: KnowledgeSource,
    file_path: Path,
    starting_chunk_index: int,
) -> list[Document]:
    """API-only 必须完整识别；本地 OCR 仅用于显式启用的混合模式。"""

    if not image_features_enabled() or source.file_type.lower() != "docx":
        return []

    fallback_documents = _load_matching_reference_images(
        source,
        starting_chunk_index=starting_chunk_index,
    ) if local_ocr_enabled() else []
    if not image_semantics_enabled():
        if not local_ocr_enabled():
            raise RuntimeError("API-only 图片建库需要启用 IMAGE_SEMANTIC_ENABLED")
        return fallback_documents

    ocr_by_order = {
        int(document.metadata["image_order"]): document.page_content
        for document in fallback_documents
        if "image_order" in document.metadata
    }
    semantic_documents = build_docx_image_semantic_documents(
        document_path=file_path,
        tenant_id=source.tenant_id,
        source_id=source.source_id,
        content_hash=source.content_hash,
        starting_chunk_index=starting_chunk_index,
        ocr_by_image_order=ocr_by_order,
        require_all=not local_ocr_enabled(),
    )

    semantic_orders = {
        int(document.metadata["image_order"])
        for document in semantic_documents
        if "image_order" in document.metadata
    }
    missing_fallbacks = [
        document
        for document in fallback_documents
        if int(document.metadata.get("image_order", -1)) not in semantic_orders
    ]
    combined = [*semantic_documents, *missing_fallbacks]

    # 跳过装饰图后序号可能不连续，统一重排，确保向量 ID 唯一且稳定。
    return [
        Document(
            page_content=document.page_content,
            metadata={
                **document.metadata,
                "chunk_index": starting_chunk_index + index,
            },
        )
        for index, document in enumerate(combined)
    ]


def index_knowledge_source(
    source: KnowledgeSource,
    file_path: Path,
) -> KnowledgeSource:
    with source_operation(source.tenant_id, source.source_id), cache_mutation_lock(source.tenant_id):
        return _index_knowledge_source(source, file_path)


def _index_knowledge_source(source: KnowledgeSource, file_path: Path) -> KnowledgeSource:
    """解析并索引一个数据源；失败时记录状态并向调用方抛出异常。"""
    # 先清除旧答案，避免重索期间继续返回即将过期的知识。
    invalidate_semantic_cache(source.tenant_id)
    if update_knowledge_source_status(source.source_id, "processing") is None:
        raise NotFound("知识源已删除，不能重新索引。")

    try:
        raw_documents = load_source_document(
            file_path,
            tenant_id=source.tenant_id,
            source_id=source.source_id,
            content_hash=source.content_hash,
        )
        chunks = split_documents(raw_documents)
        image_documents = _load_contextual_image_documents(
            source,
            file_path,
            starting_chunk_index=len(chunks),
        )
        chunks.extend(
            split_prebuilt_documents(
                image_documents,
                starting_chunk_index=len(chunks),
            )
        )
        if not chunks:
            raise ValueError("文档没有产生任何可索引文本分片")

        add_source_documents(
            tenant_id=source.tenant_id,
            source_id=source.source_id,
            documents=chunks,
        )
        updated = update_knowledge_source_status(
            source_id=source.source_id,
            status="ready",
            chunk_count=len(chunks),
        )
        if updated is None:
            raise RuntimeError("索引完成后无法更新数据源状态")
        return updated
    except Exception as error:
        update_knowledge_source_status(
            source_id=source.source_id,
            status="failed",
            error_message=str(error),
        )
        raise KnowledgeIndexError(str(error)) from error
