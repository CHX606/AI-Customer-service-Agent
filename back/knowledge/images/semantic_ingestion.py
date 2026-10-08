"""从预处理结果创建 DOCX 图片检索文档并管理批次失败。"""
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from langchain_core.documents import Document
from back.knowledge.images.semantic_formatting import format_knowledge_card
from back.knowledge.images.semantic_models import ImageFingerprint, KnowledgeImageSemanticCard
from back.knowledge.ingestion.docx.models import ExtractedImageRecord

logger = logging.getLogger(__name__)


def _image_document(record: ExtractedImageRecord, card: KnowledgeImageSemanticCard,
                    fingerprint: ImageFingerprint, cache_path: Path, ocr_text: str, *,
                    tenant_id: str, source_id: str, content_hash: str, chunk_index: int) -> Document:
    return Document(page_content=format_knowledge_card(card, record, ocr_text=ocr_text), metadata={
        "source": record.source, "document_name": record.document_name, "tenant_id": tenant_id,
        "source_id": source_id, "content_hash": content_hash, "content_type": "image_semantic",
        "image_order": record.image_order, "image_sha256": fingerprint.sha256,
        "image_dhash": fingerprint.dhash, "image_detail_dhash": fingerprint.detail_dhash,
        "image_width": fingerprint.width, "image_height": fingerprint.height,
        "image_type": card.image_type, "semantic_confidence": float(card.confidence),
        "section_id": record.section_id, "section_title": record.section_title, "block_index": record.block_index,
        "parent_block_index": record.block_index, "chunk_index": chunk_index,
        "full_image_path": record.extracted_path, "semantic_result_path": str(cache_path),
    })


def _batch_failure(error: Exception, record: ExtractedImageRecord, source_id: str, require_all: bool) -> bool:
    if require_all:
        raise RuntimeError(
            f"图片 {record.image_order} 的 API 预处理未完成（{type(error).__name__}），请重试建库"
        ) from error
    logger.exception("图片语义预处理失败，跳过当前图片",
                     extra={"source_id": source_id, "image_order": record.image_order})
    code = getattr(error, "status_code", None)
    stop = (isinstance(code, int) and code >= 400) or error.__class__.__module__.startswith(("openai", "httpx"))
    if stop:
        logger.error("视觉模型服务不可用，终止本批次后续图片调用并使用兜底资料")
    return stop


def build_image_documents(records: Iterable[ExtractedImageRecord], *, process: Callable,
                          tenant_id: str, source_id: str, content_hash: str, starting_chunk_index: int,
                          ocr_mapping: dict[int, str], require_all: bool) -> list[Document]:
    documents = []
    for record in records:
        ocr_text = ocr_mapping.get(record.image_order, "")
        try:
            card, fingerprint, path = process(record, ocr_text=ocr_text)
        except Exception as error:
            if _batch_failure(error, record, source_id, require_all):
                break
            continue
        if card.should_index:
            documents.append(_image_document(record, card, fingerprint, path, ocr_text,
                                             tenant_id=tenant_id, source_id=source_id, content_hash=content_hash,
                                             chunk_index=starting_chunk_index + len(documents)))
    return documents
