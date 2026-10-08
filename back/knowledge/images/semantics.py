"""图片语义公共入口：组装视觉、缓存、DOCX 建库和图片候选匹配。"""
from __future__ import annotations

from functools import partial
import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from back.core.features import image_features_enabled
from back.core.llm import IMAGE_UNDERSTANDING_MODEL_NAME, get_image_understanding_model
from back.core.paths import PROJECT_ROOT
from back.knowledge.ingestion.docx.image_extractor import extract_docx_images
from back.knowledge.ingestion.docx.models import ExtractedImageRecord
from back.knowledge.images import semantic_cache, semantic_ingestion, semantic_matching, semantic_vision
from back.knowledge.images.semantic_cache import (
    PROMPT_VERSION, _context_sha256, _require_reliable_card, _safe_path_segment, _semantic_cache_path,
)
from back.knowledge.images.semantic_fingerprint import (
    MAX_VISION_EDGE, _calculate_dhash, _image_data_url, _open_normalized_image, dhash_distance, fingerprint_image,
)
from back.knowledge.images.semantic_formatting import format_customer_understanding, format_knowledge_card
from back.knowledge.images.semantic_matching import (
    DEFAULT_MAX_DETAIL_DHASH_DISTANCE, DEFAULT_MAX_DHASH_DISTANCE, MAX_ASPECT_RATIO_DELTA,
    MAX_MATCH_DOCUMENTS, _documents_from_store_data,
)
from back.knowledge.images.semantic_models import (
    MAX_CONTEXT_CHARS, CustomerImageUnderstanding, ImageFingerprint, ImageSemanticMatch,
    KnowledgeImageSemanticCard, _clean_text, _clean_text_list,
)
from back.knowledge.images.semantic_vision import _structured_invoke

IMAGE_SEMANTICS_ROOT = PROJECT_ROOT / "data" / "image_semantics"


def image_semantics_enabled() -> bool:
    return image_features_enabled() and os.getenv("IMAGE_SEMANTIC_ENABLED", "1").strip().lower() in {
        "1", "true", "yes", "on",
    }


def understand_knowledge_image(record: ExtractedImageRecord, *, ocr_text: str = "",
                               model: BaseChatModel | None = None) -> KnowledgeImageSemanticCard:
    return semantic_vision.understand_knowledge_image(record, ocr_text=ocr_text,
                                                     model=model or get_image_understanding_model())


def understand_customer_image(image_bytes: bytes, *, user_message: str = "",
                              model: BaseChatModel | None = None) -> CustomerImageUnderstanding:
    return semantic_vision.understand_customer_image(image_bytes, user_message=user_message,
                                                    model=model or get_image_understanding_model())


def get_or_create_knowledge_semantics(record: ExtractedImageRecord, *, tenant_id: str, source_id: str,
                                     ocr_text: str = "", output_root: str | Path = IMAGE_SEMANTICS_ROOT,
                                     model: BaseChatModel | None = None, force: bool = False,
                                     require_quality: bool = False) -> tuple[KnowledgeImageSemanticCard, ImageFingerprint, Path]:
    path = _semantic_cache_path(record, tenant_id, source_id, output_root)
    fingerprint = fingerprint_image(record.extracted_path)
    context_hash = _context_sha256(record, ocr_text)
    model_name = str(getattr(model, "model_name", "") or getattr(model, "model", "")
                     or IMAGE_UNDERSTANDING_MODEL_NAME or "unknown")
    card = None if force else semantic_cache.load_valid_card(path, fingerprint, context_hash, model_name, require_quality)
    if card is not None:
        return card, fingerprint, path
    card = understand_knowledge_image(record, ocr_text=ocr_text, model=model)
    if require_quality:
        _require_reliable_card(card)
    semantic_cache.write_card(path, card, fingerprint, record, tenant_id=tenant_id, source_id=source_id,
                              model_name=model_name, context_hash=context_hash, ocr_text=ocr_text)
    return card, fingerprint, path


def load_cached_image_semantic_documents(*, tenant_id: str, source_id: str, content_hash: str,
                                         starting_chunk_index: int, output_root: str | Path = IMAGE_SEMANTICS_ROOT) -> list[Document]:
    if not image_semantics_enabled():
        return []
    return semantic_cache.load_documents(tenant_id=tenant_id, source_id=source_id, content_hash=content_hash,
                                         starting_chunk_index=starting_chunk_index, output_root=output_root)


def build_docx_image_semantic_documents(*, document_path: str | Path, tenant_id: str, source_id: str,
                                        content_hash: str, starting_chunk_index: int,
                                        ocr_by_image_order: dict[int, str] | None = None,
                                        output_root: str | Path = IMAGE_SEMANTICS_ROOT,
                                        model: BaseChatModel | None = None, force: bool = False,
                                        require_all: bool = False) -> list[Document]:
    if not image_semantics_enabled():
        if require_all:
            raise RuntimeError("API-only 图片建库需要启用 IMAGE_SEMANTIC_ENABLED")
        return []
    process = partial(get_or_create_knowledge_semantics, tenant_id=tenant_id, source_id=source_id,
                      output_root=output_root, model=model, force=force, require_quality=require_all)
    return semantic_ingestion.build_image_documents(
        extract_docx_images(document_path), process=process, tenant_id=tenant_id, source_id=source_id,
        content_hash=content_hash, starting_chunk_index=starting_chunk_index,
        ocr_mapping=ocr_by_image_order or {}, require_all=require_all,
    )


def load_required_docx_image_semantic_documents(*, document_path: str | Path, tenant_id: str,
                                               source_id: str, content_hash: str, starting_chunk_index: int,
                                               output_root: str | Path = IMAGE_SEMANTICS_ROOT) -> list[Document]:
    if not image_semantics_enabled():
        raise RuntimeError("API-only 图片建库需要启用 IMAGE_SEMANTIC_ENABLED")
    indexed = {
        record.image_order for record in extract_docx_images(document_path)
        if semantic_cache.validate_required_card(record, tenant_id=tenant_id, source_id=source_id,
                                                 content_hash=content_hash, output_root=output_root,
                                                 model_name=IMAGE_UNDERSTANDING_MODEL_NAME or "unknown")
    }
    documents = load_cached_image_semantic_documents(
        tenant_id=tenant_id, source_id=source_id, content_hash=content_hash,
        starting_chunk_index=starting_chunk_index, output_root=output_root,
    )
    documents = [document for document in documents if document.metadata["image_order"] in indexed]
    if len(documents) != len(indexed):
        raise RuntimeError("图片语义缓存不完整，取消索引重建")
    for index, document in enumerate(documents, start=starting_chunk_index):
        document.metadata["chunk_index"] = index
    return documents


def _get_image_metadata_collection(tenant_id: str):
    from back.infrastructure.search.opensearch import get_vector_store
    return get_vector_store(tenant_id)


def find_matching_image_semantics(image_bytes: bytes, *, tenant_id: str = "default",
                                  max_dhash_distance: int | None = None) -> ImageSemanticMatch | None:
    if not image_semantics_enabled():
        return None
    fingerprint = fingerprint_image(image_bytes)
    return semantic_matching.find_match(_get_image_metadata_collection(tenant_id), fingerprint, max_dhash_distance)
