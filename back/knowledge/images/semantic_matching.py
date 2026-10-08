"""从知识图元数据筛选精确或感知相似的图片候选。"""
import os
from langchain_core.documents import Document
from back.knowledge.images.semantic_fingerprint import dhash_distance
from back.knowledge.images.semantic_models import ImageFingerprint, ImageSemanticMatch

MAX_MATCH_DOCUMENTS = 3
DEFAULT_MAX_DHASH_DISTANCE = 4
DEFAULT_MAX_DETAIL_DHASH_DISTANCE = 24
MAX_ASPECT_RATIO_DELTA = 0.02

def _documents_from_store_data(data: dict) -> list[Document]:
    raw_documents = data.get("documents") or []
    raw_metadatas = data.get("metadatas") or []
    return [
        Document(page_content=str(content or ""), metadata=dict(metadata or {}))
        for content, metadata in zip(raw_documents, raw_metadatas, strict=False)
        if content and isinstance(metadata, dict)
    ]


def _candidate(document: Document, fingerprint: ImageFingerprint) -> tuple[int, int, Document] | None:
    metadata = document.metadata
    try:
        distance = dhash_distance(fingerprint.dhash, str(metadata.get("image_dhash", "")))
        detail_distance = dhash_distance(fingerprint.detail_dhash, str(metadata.get("image_detail_dhash", "")))
        width, height = int(metadata.get("image_width", 0)), int(metadata.get("image_height", 0))
    except ValueError:
        return None
    if width <= 0 or height <= 0:
        return None
    if abs(fingerprint.width / fingerprint.height - width / height) > MAX_ASPECT_RATIO_DELTA:
        return None
    return distance, detail_distance, document


def _perceptual_match(documents: list[Document], fingerprint: ImageFingerprint,
                      max_dhash_distance: int | None) -> ImageSemanticMatch | None:
    candidates = [candidate for document in documents if (candidate := _candidate(document, fingerprint)) is not None]
    if not candidates:
        return None
    allowed = (max_dhash_distance if max_dhash_distance is not None else
               int(os.getenv("IMAGE_SEMANTIC_MATCH_MAX_DISTANCE", str(DEFAULT_MAX_DHASH_DISTANCE))))
    detail_allowed = int(os.getenv("IMAGE_SEMANTIC_DETAIL_MATCH_MAX_DISTANCE", str(DEFAULT_MAX_DETAIL_DHASH_DISTANCE)))
    candidates.sort(key=lambda item: (item[0], item[1]))
    best, best_detail, _ = candidates[0]
    if best > max(0, min(64, allowed)) or best_detail > max(0, min(1024, detail_allowed)):
        return None
    matches = [document for distance, detail, document in candidates if distance == best and detail == best_detail]
    return ImageSemanticMatch(strategy="perceptual_dhash", distance=best, documents=tuple(matches[:MAX_MATCH_DOCUMENTS]))


def find_match(collection, fingerprint: ImageFingerprint, max_dhash_distance: int | None) -> ImageSemanticMatch | None:
    exact = _documents_from_store_data(collection.get(
        where={"image_sha256": {"$eq": fingerprint.sha256}}, include=["documents", "metadatas"],
    ))
    if exact:
        return ImageSemanticMatch(strategy="exact_sha256", distance=0, documents=tuple(exact[:MAX_MATCH_DOCUMENTS]))
    documents = _documents_from_store_data(collection.get(
        where={"content_type": {"$in": ["image_semantic", "image_diagnostic"]}}, include=["documents", "metadatas"],
    ))
    return _perceptual_match(documents, fingerprint, max_dhash_distance)
