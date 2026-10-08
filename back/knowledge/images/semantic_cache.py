"""图片语义缓存的原子写入、读取和重建前校验。"""
from hashlib import sha256
import json
import logging
from pathlib import Path
import re
from langchain_core.documents import Document
from back.knowledge.images.semantic_fingerprint import fingerprint_image
from back.knowledge.images.semantic_formatting import format_knowledge_card
from back.knowledge.images.semantic_models import ImageFingerprint, KnowledgeImageSemanticCard
from back.knowledge.ingestion.docx.models import ExtractedImageRecord

PROMPT_VERSION = "image-semantic-v1"
logger = logging.getLogger(__name__)

def _safe_path_segment(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")
    return cleaned[:80] or sha256(value.encode("utf-8")).hexdigest()[:16]

def _semantic_cache_path(
    record: ExtractedImageRecord,
    tenant_id: str,
    source_id: str,
    output_root: str | Path,
) -> Path:
    return (
        Path(output_root).resolve()
        / _safe_path_segment(tenant_id)
        / _safe_path_segment(source_id)
        / record.document_sha256[:12]
        / f"image_{record.image_order:03d}.json"
    )

def _context_sha256(record: ExtractedImageRecord, ocr_text: str) -> str:
    payload = "\n".join(
        [
            record.section_id,
            record.section_title,
            record.previous_text,
            record.next_text,
            ocr_text,
        ]
    )
    return sha256(payload.encode("utf-8")).hexdigest()

def _require_reliable_card(card: KnowledgeImageSemanticCard) -> None:
    """Do not publish empty/uncertain cards as completed API-only knowledge."""
    if not card.summary.strip() or card.confidence < 0.35:
        raise ValueError("图片语义结果不完整或置信度不足")
    if card.should_index and not card.visible_evidence:
        raise ValueError("图片语义结果缺少可见证据")


def load_valid_card(path: Path, fingerprint: ImageFingerprint, context_hash: str,
                    model_name: str, require_quality: bool) -> KnowledgeImageSemanticCard | None:
    if not path.exists():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        expected = {"status": "completed", "image_sha256": fingerprint.sha256,
                    "context_sha256": context_hash, "model": model_name, "prompt_version": PROMPT_VERSION}
        if any(cached.get(key) != value for key, value in expected.items()):
            return None
        card = KnowledgeImageSemanticCard.model_validate(cached["semantic_card"])
        if require_quality:
            _require_reliable_card(card)
        return card
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        logger.warning("图片语义缓存无效，将重新生成：%s", path)
        return None


def write_card(path: Path, card: KnowledgeImageSemanticCard, fingerprint: ImageFingerprint,
               record: ExtractedImageRecord, *, tenant_id: str, source_id: str,
               model_name: str, context_hash: str, ocr_text: str) -> None:
    data = {
        "status": "completed", "prompt_version": PROMPT_VERSION, "model": model_name,
        "tenant_id": tenant_id, "source_id": source_id, "document_sha256": record.document_sha256,
        "image_order": record.image_order, "image_sha256": fingerprint.sha256,
        "image_dhash": fingerprint.dhash, "image_detail_dhash": fingerprint.detail_dhash,
        "image_width": fingerprint.width, "image_height": fingerprint.height,
        "context_sha256": context_hash, "semantic_card": card.model_dump(mode="json"),
        "page_content": format_knowledge_card(card, record, ocr_text=ocr_text),
        "document_name": record.document_name, "full_image_path": record.extracted_path,
        "section_id": record.section_id, "section_title": record.section_title, "block_index": record.block_index,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def cached_document(path: Path, *, tenant_id: str, source_id: str, content_hash: str,
                    chunk_index: int) -> Document | None:
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        card = KnowledgeImageSemanticCard.model_validate(cached["semantic_card"])
        expected = {"status": "completed", "prompt_version": PROMPT_VERSION, "document_sha256": content_hash}
        if any(cached.get(key) != value for key, value in expected.items()) or not card.should_index:
            return None
        content = str(cached.get("page_content", "")).strip()
        if not content:
            return None
        metadata = _cached_metadata(cached, card, tenant_id, source_id, content_hash, chunk_index, path)
        return Document(page_content=content, metadata=metadata)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        logger.warning("跳过无效图片语义缓存：%s", path)
        return None


def _cached_metadata(cached: dict, card: KnowledgeImageSemanticCard, tenant_id: str,
                     source_id: str, content_hash: str, chunk_index: int, path: Path) -> dict:
    return {
        "source": str(cached.get("full_image_path", "")), "document_name": str(cached.get("document_name", "")),
        "tenant_id": tenant_id, "source_id": source_id, "content_hash": content_hash,
        "content_type": "image_semantic", "image_order": int(cached["image_order"]),
        "image_sha256": str(cached.get("image_sha256", "")), "image_dhash": str(cached.get("image_dhash", "")),
        "image_detail_dhash": str(cached.get("image_detail_dhash", "")),
        "image_width": int(cached.get("image_width", 0)), "image_height": int(cached.get("image_height", 0)),
        "image_type": card.image_type, "semantic_confidence": float(card.confidence),
        "section_id": str(cached.get("section_id", "")), "section_title": str(cached.get("section_title", "")),
        "block_index": int(cached.get("block_index", 0)), "parent_block_index": int(cached.get("block_index", 0)),
        "chunk_index": chunk_index, "full_image_path": str(cached.get("full_image_path", "")),
        "semantic_result_path": str(path),
    }


def load_documents(*, tenant_id: str, source_id: str, content_hash: str,
                   starting_chunk_index: int, output_root: str | Path) -> list[Document]:
    directory = (Path(output_root).resolve() / _safe_path_segment(tenant_id)
                 / _safe_path_segment(source_id) / content_hash[:12])
    if not directory.exists():
        return []
    documents = []
    for path in sorted(directory.glob("image_*.json")):
        document = cached_document(path, tenant_id=tenant_id, source_id=source_id,
                                   content_hash=content_hash, chunk_index=starting_chunk_index + len(documents))
        if document is not None:
            documents.append(document)
    return documents


def validate_required_card(record: ExtractedImageRecord, *, tenant_id: str, source_id: str,
                           content_hash: str, model_name: str, output_root: str | Path) -> bool:
    path = _semantic_cache_path(record, tenant_id, source_id, output_root)
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        card = KnowledgeImageSemanticCard.model_validate(cached["semantic_card"])
        _require_reliable_card(card)
        fingerprint = fingerprint_image(record.extracted_path)
        expected = {
            "status": "completed", "prompt_version": PROMPT_VERSION, "model": model_name,
            "tenant_id": tenant_id, "source_id": source_id, "document_sha256": content_hash,
            "image_order": record.image_order, "image_sha256": fingerprint.sha256,
            "context_sha256": _context_sha256(record, ""), "page_content": format_knowledge_card(card, record),
        }
        if record.document_sha256 != content_hash or any(cached.get(key) != value for key, value in expected.items()):
            raise ValueError("图片语义缓存与当前资料或模型不一致")
        return card.should_index
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"图片 {record.image_order} 的有效 API 语义缓存缺失，请先完成图片预处理") from error
