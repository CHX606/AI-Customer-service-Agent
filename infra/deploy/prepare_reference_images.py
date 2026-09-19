"""Prepare real reference-image caches, using the deployment's selected mode."""
from hashlib import sha256
import os
from pathlib import Path


def main():
    os.chdir(Path(__file__).resolve().parents[2])
    from back.core.features import image_features_enabled, local_ocr_enabled
    if not image_features_enabled():
        print("Reference image OCR skipped: image features explicitly disabled", flush=True)
        return
    if not local_ocr_enabled():
        prepare_api_images()
        return
    from back.knowledge.images.documents import (
        get_reference_ocr_index_path,
        load_image_documents,
    )
    from back.knowledge.images.parser import parse_reference_manifest
    from back.knowledge.images.splitter import (
        DEFAULT_OUTPUT_ROOT,
        REFERENCE_IMAGE_TEMPLATES,
        split_diagnostic_reference_images,
    )
    from back.knowledge.ingestion.loader import DOCUMENT_PATH

    index_path = get_reference_ocr_index_path()
    if not index_path.is_file():
        split_diagnostic_reference_images()
        digest = sha256(DOCUMENT_PATH.read_bytes()).hexdigest()[:12]
        manifest = DEFAULT_OUTPUT_ROOT / f"reference_{digest}" / "manifest.json"
        parse_reference_manifest(manifest)
    documents = load_image_documents()
    expected = set(REFERENCE_IMAGE_TEMPLATES)
    actual = {int(document.metadata["image_order"]) for document in documents}
    assert actual == expected, f"Unexpected reference images: {actual} != {expected}"
    assert all(document.page_content.strip() for document in documents)
    print(f"PASS: {len(documents)} real OCR reference documents ready", flush=True)


def prepare_api_images():
    import sys
    from back.knowledge.images.semantics import (
        build_docx_image_semantic_documents,
        load_required_docx_image_semantic_documents,
    )
    from back.knowledge.ingestion.docx.image_extractor import extract_docx_images
    from back.knowledge.ingestion.loader import DOCUMENT_PATH, compute_file_hash

    options = dict(document_path=DOCUMENT_PATH, tenant_id="default",
                   source_id="legacy_kelecloud_docx", content_hash=compute_file_hash(DOCUMENT_PATH),
                   starting_chunk_index=0)
    print("Preparing all reference images with the vision API (cached results reused)", flush=True)
    documents = build_docx_image_semantic_documents(**options, require_all=True)
    cached = load_required_docx_image_semantic_documents(**options)
    assert len(documents) == len(cached)
    assert {7, 8, 9, 10, 11} <= {int(document.metadata["image_order"]) for document in cached}
    assert "paddle" not in sys.modules and "paddleocr" not in sys.modules
    print(f"PASS: {len(extract_docx_images(DOCUMENT_PATH))} images processed; "
          f"{len(cached)} indexable semantic cards; local OCR never imported", flush=True)


if __name__ == "__main__":
    main()
