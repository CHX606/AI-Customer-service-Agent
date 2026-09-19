"""Verify the real configured vision API with built-in diagnostic screenshots.

Uses the application's typed schemas and stores a real reusable knowledge card.
Do not print credentials, raw model replies, or exception response bodies.
"""
import json
from pathlib import Path
import sys
import time


def main():
    from back.knowledge.ingestion.loader import DOCUMENT_PATH
    from back.knowledge.ingestion.docx.image_extractor import extract_docx_images
    from back.knowledge.images.semantics import (
        get_or_create_knowledge_semantics, understand_customer_image,
    )
    records = {record.image_order: record for record in extract_docx_images(DOCUMENT_PATH)}
    started = time.monotonic()
    card, _, cache_path = get_or_create_knowledge_semantics(
        records[7], tenant_id="default", source_id="legacy_kelecloud_docx",
    )
    text = card.summary + " " + " ".join(card.visible_evidence)
    assert card.should_index and card.confidence >= 0.35
    assert "订阅" in text and "节点" in text, "Knowledge image missed visible diagnostic evidence"
    assert cache_path.is_file()
    print(json.dumps({"check": "knowledge_image_structured_api", "status": "PASS",
                      "image_order": 7, "confidence": card.confidence,
                      "seconds": round(time.monotonic() - started, 2)}), flush=True)
    started = time.monotonic()
    customer = understand_customer_image(
        Path(records[9].extracted_path).read_bytes(),
        user_message="请识别截图中的客户端和报错信息。",
    )
    text = (customer.summary + " " + " ".join(customer.visible_evidence)).lower()
    assert customer.confidence >= 0.35
    assert any(term in text for term in ("tls", "fetch", "网络", "失败")), "Customer image missed visible error"
    assert "paddle" not in sys.modules and "paddleocr" not in sys.modules
    print(json.dumps({"check": "customer_image_structured_api", "status": "PASS",
                      "confidence": customer.confidence, "ocr_imported": False,
                      "seconds": round(time.monotonic() - started, 2)}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "FAIL", "error_type": type(error).__name__,
                          "http_status": getattr(error, "status_code", None)}), flush=True)
        raise SystemExit(1)
