"""Real image-chat acceptance through the web proxy, without inserting customer knowledge."""
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid

import httpx
from PIL import Image, ImageDraw, ImageFont


def main():
    from back.core.features import image_features_enabled, local_ocr_enabled
    from back.infrastructure.search.opensearch import get_vector_store
    from back.knowledge.images.semantics import IMAGE_SEMANTICS_ROOT, find_matching_image_semantics
    from back.knowledge.ingestion.docx.image_extractor import extract_docx_images
    from back.knowledge.ingestion.loader import DOCUMENT_PATH
    from back.tenant.service import get_all_knowledge_sources

    assert image_features_enabled() and not local_ocr_enabled()
    store = get_vector_store("default")

    def snapshot():
        return {
            "count": store.count(),
            "sources": sorted((source.source_id, source.content_hash, source.status, source.chunk_count)
                              for source in get_all_knowledge_sources("default")),
            "image_caches": {str(path): sha256(path.read_bytes()).hexdigest()
                             for path in IMAGE_SEMANTICS_ROOT.rglob("*.json")},
        }

    before = snapshot()
    records = extract_docx_images(DOCUMENT_PATH)
    known = Path(next(record for record in records if record.image_order == 9).extracted_path).read_bytes()
    assert find_matching_image_semantics(known).strategy == "exact_sha256"
    buffer = BytesIO()
    with Image.new("RGB", (760, 340), "#edf2f7") as image:
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=28)
        draw.text((30, 30), "Clash Verge - Subscription", fill="#172b4d", font=font)
        draw.rounded_rectangle((24, 95, 730, 300), radius=10, fill="white", outline="red", width=3)
        draw.text((50, 125), "Failed to fetch subscription", fill="red", font=font)
        draw.text((50, 185), "TLS handshake failed", fill="black", font=font)
        draw.text((50, 245), "Import failed. Please try again.", fill="black", font=font)
        image.save(buffer, format="PNG")
    unknown = buffer.getvalue()
    assert find_matching_image_semantics(unknown) is None
    fixture = Path("/tmp/ai-customer-service-unknown-image.png")
    fixture.write_bytes(unknown)
    reports = []
    with httpx.Client(timeout=300, trust_env=False) as client:
        for name, content, strategy in [("known_image", known, "exact_sha256"), ("unknown_image", unknown, "vision_model")]:
            session_id = "deploy-image-" + uuid.uuid4().hex[:16]
            started = time.monotonic()
            response = client.post("http://web/api/chat/image", files={"image": (name + ".png", content, "image/png")},
                                   data={"session_id": session_id, "tenant_id": "default", "message": "截图里这个导入问题该怎么处理？"})
            assert response.status_code == 200, f"{name}: HTTP {response.status_code}"
            answer = response.json()["answer"]
            assert len(answer) > 20 and any(word in answer for word in ["订阅", "TLS", "导入", "网络"])
            with sqlite3.connect("file:" + os.environ["SESSION_DB_PATH"] + "?mode=ro", uri=True) as connection:
                saved = connection.execute("SELECT state_json FROM chat_sessions WHERE tenant_id=? AND session_id=?",
                                           ("default", session_id)).fetchone()
            assert saved and f"理解策略={strategy}" in saved[0]
            reports.append({"check": name, "status": "PASS", "strategy": strategy,
                            "seconds": round(time.monotonic() - started, 2), "session_id": session_id, "answer": answer})
            print(json.dumps({key: value for key, value in reports[-1].items() if key != "answer"}), flush=True)
    assert snapshot() == before, "Customer images unexpectedly changed the knowledge base"
    report_path = Path("/app/data/deployment-image-acceptance.json")
    descriptor = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(reports, handle, ensure_ascii=False, indent=2)
    print("PASS: customer screenshots did not change knowledge sources, vectors or image-semantic caches", flush=True)


if __name__ == "__main__":
    main()
