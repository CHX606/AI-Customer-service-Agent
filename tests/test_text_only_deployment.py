"""User-approved text-only mode must never run a hidden image fallback."""
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

from back.core.features import image_features_enabled
from back.interfaces.http.app import app
from back.knowledge.images.semantics import image_semantics_enabled
from back.knowledge.indexing import service as indexing
from back.knowledge.ingestion import knowledge_loader


@pytest.mark.parametrize("value", ["0", "false", "off", "no", ""])
def test_switch_disables_all_image_semantics(monkeypatch, value):
    monkeypatch.setenv("IMAGE_FEATURES_ENABLED", value)
    monkeypatch.setenv("IMAGE_SEMANTIC_ENABLED", "1")
    assert not image_features_enabled()
    assert not image_semantics_enabled()


def test_original_default_is_preserved(monkeypatch):
    monkeypatch.delenv("IMAGE_FEATURES_ENABLED", raising=False)
    assert image_features_enabled()


def test_image_endpoint_closed_before_service_creation(monkeypatch):
    monkeypatch.setenv("IMAGE_FEATURES_ENABLED", "0")
    image_service = Mock(side_effect=AssertionError("Image service must not be created"))
    monkeypatch.setattr("back.interfaces.http.chat.get_image_chat_service", image_service)
    client = TestClient(app)
    response = client.post("/chat/image", data={"session_id": "disabled-image"},
                           files={"image": ("image.png", b"not-even-an-image", "image/png")})
    assert response.status_code == 503
    assert "图片问答暂未开放" in response.json()["detail"]
    assert client.get("/health").json()["features"] == {"image_chat": False}
    assert client.get("/public/profile").status_code == 200
    image_service.assert_not_called()


def test_text_index_does_not_require_missing_ocr_files(monkeypatch):
    monkeypatch.setenv("IMAGE_FEATURES_ENABLED", "0")
    forbidden = Mock(side_effect=AssertionError("OCR cache must not be read"))
    monkeypatch.setattr(knowledge_loader, "load_image_documents", forbidden)
    documents = knowledge_loader.load_knowledge_documents("default")
    assert len(documents) > 10
    assert any("续费" in document.page_content for document in documents)
    assert not any(document.metadata.get("content_type", "").startswith("image") for document in documents)
    forbidden.assert_not_called()


def test_docx_reindex_does_not_call_ocr_or_image_api(monkeypatch, tmp_path):
    monkeypatch.setenv("IMAGE_FEATURES_ENABLED", "0")
    monkeypatch.setenv("IMAGE_SEMANTIC_ENABLED", "1")
    forbidden = Mock(side_effect=AssertionError("Image processing must not run"))
    monkeypatch.setattr(indexing, "load_image_documents", forbidden)
    monkeypatch.setattr(indexing, "build_docx_image_semantic_documents", forbidden)
    source = SimpleNamespace(file_type="docx", tenant_id="default")
    assert indexing._load_matching_reference_images(source, 0) == []
    assert indexing._load_contextual_image_documents(source, tmp_path / "source.docx", 0) == []
    forbidden.assert_not_called()


def test_disabled_startup_and_query_do_not_import_paddle():
    environment = {**os.environ, "IMAGE_FEATURES_ENABLED": "0", "PRELOAD_OCR_MODEL": "1",
                   "PRELOAD_RAG_MODELS": "0", "PYTHON_DOTENV_DISABLED": "1"}
    code = """
import sys
from back.bootstrap import preload_models
preload_models()
from back.knowledge.images.query import analyze_user_image
try:
    analyze_user_image(b'not an image')
except RuntimeError as error:
    assert '暂未开放' in str(error)
else:
    raise AssertionError('disabled query accepted')
assert 'back.knowledge.images.parser' not in sys.modules
assert 'paddleocr' not in sys.modules
assert 'paddle' not in sys.modules
print('PASS: disabled OCR never imported')
"""
    result = subprocess.run([sys.executable, "-c", code], env=environment, text=True,
                            capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
