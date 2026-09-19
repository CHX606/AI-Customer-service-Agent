"""API-only vision must fail safely without silently losing knowledge or loading OCR."""
from dataclasses import replace
from io import BytesIO
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

from langchain_core.documents import Document
from PIL import Image
import pytest

from back.core.features import local_ocr_enabled
from back.knowledge.images import query, semantics
from back.knowledge.indexing import service as indexing
from back.knowledge.ingestion import knowledge_loader
from test_image_semantics import _card, _FakeModel, _record


@pytest.fixture(autouse=True)
def api_only(monkeypatch):
    monkeypatch.setenv("IMAGE_FEATURES_ENABLED", "1")
    monkeypatch.setenv("IMAGE_SEMANTIC_ENABLED", "1")
    monkeypatch.setenv("LOCAL_OCR_ENABLED", "0")


def _image():
    buffer = BytesIO()
    with Image.new("RGB", (120, 120), "white") as image:
        image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize("failure", ["provider", "low_confidence", "circuit", "disabled"])
def test_unavailable_vision_never_falls_back_to_ocr(monkeypatch, failure):
    assert not local_ocr_enabled()
    monkeypatch.setattr(query, "find_matching_image_semantics", Mock(return_value=None))
    forbidden = Mock(side_effect=AssertionError("Local OCR must never be called"))
    monkeypatch.setattr(query, "parse_image", forbidden)
    vision = Mock(return_value=semantics.CustomerImageUnderstanding(summary="模糊图片", confidence=0.1))
    monkeypatch.setattr(query, "understand_customer_image", vision)
    if failure == "provider":
        vision.side_effect = RuntimeError("test provider unavailable")
    elif failure == "circuit":
        query._record_vision_failure()
    elif failure == "disabled":
        monkeypatch.setenv("IMAGE_SEMANTIC_ENABLED", "0")
    with pytest.raises(RuntimeError, match="图片 API"):
        query.analyze_user_image(_image())
    forbidden.assert_not_called()
    if failure in {"circuit", "disabled"}:
        vision.assert_not_called()


def test_api_success_uses_semantic_retrieval_description(monkeypatch):
    monkeypatch.setattr(query, "find_matching_image_semantics", Mock(return_value=None))
    monkeypatch.setattr(query, "understand_customer_image", Mock(return_value=
        semantics.CustomerImageUnderstanding(summary="Clash TLS 报错", visible_evidence=["TLS failed"], confidence=0.9)))
    forbidden = Mock(side_effect=AssertionError("Local OCR must never be called"))
    monkeypatch.setattr(query, "parse_image", forbidden)
    result = query.analyze_user_image(_image())
    assert result.understanding_strategy == "vision_model"
    assert "TLS failed" in result.semantic_text
    assert result.ocr_text == ""
    forbidden.assert_not_called()


def test_docx_api_mode_does_not_load_legacy_ocr_cache(monkeypatch, tmp_path):
    forbidden = Mock(side_effect=AssertionError("Legacy OCR cache must not be read"))
    monkeypatch.setattr(indexing, "_load_matching_reference_images", forbidden)
    builder = Mock(return_value=[Document(page_content="图片语义", metadata={"image_order": 7})])
    monkeypatch.setattr(indexing, "build_docx_image_semantic_documents", builder)
    source = SimpleNamespace(file_type="docx", tenant_id="default", source_id="source", content_hash="hash")
    result = indexing._load_contextual_image_documents(source, tmp_path / "source.docx", 12)
    assert result[0].metadata["chunk_index"] == 12
    assert builder.call_args.kwargs["require_all"] is True
    assert builder.call_args.kwargs["ocr_by_image_order"] == {}
    forbidden.assert_not_called()


def test_legacy_rebuild_uses_verified_semantics_not_ocr(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("Legacy OCR cache must not be read"))
    monkeypatch.setattr(knowledge_loader, "load_image_documents", forbidden)
    loader = Mock(return_value=[Document(page_content="API 图片知识", metadata={"content_type": "image_semantic"})])
    monkeypatch.setattr(knowledge_loader, "load_required_docx_image_semantic_documents", loader)
    result = knowledge_loader._load_legacy_documents("default")
    assert any("API 图片知识" in item.page_content for item in result)
    loader.assert_called_once()
    forbidden.assert_not_called()


def test_strict_builder_propagates_failure_instead_of_partial_success(monkeypatch, tmp_path):
    record = _record(tmp_path)
    monkeypatch.setattr(semantics, "extract_docx_images", lambda _path: [record])
    with pytest.raises(RuntimeError, match="图片 7 的 API 预处理未完成"):
        semantics.build_docx_image_semantic_documents(
            document_path=tmp_path / "source.docx", tenant_id="default", source_id="source",
            content_hash=record.document_sha256, starting_chunk_index=0,
            output_root=tmp_path / "cache", model=_FakeModel(RuntimeError("test unavailable")), require_all=True,
        )
    assert not list((tmp_path / "cache").rglob("*.json"))


@pytest.mark.parametrize("card", [_card(summary=""), _card(confidence=0.1), _card(visible_evidence=[])])
def test_low_quality_card_is_not_saved_as_complete(tmp_path, card):
    with pytest.raises(ValueError):
        semantics.get_or_create_knowledge_semantics(
            _record(tmp_path), tenant_id="default", source_id="source", output_root=tmp_path / "cache",
            model=_FakeModel(card), require_quality=True,
        )
    assert not list((tmp_path / "cache").rglob("*.json"))


@pytest.mark.parametrize("damage", [None, "missing", "context", "hash", "model", "content", "quality"])
def test_strict_cache_loader_validates_all_images_without_api(monkeypatch, tmp_path, damage):
    record = _record(tmp_path)
    model = _FakeModel(_card(), name=semantics.IMAGE_UNDERSTANDING_MODEL_NAME or "unknown")
    _, _, path = semantics.get_or_create_knowledge_semantics(
        record, tenant_id="default", source_id="source", output_root=tmp_path / "cache", model=model,
    )
    if damage == "missing":
        path.unlink()
    elif damage == "context":
        record = replace(record, next_text="新内容")
    elif damage:
        cached = json.loads(path.read_text())
        if damage == "quality":
            cached["semantic_card"]["confidence"] = 0.1
        else:
            cached[{"hash": "image_sha256", "model": "model", "content": "page_content"}[damage]] = "invalid"
        path.write_text(json.dumps(cached))
    monkeypatch.setattr(semantics, "extract_docx_images", lambda _path: [record])
    forbidden = Mock(side_effect=AssertionError("Index rebuild must never call a model"))
    monkeypatch.setattr(semantics, "understand_knowledge_image", forbidden)
    options = dict(document_path=tmp_path / "source.docx", tenant_id="default", source_id="source",
                   content_hash=record.document_sha256, starting_chunk_index=10, output_root=tmp_path / "cache")
    if damage:
        with pytest.raises(RuntimeError, match="有效 API 语义缓存缺失"):
            semantics.load_required_docx_image_semantic_documents(**options)
    else:
        documents = semantics.load_required_docx_image_semantic_documents(**options)
        assert len(documents) == 1 and documents[0].metadata["chunk_index"] == 10
    forbidden.assert_not_called()


def test_api_only_preload_and_explicit_parser_never_import_paddle():
    environment = {**os.environ, "PRELOAD_OCR_MODEL": "1", "PRELOAD_RAG_MODELS": "0", "PYTHON_DOTENV_DISABLED": "1"}
    code = """
import sys
from back.bootstrap import preload_models
preload_models()
from back.knowledge.images.parser import get_image_parser
try:
    get_image_parser()
except RuntimeError:
    pass
else:
    raise AssertionError('Local OCR unexpectedly enabled')
assert 'paddle' not in sys.modules and 'paddleocr' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], env=environment, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
