"""CrossEncoder 与 OpenSearch RRF 融合、候选去重测试。"""

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import sleep
from unittest.mock import patch

from langchain_core.documents import Document

import back.knowledge.retrieval.reranker as reranker


class FakeReranker:
    def __init__(self, scores: list[float]):
        self.scores = scores
        self.pairs = []

    def predict(self, pairs, **_kwargs):
        self.pairs = list(pairs)
        assert len(self.pairs) == len(self.scores)
        return self.scores


def document(
    name: str,
    content: str,
    rrf_score: float | None = None,
) -> Document:
    metadata = {
        "section_id": name,
        "section_title": name,
        "source_id": f"source_{name}",
    }
    if rrf_score is not None:
        metadata["rrf_score"] = rrf_score
    return Document(page_content=content, metadata=metadata)


def test_document_text_contains_section_and_content():
    item = document("4.3", "导入订阅说明")

    assert reranker.build_document_text(item) == (
        "章节：4.3 4.3\n内容：导入订阅说明"
    )


def test_default_reranker_uses_bge_v2_m3(monkeypatch):
    monkeypatch.delenv("RERANKER_MODEL", raising=False)

    assert reranker.get_reranker_model_name() == "BAAI/bge-reranker-v2-m3"


def test_reranker_model_can_be_switched_for_ab_test(monkeypatch):
    monkeypatch.setenv("RERANKER_MODEL", reranker.LEGACY_MODEL_NAME)
    monkeypatch.setenv("RERANKER_BACKEND", "torch")
    monkeypatch.delenv("RERANKER_MODEL_PATH", raising=False)
    monkeypatch.delenv("RERANKER_ONNX_FILE", raising=False)
    reranker._load_reranker_model.cache_clear()
    try:
        with patch.object(reranker, "CrossEncoder") as cross_encoder:
            reranker.get_reranker_model()

        cross_encoder.assert_called_once_with(
            model_name_or_path="BAAI/bge-reranker-base",
            backend="torch",
            max_length=reranker.MAX_LENGTH,
            model_kwargs=None,
            processor_kwargs={"fix_mistral_regex": False},
        )
    finally:
        reranker._load_reranker_model.cache_clear()


def test_onnx_backend_loads_exported_file(monkeypatch, tmp_path):
    monkeypatch.setenv("RERANKER_BACKEND", "onnx")
    monkeypatch.setenv("RERANKER_MODEL_PATH", str(tmp_path))
    monkeypatch.setenv("RERANKER_ONNX_FILE", "onnx/model_O3.onnx")
    monkeypatch.setenv("RERANKER_MAX_LENGTH", "512")
    reranker._load_reranker_model.cache_clear()
    try:
        with patch.object(reranker, "CrossEncoder") as cross_encoder:
            reranker.get_reranker_model()

        cross_encoder.assert_called_once_with(
            model_name_or_path=str(tmp_path),
            backend="onnx",
            max_length=512,
            model_kwargs={
                "provider": "CPUExecutionProvider",
                "file_name": "onnx/model_O3.onnx",
            },
            processor_kwargs={"fix_mistral_regex": False},
        )
    finally:
        reranker._load_reranker_model.cache_clear()


def test_invalid_reranker_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("RERANKER_BACKEND", "invalid")

    try:
        reranker.get_reranker_backend()
    except ValueError as error:
        assert "torch 或 onnx" in str(error)
    else:
        raise AssertionError("无效后端应该被拒绝")


def test_torch_backend_ignores_onnx_model_path(monkeypatch, tmp_path):
    monkeypatch.setenv("RERANKER_BACKEND", "torch")
    monkeypatch.setenv("RERANKER_MODEL_PATH", str(tmp_path))

    assert reranker.get_reranker_model_source() == reranker.get_reranker_model_name()


def test_duplicate_content_is_scored_only_once():
    first = document("1", "完全相同的正文", 0.032)
    duplicate = Document(
        page_content=first.page_content,
        metadata={**first.metadata, "source_id": "uploaded_copy"},
    )
    second = document("2", "另一段正文", 0.030)
    model = FakeReranker([0.8, 0.2])

    with patch.object(reranker, "get_reranker_model", return_value=model):
        results = reranker.rerank_documents("问题", [first, duplicate, second])

    assert len(model.pairs) == 2
    assert [item.page_content for item in results] == ["完全相同的正文", "另一段正文"]


def test_rrf_signal_can_preserve_a_strong_retrieval_result():
    first = document("1", "OpenSearch 排名第一", 0.032)
    second = document("2", "OpenSearch 排名第二", 0.031)
    model = FakeReranker([0.50, 0.60])

    with patch.object(reranker, "get_reranker_model", return_value=model):
        results = reranker.rerank_documents("问题", [first, second])

    assert [item.metadata["section_id"] for item in results] == ["1", "2"]
    assert results[0].metadata["reranker_rank"] == 2
    assert results[0].metadata["fusion_rank"] == 1
    assert results[0].metadata["retrieval_rrf_score"] == 0.032


def test_reranker_can_still_overturn_retrieval_when_confidence_gap_is_large():
    first = document("1", "OpenSearch 排名第一", 0.032)
    second = document("2", "精排明显更相关", 0.031)
    model = FakeReranker([0.0, 1.0])

    with patch.object(reranker, "get_reranker_model", return_value=model):
        results = reranker.rerank_documents("问题", [first, second])

    assert [item.metadata["section_id"] for item in results] == ["2", "1"]


def test_reranker_records_queue_and_inference_metrics(monkeypatch):
    monkeypatch.setenv("RERANKER_MAX_CONCURRENCY", "2")
    model = FakeReranker([0.8])
    reranker._get_reranker_semaphore.cache_clear()
    try:
        with patch.object(reranker, "get_reranker_model", return_value=model):
            result = reranker.rerank_documents("问题", [document("1", "正文")])

        metadata = result[0].metadata
        assert metadata["reranker_max_concurrency"] == 2
        assert metadata["reranker_queue_wait_ms"] >= 0
        assert metadata["reranker_inference_ms"] >= 0
    finally:
        reranker._get_reranker_semaphore.cache_clear()


def test_reranker_bounded_queue_limits_parallel_cpu_inference(monkeypatch):
    class ConcurrentFakeReranker:
        def __init__(self):
            self.active = 0
            self.max_active = 0
            self.lock = Lock()

        def predict(self, pairs, **_kwargs):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                sleep(0.02)
                return [0.5] * len(pairs)
            finally:
                with self.lock:
                    self.active -= 1

    monkeypatch.setenv("RERANKER_MAX_CONCURRENCY", "2")
    monkeypatch.setenv("RERANKER_QUEUE_TIMEOUT_SECONDS", "1")
    model = ConcurrentFakeReranker()
    docs = [document("1", "正文")]
    reranker._get_reranker_semaphore.cache_clear()
    try:
        with (
            patch.object(reranker, "get_reranker_model", return_value=model),
            ThreadPoolExecutor(max_workers=6) as executor,
        ):
            results = list(
                executor.map(
                    lambda _index: reranker.rerank_documents("问题", docs),
                    range(6),
                )
            )

        assert all(results)
        assert model.max_active == 2
    finally:
        reranker._get_reranker_semaphore.cache_clear()
