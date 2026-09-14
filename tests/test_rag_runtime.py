"""本地RAG模型预热测试。"""

from unittest.mock import Mock, patch

import pytest

import back.knowledge.retrieval.runtime as runtime


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, True),
        ("1", True),
        ("true", True),
        ("yes", True),
        ("0", False),
        ("false", False),
        ("NO", False),
        ("off", False),
    ],
)
def test_preload_flag(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("PRELOAD_RAG_MODELS", raising=False)
    else:
        monkeypatch.setenv("PRELOAD_RAG_MODELS", value)
    assert runtime.should_preload_rag_models() is expected


def test_warm_rag_models_loads_each_model_once():
    embedding = Mock(return_value=object())
    reranker = Mock(return_value=object())
    with (
        patch.object(runtime, "get_embedding_model", embedding),
        patch.object(runtime, "get_reranker_model", reranker),
    ):
        status = runtime.warm_rag_models()

    assert status == {"embedding": True, "reranker": True}
    embedding.assert_called_once_with()
    reranker.assert_called_once_with()


def test_embedding_failure_does_not_skip_reranker_or_raise():
    reranker = Mock(return_value=object())
    with (
        patch.object(runtime, "get_embedding_model", side_effect=RuntimeError("bad")),
        patch.object(runtime, "get_reranker_model", reranker),
    ):
        status = runtime.warm_rag_models()

    assert status == {"embedding": False, "reranker": True}
    reranker.assert_called_once_with()


def test_reranker_failure_does_not_erase_embedding_success():
    with (
        patch.object(runtime, "get_embedding_model", return_value=object()),
        patch.object(runtime, "get_reranker_model", side_effect=RuntimeError("bad")),
    ):
        status = runtime.warm_rag_models()

    assert status == {"embedding": True, "reranker": False}
