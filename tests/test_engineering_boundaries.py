"""分层重构的契约、持久化和故障隔离验证。"""

import ast
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from threading import Barrier
from unittest.mock import Mock

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
import pytest

from back.bootstrap import get_chat_service, get_knowledge_service
from back.domain.chat import ChatCommand
from back.domain.errors import Conflict, DependencyUnavailable
from back.infrastructure.persistence.sessions import SQLiteSessionRepository
from back.interfaces.http.app import app, create_app


def conversation(text="回复"):
    return {
        "messages": [HumanMessage(content="问题"), AIMessage(content=text)],
        "active_issue": {"summary": "已有问题", "status": "answered"},
    }


def test_sessions_survive_repository_restart_and_isolate_tenants(tmp_path):
    path = tmp_path / "sessions.db"
    first = SQLiteSessionRepository(path)
    first.save(("tenant_a", "same-id"), conversation("A 的回复"), 0)
    restarted = SQLiteSessionRepository(path)
    snapshot = restarted.load(("tenant_a", "same-id"))
    assert snapshot.version == 1
    assert snapshot.state["messages"][-1].content == "A 的回复"
    assert snapshot.state["active_issue"]["summary"] == "已有问题"
    assert restarted.load(("tenant_b", "same-id")).state == {}


def test_concurrent_session_writes_do_not_overwrite_each_other(tmp_path):
    path = tmp_path / "sessions.db"
    first, second = SQLiteSessionRepository(path), SQLiteSessionRepository(path)
    first.load(("tenant", "session"))
    barrier = Barrier(2)

    def save(repository, answer):
        barrier.wait(timeout=5)
        try:
            repository.save(("tenant", "session"), conversation(answer), 0)
            return answer
        except Conflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(save, first, "first"), pool.submit(save, second, "second")]
        results = [job.result(timeout=10) for job in jobs]
    assert results.count("conflict") == 1
    winner = next(result for result in results if result != "conflict")
    assert first.load(("tenant", "session")).state["messages"][-1].content == winner


def test_session_serialization_keeps_bounded_history_without_search_documents(tmp_path):
    repository = SQLiteSessionRepository(tmp_path / "sessions.db", history_limit=4)
    state = conversation()
    state["messages"] = [HumanMessage(content=str(i)) for i in range(10)]
    state["retrieved_documents"] = [object()]
    repository.save(("tenant", "session"), state, 0)
    saved = repository.load(("tenant", "session")).state
    assert [message.content for message in saved["messages"]] == ["6", "7", "8", "9"]
    assert "retrieved_documents" not in saved


def test_corrupt_session_does_not_break_another_session(tmp_path):
    path = tmp_path / "sessions.db"
    repository = SQLiteSessionRepository(path)
    repository.save(("tenant", "bad"), conversation(), 0)
    repository.save(("tenant", "good"), conversation("仍然可读"), 0)
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE chat_sessions SET state_json='invalid' WHERE session_id='bad'")
    with pytest.raises(DependencyUnavailable):
        repository.load(("tenant", "bad"))
    assert repository.load(("tenant", "good")).state["messages"][-1].content == "仍然可读"


def test_model_failure_is_limited_to_chat_request(monkeypatch):
    monkeypatch.setenv("PUBLIC_CHAT_TENANTS", "default")
    monkeypatch.setattr(get_chat_service().engine, "invoke", Mock(side_effect=RuntimeError("private error")))
    client = TestClient(app)
    result = client.post("/chat", json={"message": "问题", "session_id": "failed-engine"})
    assert result.status_code == 503
    assert "private error" not in result.text
    assert client.get("/health").status_code == 200
    assert client.get("/public/profile").status_code == 200
    assert get_chat_service().sessions.load(("default", "failed-engine")).state == {}


def test_unavailable_session_store_does_not_break_profile_or_health(monkeypatch):
    monkeypatch.setenv("PUBLIC_CHAT_TENANTS", "default")
    monkeypatch.setattr(get_chat_service().sessions, "load", Mock(side_effect=DependencyUnavailable("会话库不可用")))
    client = TestClient(app)
    assert client.post("/chat", json={"message": "问题", "session_id": "x"}).status_code == 503
    assert client.get("/public/profile").status_code == 200
    assert client.get("/health").status_code == 200


def test_search_startup_failure_does_not_prevent_public_api(monkeypatch):
    monkeypatch.setattr("back.interfaces.http.app.initialize_search", Mock(side_effect=RuntimeError("search down")))
    with TestClient(create_app()) as client:
        assert client.get("/health").json()["status"] == "degraded"
        assert client.get("/public/profile").status_code == 200


def test_knowledge_delete_keeps_record_when_file_cleanup_fails(monkeypatch):
    service = get_knowledge_service()
    source = service.repository.list("default")[0]
    delete_index = Mock(return_value=1)
    monkeypatch.setattr(service.indexer, "delete", delete_index)
    monkeypatch.setattr(service.files, "delete", Mock(side_effect=OSError("disk unavailable")))
    with pytest.raises(DependencyUnavailable):
        service.delete("default", source.source_id)
    assert service.repository.get(source.source_id) is not None
    delete_index.assert_called_once_with("default", source.source_id)


def test_knowledge_delete_stops_before_files_when_search_fails(monkeypatch):
    service = get_knowledge_service()
    source = service.repository.list("default")[0]
    monkeypatch.setattr(service.indexer, "delete", Mock(side_effect=RuntimeError("search down")))
    files = Mock()
    monkeypatch.setattr(service.files, "delete", files)
    with pytest.raises(DependencyUnavailable):
        service.delete("default", source.source_id)
    files.assert_not_called()
    assert service.repository.get(source.source_id) is not None


def test_unauthorized_image_request_does_not_run_interpreter(monkeypatch):
    interpreter = Mock()
    monkeypatch.setattr("back.infrastructure.services.CustomerImageInterpreter.describe", interpreter)
    result = TestClient(app).post("/chat/image", data={"session_id": "unauthorized"},
                                  files={"image": ("sample.png", b"image", "image/png")})
    assert result.status_code == 403
    interpreter.assert_not_called()


def test_gateway_import_does_not_require_model_configuration(tmp_path):
    environment = os.environ.copy()
    environment.update(API_KEY="", BASE_URL="", MODEL="", PYTHON_DOTENV_DISABLED="1")
    result = subprocess.run([sys.executable, "-c",
        "import sys; from back.api import app; "
        "assert 'back.core.llm' not in sys.modules; "
        "assert 'back.agent.workflow.graph' not in sys.modules; print('ok')"],
        env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_application_and_domain_cannot_import_gateway_or_storage():
    root = Path(__file__).resolve().parents[1] / "back"
    forbidden = ("back.infrastructure", "back.interfaces", "fastapi", "sqlite3", "opensearchpy")
    violations = []
    for folder in ("application", "domain"):
        for path in (root / folder).glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                modules = [node.module or ""] if isinstance(node, ast.ImportFrom) else (
                    [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
                violations.extend(f"{path.name}: {module}" for module in modules
                                  if module.startswith(forbidden))
    assert not violations, violations
