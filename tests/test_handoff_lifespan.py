"""人工通知故障与专用线程池不能阻塞聊天，且关闭必须有界。"""

import asyncio
import threading
import time
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import anyio.to_thread
import httpx
import pytest

from back.domain.chat import ChatResult
import back.interfaces.http.app as http_app
import back.interfaces.http.chat as chat_http


def test_notification_factory_failure_is_isolated_and_does_not_log_secret(monkeypatch, caplog):
    async def scenario():
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        def unavailable():
            loop.call_soon(stop.set)
            raise RuntimeError("smtp-password-should-never-appear")
        monkeypatch.setattr(http_app, "get_handoff_service", unavailable)
        with ThreadPoolExecutor(max_workers=1) as executor:
            await asyncio.wait_for(http_app._deliver_handoff_notifications(stop, executor), timeout=1)
    asyncio.run(scenario())
    assert "RuntimeError" in caplog.text
    assert "smtp-password-should-never-appear" not in caplog.text


def test_blocked_smtp_does_not_consume_shared_chat_thread_pool(monkeypatch):
    monkeypatch.setenv("PUBLIC_CHAT_TENANTS", "default")
    for initialize in ("initialize_database", "initialize_search", "preload_models"):
        monkeypatch.setattr(http_app, initialize, Mock())
    chat = SimpleNamespace(tenants=SimpleNamespace(get=Mock(return_value=object())),
                           execute=Mock(return_value=ChatResult("正常问答不等待邮件", "smtp-isolation", "default")))
    monkeypatch.setattr(chat_http, "get_chat_service", lambda: chat)

    async def scenario():
        entered = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        def slow_dispatch(*, limit):
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(timeout=2):
                raise AssertionError("测试中的SMTP替身没有被及时释放")
            return 0
        monkeypatch.setattr(http_app, "get_handoff_service", lambda: SimpleNamespace(dispatch_pending=slow_dispatch))
        limiter = anyio.to_thread.current_default_thread_limiter()
        original_tokens = limiter.total_tokens
        limiter.total_tokens = 1
        try:
            application = http_app.create_app()
            async with http_app.lifespan(application):
                try:
                    await asyncio.wait_for(entered.wait(), timeout=1)
                    transport = httpx.ASGITransport(app=application)
                    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                        result = await asyncio.wait_for(client.post("/chat", json={
                            "tenant_id": "default", "session_id": "smtp-isolation", "message": "能退款吗",
                        }), timeout=0.5)
                        assert result.status_code == 200
                        assert result.json()["answer"] == "正常问答不等待邮件"
                        health = await asyncio.wait_for(client.get("/health"), timeout=0.5)
                        assert health.json()["status"] == "ok"
                finally:
                    release.set()
        finally:
            limiter.total_tokens = original_tokens
            release.set()
    asyncio.run(scenario())


def test_lifespan_shutdown_cancels_stalled_async_worker_after_bounded_wait(monkeypatch):
    for initialize in ("initialize_database", "initialize_search", "preload_models"):
        monkeypatch.setattr(http_app, initialize, Mock())
    real_wait_for = asyncio.wait_for
    monkeypatch.setattr(http_app, "HANDOFF_WORKER_SHUTDOWN_SECONDS", 0.05)

    async def scenario():
        entered = asyncio.Event()
        cancelled = asyncio.Event()
        async def stalled_worker(stop, executor):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        monkeypatch.setattr(http_app, "_deliver_handoff_notifications", stalled_worker)
        application = http_app.create_app()
        manager = http_app.lifespan(application)
        await manager.__aenter__()
        await real_wait_for(entered.wait(), timeout=1)
        started = time.monotonic()
        try:
            await real_wait_for(manager.__aexit__(None, None, None), timeout=1)
        except TimeoutError:
            pytest.fail("人工通知worker无响应时，应用退出缺少有界等待")
        assert time.monotonic() - started < 0.5
        await real_wait_for(cancelled.wait(), timeout=0.5)
    asyncio.run(scenario())


def test_unexpected_worker_failure_does_not_break_lifespan_close_or_log_secret(monkeypatch, caplog):
    for initialize in ("initialize_database", "initialize_search", "preload_models"):
        monkeypatch.setattr(http_app, initialize, Mock())
    async def scenario():
        entered = asyncio.Event()
        async def failed_worker(stop, executor):
            entered.set()
            raise RuntimeError("worker-exception-secret-text")
        monkeypatch.setattr(http_app, "_deliver_handoff_notifications", failed_worker)
        application = http_app.create_app()
        async with http_app.lifespan(application):
            await asyncio.wait_for(entered.wait(), timeout=1)
            assert application.state.startup_checks == {"database": "ok", "search": "ok", "models": "ok"}
    asyncio.run(scenario())
    assert "RuntimeError" in caplog.text
    assert "worker-exception-secret-text" not in caplog.text
