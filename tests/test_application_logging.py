"""应用日志在 Uvicorn 启动和外部日志配置下均保持可观测。"""

import asyncio
from io import StringIO
import logging

import pytest
import uvicorn

import back.interfaces.http.app as app_module


@pytest.fixture
def application_logs(monkeypatch):
    root = logging.getLogger()
    namespace = logging.getLogger("back")
    root_level, namespace_level = root.level, namespace.level
    monkeypatch.setattr(root, "handlers", [])
    monkeypatch.setattr(namespace, "handlers", [])
    monkeypatch.setattr(namespace, "propagate", True)
    root.setLevel(logging.WARNING)
    namespace.setLevel(logging.NOTSET)
    try:
        yield root, namespace
    finally:
        for handler in namespace.handlers:
            if handler.get_name() == "customer_service_default":
                handler.close()
        root.setLevel(root_level)
        namespace.setLevel(namespace_level)


def _record_decision():
    logging.getLogger("back.agent.workflow.response_nodes").info("grounded status=partial")


def test_default_uvicorn_lifespan_emits_application_info(application_logs, monkeypatch, capsys):
    root, namespace = application_logs
    root.handlers.clear()
    uvicorn.Config("back.api:app")
    for name in ("initialize_database", "initialize_search", "preload_models"):
        monkeypatch.setattr(app_module, name, lambda: None)

    async def start_application():
        async with app_module.lifespan(app_module.create_app()):
            _record_decision()

    asyncio.run(start_application())

    assert capsys.readouterr().err.count("grounded status=partial") == 1
    assert namespace.getEffectiveLevel() == logging.INFO
    assert root.level == logging.WARNING
    assert logging.getLogger("application_logging_test_third_party").getEffectiveLevel() == logging.WARNING


def test_repeated_configuration_does_not_duplicate_output(application_logs, capsys):
    root, namespace = application_logs
    root.handlers.clear()
    app_module._configure_application_logging()
    app_module._configure_application_logging()
    _record_decision()
    namespace.warning("warning appears once")

    output = capsys.readouterr().err
    assert len(namespace.handlers) == 1
    assert output.count("grounded status=partial") == 1
    assert output.count("warning appears once") == 1


def test_existing_root_output_is_reused_without_changing_its_config(application_logs):
    root, namespace = application_logs
    root.handlers.clear()
    output = StringIO()
    handler = logging.StreamHandler(output)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("external:%(message)s"))
    root.addHandler(handler)

    app_module._configure_application_logging()
    _record_decision()

    assert output.getvalue() == "external:grounded status=partial\n"
    assert namespace.handlers == []
    assert namespace.propagate is True
    assert root.handlers == [handler]
    assert root.level == logging.WARNING
    assert handler.level == logging.INFO


def test_existing_namespace_configuration_is_preserved(application_logs):
    _, namespace = application_logs
    output = StringIO()
    handler = logging.StreamHandler(output)
    handler.setLevel(logging.DEBUG)
    namespace.addHandler(handler)
    namespace.setLevel(logging.DEBUG)
    namespace.propagate = False

    app_module._configure_application_logging()
    namespace.debug("configured debug remains enabled")

    assert namespace.handlers == [handler]
    assert namespace.level == logging.DEBUG
    assert namespace.propagate is False
    assert output.getvalue() == "configured debug remains enabled\n"


def test_existing_caplog_capture_still_receives_records(application_logs, caplog):
    root, namespace = application_logs
    root.addHandler(caplog.handler)
    with caplog.at_level(logging.INFO):
        app_module._configure_application_logging()
        _record_decision()

    assert namespace.handlers == []
    assert namespace.propagate is True
    assert "grounded status=partial" in caplog.text


def test_external_output_replaces_default_handler_on_reinitialization(application_logs, capsys):
    root, namespace = application_logs
    root.handlers.clear()
    app_module._configure_application_logging()
    fallback = namespace.handlers[0]
    output = StringIO()
    handler = logging.StreamHandler(output)
    handler.setLevel(logging.INFO)
    root.addHandler(handler)

    app_module._configure_application_logging()
    _record_decision()

    assert fallback not in namespace.handlers
    assert namespace.propagate is True
    assert output.getvalue() == "grounded status=partial\n"
    assert capsys.readouterr().err == ""


def test_null_handler_does_not_silence_default_application_logging(application_logs, capsys):
    root, namespace = application_logs
    root.handlers.clear()
    root.addHandler(logging.NullHandler())

    app_module._configure_application_logging()
    _record_decision()

    assert len(namespace.handlers) == 1
    assert capsys.readouterr().err.count("grounded status=partial") == 1


def test_explicit_application_warning_level_is_respected(application_logs):
    root, namespace = application_logs
    output = StringIO()
    handler = logging.StreamHandler(output)
    root.addHandler(handler)
    namespace.setLevel(logging.WARNING)

    app_module._configure_application_logging()
    _record_decision()

    assert namespace.level == logging.WARNING
    assert namespace.handlers == []
    assert output.getvalue() == ""
