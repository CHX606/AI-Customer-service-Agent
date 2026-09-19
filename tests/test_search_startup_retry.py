from unittest.mock import Mock
import pytest
from back import bootstrap


def test_startup_recovers_when_search_starts_later(monkeypatch):
    monkeypatch.setenv("OPENSEARCH_BOOTSTRAP_ON_STARTUP", "1")
    monkeypatch.setenv("OPENSEARCH_STARTUP_RETRY_SECONDS", "30")
    ensure = Mock(side_effect=[ConnectionError("starting"), ConnectionError("starting"), None])
    monkeypatch.setattr("back.infrastructure.search.opensearch.ensure_search_backend", ensure)
    sleep = Mock()
    monkeypatch.setattr(bootstrap.time, "sleep", sleep)
    bootstrap.initialize_search()
    assert ensure.call_count == 3 and sleep.call_count == 2


def test_startup_outage_still_fails_after_deadline(monkeypatch):
    monkeypatch.setenv("OPENSEARCH_BOOTSTRAP_ON_STARTUP", "1")
    monkeypatch.setenv("OPENSEARCH_STARTUP_RETRY_SECONDS", "0")
    ensure = Mock(side_effect=ConnectionError("unavailable"))
    monkeypatch.setattr("back.infrastructure.search.opensearch.ensure_search_backend", ensure)
    sleep = Mock()
    monkeypatch.setattr(bootstrap.time, "sleep", sleep)
    with pytest.raises(ConnectionError):
        bootstrap.initialize_search()
    ensure.assert_called_once()
    sleep.assert_not_called()


def test_explicitly_disabled_bootstrap_does_not_retry(monkeypatch):
    monkeypatch.setenv("OPENSEARCH_BOOTSTRAP_ON_STARTUP", "0")
    ensure = Mock(side_effect=AssertionError("Search must not be contacted"))
    monkeypatch.setattr("back.infrastructure.search.opensearch.ensure_search_backend", ensure)
    bootstrap.initialize_search()
    ensure.assert_not_called()
