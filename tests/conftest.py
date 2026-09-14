"""Pytest 全局测试隔离配置与测试夹具。

确保所有测试使用临时 SQLite 数据库和临时上传目录，
并在每个测试前后清理 lru_cache 与内存状态，严格隔离生产环境 data/app.db。
"""

import json
from pathlib import Path
import pytest

import back.interfaces.http.app
import back.bootstrap
import socket
import back.core.llm
import back.knowledge.retrieval.embeddings
import back.infrastructure.search.opensearch
import back.knowledge.images.query
import back.tenant.service
import back.infrastructure.persistence.tenants
from back.tenant.service import init_tenant_system


@pytest.fixture(autouse=True)
def isolate_test_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request):
    """自动为每一个测试用例隔离数据库、文件上传与向量存储目录。"""
    test_db_path = tmp_path / "test_app.db"
    test_data_dir = tmp_path / "data"

    monkeypatch.setenv("SESSION_DB_PATH", str(tmp_path / "sessions.db"))
    def reject_network(*args, **kwargs):
        raise AssertionError("离线测试禁止真实网络请求，请注入测试替身")
    if not request.node.get_closest_marker("live_model"):
        monkeypatch.setattr(socket, "create_connection", reject_network)

    # 1. 隔离存储路径
    monkeypatch.setattr(back.infrastructure.persistence.tenants, "DEFAULT_DB_PATH", test_db_path)
    monkeypatch.setattr(back.tenant.service, "DATA_DIR", test_data_dir)

    # 2. 设置测试用密钥环境（包含 Master Key 和各租户专属 Key）
    monkeypatch.setenv("ADMIN_API_KEY", "master-admin-key")
    monkeypatch.setenv("ADMIN_AUTH_DISABLED", "0")
    monkeypatch.setenv("PRELOAD_RAG_MODELS", "0")
    monkeypatch.setenv("OPENSEARCH_BOOTSTRAP_ON_STARTUP", "0")
    monkeypatch.setenv("OPENSEARCH_PASSWORD", "test-password")
    # 旧测试默认关闭语义缓存，专项测试显式开启，避免加载真实向量模型。
    monkeypatch.setenv("SEMANTIC_CACHE_ENABLED", "0")
    # 图片语义专项测试显式开启，其他测试保持离线且不访问视觉模型。
    monkeypatch.setenv("IMAGE_SEMANTIC_ENABLED", "0")
    monkeypatch.setenv(
        "TENANT_ADMIN_KEYS",
        json.dumps(
            {
                "default": "default-admin-key",
                "tenant_a": "key-tenant-a",
                "tenant_b": "key-tenant-b",
            }
        ),
    )
    monkeypatch.setenv(
        "TENANT_ACCESS_KEYS",
        json.dumps(
            {
                "tenant_a": "token-tenant-a",
                "tenant_b": "token-tenant-b",
            }
        ),
    )

    # 3. 清理缓存与内存状态
    back.infrastructure.search.opensearch.get_vector_store.cache_clear()
    back.infrastructure.search.opensearch.get_opensearch_client.cache_clear()
    back.knowledge.retrieval.embeddings.get_embedding_model.cache_clear()
    back.bootstrap.get_chat_service.cache_clear()
    back.bootstrap.get_knowledge_service.cache_clear()
    back.bootstrap.get_profile_service.cache_clear()
    back.core.llm.get_image_understanding_model.cache_clear()
    back.knowledge.images.query.reset_image_vision_circuit()

    # 4. 初始化临时数据库与 default 租户
    init_tenant_system()

    yield

    # 5. 测试结束后再次清理缓存
    back.infrastructure.search.opensearch.get_vector_store.cache_clear()
    back.infrastructure.search.opensearch.get_opensearch_client.cache_clear()
    back.knowledge.retrieval.embeddings.get_embedding_model.cache_clear()
    back.bootstrap.get_chat_service.cache_clear()
    back.bootstrap.get_knowledge_service.cache_clear()
    back.bootstrap.get_profile_service.cache_clear()
    back.core.llm.get_image_understanding_model.cache_clear()
    back.knowledge.images.query.reset_image_vision_circuit()


def pytest_addoption(parser):
    parser.addoption("--run-live-model", action="store_true", help="显式运行真实模型集成测试")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-live-model"):
        return
    for item in items:
        if item.get_closest_marker("live_model"):
            item.add_marker(pytest.mark.skip(reason="真实模型测试需显式传入 --run-live-model"))
