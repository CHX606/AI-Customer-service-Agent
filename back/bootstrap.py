"""依赖组装入口；应用服务自身不选择数据库、模型或框架实现。"""

from functools import lru_cache
import logging
import os
from pathlib import Path
import time
from dotenv import load_dotenv

from back.application.chat import ChatService
from back.core.paths import DATA_DIR
from back.core.features import local_ocr_enabled
from back.infrastructure.agent import LangGraphEngine
from back.infrastructure.persistence.sessions import SQLiteSessionRepository
from back.infrastructure.services import SemanticAnswerCache, SQLiteTenantRepository

load_dotenv()


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    path = Path(os.getenv("SESSION_DB_PATH") or str(DATA_DIR / "sessions.db"))
    return ChatService(LangGraphEngine(), SQLiteSessionRepository(path), SemanticAnswerCache(), SQLiteTenantRepository())


@lru_cache(maxsize=1)
def get_profile_service():
    from back.application.tenants import TenantProfileService
    return TenantProfileService(SQLiteTenantRepository(), SemanticAnswerCache())


def get_image_chat_service():
    from back.application.images import ImageChatService
    from back.infrastructure.services import CustomerImageInterpreter
    return ImageChatService(get_chat_service(), CustomerImageInterpreter())


@lru_cache(maxsize=1)
def get_knowledge_service():
    from back.application.knowledge import KnowledgeService
    from back.infrastructure.files import LocalSourceFiles
    from back.infrastructure.services import DefaultKnowledgeIndexer, SQLiteKnowledgeRepository
    from back.tenant import service
    return KnowledgeService(SQLiteKnowledgeRepository(), LocalSourceFiles(service.DATA_DIR),
                            DefaultKnowledgeIndexer(), SemanticAnswerCache(), SQLiteTenantRepository())


def initialize_database():
    from back.tenant.service import init_tenant_system
    init_tenant_system()


def initialize_search():
    if os.getenv("OPENSEARCH_BOOTSTRAP_ON_STARTUP", "1").lower() in {"0", "false", "no", "off"}:
        return
    from back.infrastructure.search.opensearch import ensure_search_backend
    # Docker restarts containers independently on host boot. OpenSearch usually
    # needs longer than Python; a transient outage must not become permanent
    # startup degradation until someone manually restarts the backend again.
    deadline = time.monotonic() + max(0, float(os.getenv("OPENSEARCH_STARTUP_RETRY_SECONDS", "180")))
    while True:
        try:
            ensure_search_backend()
            return
        except Exception as error:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            logging.getLogger(__name__).warning("等待搜索服务启动：%s", type(error).__name__)
            time.sleep(min(2.0, remaining))


def preload_models():
    # API-only deployments must not import or warm Paddle, even if preload is set.
    if local_ocr_enabled() and os.getenv("PRELOAD_OCR_MODEL", "0").lower() in {"1", "true", "yes", "on"}:
        from back.knowledge.images.parser import get_image_parser
        get_image_parser()
    if os.getenv("PRELOAD_RAG_MODELS", "1").lower() in {"0", "false", "no", "off"}:
        return
    from back.knowledge.retrieval.runtime import warm_rag_models
    if not all(warm_rag_models().values()):
        raise RuntimeError("部分检索模型预热失败")
