"""RAG 本地模型运行时预热。"""

import logging
import os
from time import perf_counter

from back.knowledge.retrieval.embeddings import get_embedding_model
from back.knowledge.retrieval.reranker import get_reranker_model


logger = logging.getLogger(__name__)


def should_preload_rag_models() -> bool:
    """默认预热；可通过环境变量在轻量部署或测试中关闭。"""
    value = os.getenv("PRELOAD_RAG_MODELS", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def warm_rag_models() -> dict[str, bool]:
    """提前加载本地模型；单个模型失败不阻止服务启动。"""
    status: dict[str, bool] = {}
    for name, factory in (
        ("embedding", get_embedding_model),
        ("reranker", get_reranker_model),
    ):
        started = perf_counter()
        try:
            factory()
        except Exception:
            status[name] = False
            logger.exception("RAG %s 模型预热失败，将在请求时重试", name)
        else:
            status[name] = True
            logger.info(
                "RAG %s 模型预热完成，耗时 %.3f 秒",
                name,
                perf_counter() - started,
            )
    return status
