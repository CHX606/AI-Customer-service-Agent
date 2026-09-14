"""
混合检索候选结果精排模块。

Reranker同时读取用户问题和候选chunk，计算两者的真实相关程度，
再从RRF宽召回结果中选出最终Top K。
"""

from functools import lru_cache
from threading import BoundedSemaphore
import os
from time import perf_counter

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from dotenv import load_dotenv
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


load_dotenv()


DEFAULT_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
LEGACY_MODEL_NAME = "BAAI/bge-reranker-base"
DEFAULT_BACKEND = "torch"
FINAL_RESULTS = 5
MAX_LENGTH = 512
BATCH_SIZE = 8
RETRIEVAL_WEIGHT = 0.45
RERANKER_WEIGHT = 0.55
RRF_CONSTANT = 60
DEFAULT_MAX_CONCURRENCY = 2
DEFAULT_QUEUE_TIMEOUT_SECONDS = 30.0


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} 必须是整数") from error
    if value < minimum:
        raise ValueError(f"{name} 不能小于 {minimum}")
    return value


def _env_float(name: str, default: float, *, minimum: float = 0.1) -> float:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = float(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} 必须是数字") from error
    if value < minimum:
        raise ValueError(f"{name} 不能小于 {minimum}")
    return value


def get_reranker_model_name() -> str:
    """读取当前精排模型，默认使用 BGE v2 多语言版本。"""

    model_name = os.getenv("RERANKER_MODEL", DEFAULT_MODEL_NAME).strip()
    if not model_name:
        raise ValueError("RERANKER_MODEL 不能为空")
    return model_name


def get_reranker_backend() -> str:
    """读取精排推理后端，支持原始 PyTorch 与 ONNX Runtime。"""

    backend = os.getenv("RERANKER_BACKEND", DEFAULT_BACKEND).strip().lower()
    if backend not in {"torch", "onnx"}:
        raise ValueError("RERANKER_BACKEND 仅支持 torch 或 onnx")
    return backend


def get_reranker_model_source() -> str:
    """ONNX 可加载本地导出目录，未配置时继续使用模型名称。"""

    if get_reranker_backend() != "onnx":
        return get_reranker_model_name()
    return os.getenv("RERANKER_MODEL_PATH", "").strip() or get_reranker_model_name()


def get_reranker_onnx_file() -> str:
    """返回本地导出目录中的 ONNX 文件名。"""

    return os.getenv("RERANKER_ONNX_FILE", "").strip()


def get_reranker_max_length() -> int:
    return _env_int("RERANKER_MAX_LENGTH", MAX_LENGTH)


def get_reranker_batch_size() -> int:
    return _env_int("RERANKER_BATCH_SIZE", BATCH_SIZE)


def get_reranker_max_concurrency() -> int:
    return _env_int("RERANKER_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY)


def get_reranker_queue_timeout_seconds() -> float:
    return _env_float(
        "RERANKER_QUEUE_TIMEOUT_SECONDS",
        DEFAULT_QUEUE_TIMEOUT_SECONDS,
    )


@lru_cache(maxsize=8)
def _get_reranker_semaphore(max_concurrency: int) -> BoundedSemaphore:
    """按配置复用进程内有界队列，防止 CPU 推理无限并发。"""
    return BoundedSemaphore(max_concurrency)


@lru_cache(maxsize=8)
def _load_reranker_model(
    model_source: str,
    backend: str,
    max_length: int,
    onnx_file: str,
) -> CrossEncoder:
    """按模型名称加载并缓存 CrossEncoder，支持同进程 A/B 测试。"""

    model_kwargs = None
    if backend == "onnx":
        model_kwargs = {
            "provider": "CPUExecutionProvider",
        }
        if onnx_file:
            model_kwargs["file_name"] = onnx_file

    return CrossEncoder(
        model_name_or_path=model_source,
        backend=backend,
        max_length=max_length,
        model_kwargs=model_kwargs,
        # BGE v2-m3 使用 XLM-R/SentencePiece；显式关闭 Transformers 对大词表
        # Tokenizer 的 Mistral 正则误判。启用该补丁会破坏 Metaspace 预分词器。
        processor_kwargs={"fix_mistral_regex": False},
    )


def get_reranker_model() -> CrossEncoder:
    """返回当前配置的精排模型。"""

    return _load_reranker_model(
        get_reranker_model_source(),
        get_reranker_backend(),
        get_reranker_max_length(),
        get_reranker_onnx_file(),
    )


def build_document_text(document: Document) -> str:
    """整理供Reranker判断的标题和正文。"""

    section_id = document.metadata.get(
        "section_id",
        "",
    )
    section_title = document.metadata.get(
        "section_title",
        "",
    )

    return (
        f"章节：{section_id} {section_title}\n"
        f"内容：{document.page_content}"
    )


def _deduplicate_documents(documents: list[Document]) -> list[Document]:
    """过滤跨数据源重复入库产生的完全相同候选块。"""

    unique_documents = []
    seen: set[tuple[str, str]] = set()
    for document in documents:
        key = (
            str(document.metadata.get("section_id", "")),
            " ".join(document.page_content.split()),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_documents.append(document)
    return unique_documents


def _retrieval_score(document: Document, rank: int) -> float:
    """读取 OpenSearch/多查询 RRF 分数，无元数据时按候选顺序回退。"""

    for field in ("rrf_score", "opensearch_score"):
        value = document.metadata.get(field)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return 1.0 / (RRF_CONSTANT + rank)


def _normalize_retrieval_scores(scores: list[float]) -> list[float]:
    """把当前候选集的 RRF 分数归一化，保持不同查询间可融合。"""

    if not scores:
        return []
    lower = min(scores)
    upper = max(scores)
    if upper == lower:
        return [1.0 for _ in scores]
    return [(score - lower) / (upper - lower) for score in scores]

def rerank_documents(
    query: str,
    documents: list[Document],
    limit: int = FINAL_RESULTS,
) -> list[Document]:
    """根据问题与chunk的匹配程度重新排序候选资料。"""

    if not documents:
        return []

    documents = _deduplicate_documents(documents)
    reranker = get_reranker_model()

    query_document_pairs = [
        (
            query,
            build_document_text(document),
        )
        for document in documents
    ]

    max_concurrency = get_reranker_max_concurrency()
    semaphore = _get_reranker_semaphore(max_concurrency)
    queue_started = perf_counter()
    acquired = semaphore.acquire(timeout=get_reranker_queue_timeout_seconds())
    queue_wait_ms = (perf_counter() - queue_started) * 1000
    if not acquired:
        raise TimeoutError("Reranker 等待队列超时")
    inference_started = perf_counter()
    try:
        scores = reranker.predict(
            query_document_pairs,
            batch_size=get_reranker_batch_size(),
            show_progress_bar=False,
            activation_fn=torch.nn.Sigmoid(),
        )
    finally:
        inference_ms = (perf_counter() - inference_started) * 1000
        semaphore.release()
    reranker_scores = [float(score) for score in scores]
    reranker_ranks = {
        index: rank
        for rank, index in enumerate(
            sorted(
                range(len(reranker_scores)),
                key=reranker_scores.__getitem__,
                reverse=True,
            ),
            start=1,
        )
    }

    retrieval_scores = [
        _retrieval_score(document, rank)
        for rank, document in enumerate(documents, start=1)
    ]
    normalized_retrieval_scores = _normalize_retrieval_scores(retrieval_scores)
    scored_documents = []
    for candidate_index, (
        document,
        reranker_score,
        retrieval_score,
        normalized_retrieval_score,
    ) in enumerate(
        zip(
            documents,
            reranker_scores,
            retrieval_scores,
            normalized_retrieval_scores,
            strict=True,
        ),
    ):
        retrieval_rank = candidate_index + 1
        reranker_rank = reranker_ranks[candidate_index]
        fusion_score = (
            RETRIEVAL_WEIGHT * normalized_retrieval_score
            + RERANKER_WEIGHT * reranker_score
        )
        scored_documents.append(
            (
                document,
                fusion_score,
                reranker_score,
                retrieval_score,
                retrieval_rank,
                reranker_rank,
            )
        )

    scored_documents.sort(
        key=lambda item: (item[1], item[2], -item[4]),
        reverse=True,
    )

    reranked_documents = []

    for rank, (
        document,
        fusion_score,
        reranker_score,
        retrieval_score,
        retrieval_rank,
        reranker_rank,
    ) in enumerate(
        scored_documents[:limit],
        start=1,
    ):
        reranked_documents.append(
            Document(
                page_content=document.page_content,
                metadata={
                    **document.metadata,
                    "retrieval_method": (
                        "hybrid_rrf_then_reranker"
                    ),
                    "reranker_rank": reranker_rank,
                    "reranker_score": reranker_score,
                    "reranker_model": get_reranker_model_name(),
                    "reranker_backend": get_reranker_backend(),
                    "reranker_max_concurrency": max_concurrency,
                    "reranker_queue_wait_ms": round(queue_wait_ms, 2),
                    "reranker_inference_ms": round(inference_ms, 2),
                    "retrieval_rank": retrieval_rank,
                    "retrieval_rrf_score": retrieval_score,
                    "fusion_rank": rank,
                    "fusion_score": fusion_score,
                },
            )
        )

    return reranked_documents
