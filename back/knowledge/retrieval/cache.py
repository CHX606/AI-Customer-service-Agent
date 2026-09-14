"""租户隔离的持久化语义问答缓存。"""

import json
import math
import os
import re
import unicodedata
from contextlib import contextmanager
from filelock import Timeout

from dotenv import load_dotenv

from back.knowledge.retrieval.embeddings import get_embedding_model
from back.infrastructure.persistence.semantic_cache import SQLiteSemanticCacheRepository
from back.domain.chat import CacheHit as SemanticCacheHit
from back.domain.tenant import validate_tenant_id
from back.infrastructure.operation_locks import operation_lock, cache_mutation_lock


load_dotenv()

DEFAULT_SIMILARITY_THRESHOLD = 0.88
DEFAULT_MAX_ENTRIES_PER_TENANT = 200
MIN_CACHEABLE_QUESTION_LENGTH = 4

_SENSITIVE_PATTERNS = (
    re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
    re.compile(r"(?:https?://|www\.)", re.IGNORECASE),
    re.compile(r"(?<!\d)\d{6,}(?!\d)"),
    re.compile(
        r"(?:我的|本人|我这边的)(?:订单|账号|账户|套餐|余额|流量|支付|退款|手机号|邮箱)"
    ),
    re.compile(r"(?:订单号|交易号|流水号|手机号|身份证号|验证码)"),
)

_repository = SQLiteSemanticCacheRepository()


def get_cache_revision(tenant_id: str) -> int | None:
    """更新期间不使用答案缓存；普通聊天无需等待长时间的知识处理。"""
    tenant_id = validate_tenant_id(tenant_id)
    try:
        with operation_lock("cache", tenant_id).acquire(timeout=0):
            return _repository.revision(tenant_id)
    except Timeout:
        return None


@contextmanager
def semantic_cache_mutation(tenant_id: str):
    with cache_mutation_lock(tenant_id):
        invalidate_semantic_cache(tenant_id)
        yield


def is_semantic_cache_enabled() -> bool:
    """缓存默认关闭，生产环境需显式开启。"""
    value = os.getenv("SEMANTIC_CACHE_ENABLED", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def get_similarity_threshold() -> float:
    """读取并限制相似度阈值，避免误配置导致宽泛命中。"""
    try:
        value = float(
            os.getenv(
                "SEMANTIC_CACHE_THRESHOLD",
                str(DEFAULT_SIMILARITY_THRESHOLD),
            )
        )
    except (TypeError, ValueError):
        return DEFAULT_SIMILARITY_THRESHOLD
    return min(0.999, max(0.80, value))


def get_max_entries_per_tenant() -> int:
    """读取单租户缓存容量并限制在合理范围。"""
    try:
        value = int(
            os.getenv(
                "SEMANTIC_CACHE_MAX_ENTRIES",
                str(DEFAULT_MAX_ENTRIES_PER_TENANT),
            )
        )
    except (TypeError, ValueError):
        return DEFAULT_MAX_ENTRIES_PER_TENANT
    return min(5000, max(1, value))


def normalize_question(question: str) -> str:
    """统一全半角、大小写、空白和标点，用于精确缓存命中。"""
    normalized = unicodedata.normalize("NFKC", question).strip().lower()
    return re.sub(r"[\W_]+", "", normalized, flags=re.UNICODE)


def is_safe_cache_question(question: str) -> bool:
    """过滤上下文不足或可能包含个人标识的提问。"""
    normalized = normalize_question(question)
    if len(normalized) < MIN_CACHEABLE_QUESTION_LENGTH:
        return False
    return not any(pattern.search(question) for pattern in _SENSITIVE_PATTERNS)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return -1.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return -1.0
    return dot / (left_norm * right_norm)



def find_semantic_answer(
    tenant_id: str,
    question: str,
) -> SemanticCacheHit | None:
    """优先精确匹配，再以严格阈值进行租户内语义匹配。"""
    if not is_semantic_cache_enabled() or not is_safe_cache_question(question):
        return None

    safe_tenant_id = validate_tenant_id(tenant_id)
    revision = get_cache_revision(safe_tenant_id)
    if revision is None:
        return None
    normalized = normalize_question(question)
    cache_id, answer, rows = _repository.lookup(safe_tenant_id, normalized)

    if cache_id >= 0:
        if get_cache_revision(safe_tenant_id) != revision:
            return None
        _repository.record_hit(cache_id)
        return SemanticCacheHit(answer=answer, similarity=1.0, exact=True)
    if not rows:
        return None

    query_embedding = [
        float(value)
        for value in get_embedding_model().embed_query(question.strip())
    ]
    best_row = None
    best_similarity = -1.0
    for row in rows:
        try:
            stored_embedding = [float(value) for value in json.loads(row["embedding_json"])]
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        similarity = _cosine_similarity(query_embedding, stored_embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_row = row

    if best_row is None or best_similarity < get_similarity_threshold():
        return None

    if get_cache_revision(safe_tenant_id) != revision:
        return None

    _repository.record_hit(int(best_row["cache_id"]))
    return SemanticCacheHit(
        answer=str(best_row["answer"]),
        similarity=best_similarity,
        exact=False,
    )


def store_semantic_answer(tenant_id: str, question: str, answer: str, *, expected_revision: int | None = None) -> bool:
    """写入可靠回答；相同规范化问题会更新而不是重复堆积。"""
    if (
        not is_semantic_cache_enabled()
        or not is_safe_cache_question(question)
        or not answer.strip()
    ):
        return False

    safe_tenant_id = validate_tenant_id(tenant_id)
    revision = expected_revision if expected_revision is not None else get_cache_revision(safe_tenant_id)
    if revision is None:
        return False
    embedding = [float(value) for value in get_embedding_model().embed_query(question.strip())]
    try:
        with operation_lock("cache", safe_tenant_id).acquire(timeout=0):
            return _repository.upsert(safe_tenant_id, normalize_question(question), question, answer,
                                      embedding, get_max_entries_per_tenant(), expected_revision=revision)
    except Timeout:
        return False


def invalidate_semantic_cache(tenant_id: str) -> int:
    """清空租户缓存，错误交由应用服务决定是否允许降级。"""
    return _repository.invalidate(validate_tenant_id(tenant_id))


def count_semantic_cache_entries(tenant_id: str) -> int:
    return _repository.count(validate_tenant_id(tenant_id))
