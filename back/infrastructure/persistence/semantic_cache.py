"""语义缓存 SQL 仓储；不执行嵌入计算或业务缓存策略。"""
from datetime import datetime
import json
from threading import Lock
from back.infrastructure.persistence import tenants as tenant_store

_schema_lock = Lock()
_schema_ready_paths: set[str] = set()

def _ensure_cache_schema() -> None:
    """兼容未重启的旧数据库和独立脚本，按数据库路径只迁移一次。"""
    path_key = str(tenant_store.DEFAULT_DB_PATH.resolve())
    if path_key in _schema_ready_paths:
        return
    with _schema_lock:
        if path_key in _schema_ready_paths:
            return
        tenant_store.init_db()
        _schema_ready_paths.add(path_key)

class SQLiteSemanticCacheRepository:
    def _connect(self):
        _ensure_cache_schema()
        return tenant_store.get_db_connection()

    def revision(self, tenant_id):
        conn = self._connect()
        try:
            row = conn.execute("SELECT revision FROM semantic_cache_versions WHERE tenant_id=?", (tenant_id,)).fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()

    def record_hit(self, cache_id: int) -> None:
        now = datetime.now().isoformat()
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    """
                    UPDATE semantic_answer_cache
                    SET hit_count = hit_count + 1, last_hit_at = ?
                    WHERE cache_id = ?
                    """,
                    (now, cache_id),
                )
        finally:
            conn.close()

    def lookup(self, tenant_id, normalized):
        conn = self._connect()
        try:
            exact_row = conn.execute(
                """
                SELECT cache_id, answer
                FROM semantic_answer_cache
                WHERE tenant_id = ? AND normalized_question = ?
                """,
                (tenant_id, normalized),
            ).fetchone()
            if exact_row is not None:
                cache_id = int(exact_row["cache_id"])
                answer = str(exact_row["answer"])
                rows = []
            else:
                cache_id = -1
                answer = ""
                rows = conn.execute(
                    """
                    SELECT cache_id, answer, embedding_json
                    FROM semantic_answer_cache
                    WHERE tenant_id = ?
                    """,
                    (tenant_id,),
                ).fetchall()
        finally:
            conn.close()

        return cache_id, answer, rows


    def upsert(self, tenant_id, normalized, question, answer, embedding, limit, *, expected_revision=None):
        now = datetime.now().isoformat()
        conn = self._connect()
        try:
            with conn:
                # 在同一写事务中检查版本，不能让失效插入到检查与写入之间。
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute("SELECT revision FROM semantic_cache_versions WHERE tenant_id=?", (tenant_id,)).fetchone()
                revision = int(row[0]) if row else 0
                if expected_revision is not None and revision != expected_revision:
                    return False
                conn.execute(
                    """
                    INSERT INTO semantic_answer_cache (
                        tenant_id, normalized_question, question, answer,
                        embedding_json, hit_count, created_at, updated_at, last_hit_at
                    ) VALUES (?, ?, ?, ?, ?, 0, ?, ?, NULL)
                    ON CONFLICT(tenant_id, normalized_question) DO UPDATE SET
                        question = excluded.question,
                        answer = excluded.answer,
                        embedding_json = excluded.embedding_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        tenant_id,
                        normalized,
                        question.strip(),
                        answer.strip(),
                        json.dumps(embedding),
                        now,
                        now,
                    ),
                )
                conn.execute(
                    """
                    DELETE FROM semantic_answer_cache
                    WHERE tenant_id = ? AND cache_id NOT IN (
                        SELECT cache_id
                        FROM semantic_answer_cache
                        WHERE tenant_id = ?
                        ORDER BY COALESCE(last_hit_at, updated_at) DESC, cache_id DESC
                        LIMIT ?
                    )
                    """,
                    (
                        tenant_id,
                        tenant_id,
                        limit,
                    ),
                )
        finally:
            conn.close()
        return True

    def invalidate(self, tenant_id):
        conn = self._connect()
        try:
            with conn:
                conn.execute("""INSERT INTO semantic_cache_versions (tenant_id, revision) VALUES (?, 1)
                    ON CONFLICT(tenant_id) DO UPDATE SET revision=revision+1""", (tenant_id,))
                cursor = conn.execute(
                    "DELETE FROM semantic_answer_cache WHERE tenant_id = ?",
                    (tenant_id,),
                )
                return cursor.rowcount
        finally:
            conn.close()

    def count(self, tenant_id):
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS total FROM semantic_answer_cache WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchone()
            return int(row["total"])
        finally:
            conn.close()
