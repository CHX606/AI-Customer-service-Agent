"""SQLite 数据库存储与管理模块。"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from back.core.paths import DATA_DIR
from back.domain.errors import Conflict
from back.domain.tenant import (
    KnowledgeSource,
    KnowledgeSourceStatus,
    TenantProfile,
)


DEFAULT_DB_PATH = DATA_DIR / "app.db"


def get_db_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """获取 SQLite 数据库连接，启用外键约束与行字典访问。"""
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | str | None = None) -> None:
    """初始化数据库表结构。"""
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tenants (
                    tenant_id TEXT PRIMARY KEY,
                    company_name TEXT NOT NULL,
                    brand_name_en TEXT,
                    assistant_name TEXT NOT NULL,
                    short_description TEXT NOT NULL,
                    business_scope_json TEXT NOT NULL,
                    business_hours TEXT,
                    public_contact TEXT,
                    welcome_title TEXT NOT NULL,
                    welcome_description TEXT NOT NULL,
                    tone TEXT NOT NULL,
                    handoff_message TEXT NOT NULL,
                    suggested_questions_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_sources (
                    source_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    stored_filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_knowledge_sources_tenant
                ON knowledge_sources (tenant_id);
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_answer_cache (
                    cache_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL,
                    normalized_question TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_hit_at TEXT,
                    UNIQUE (tenant_id, normalized_question),
                    FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE
                );
                """
            )
            conn.execute("""CREATE TABLE IF NOT EXISTS semantic_cache_versions (
                tenant_id TEXT PRIMARY KEY,
                revision INTEGER NOT NULL DEFAULT 0
            )""")
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_semantic_answer_cache_tenant
                ON semantic_answer_cache (tenant_id);
                """
            )
    finally:
        conn.close()


def _row_to_tenant_profile(row: sqlite3.Row) -> TenantProfile:
    """将 SQLite 行转换为 TenantProfile 实例。"""
    return TenantProfile(
        tenant_id=row["tenant_id"],
        company_name=row["company_name"],
        brand_name_en=row["brand_name_en"],
        assistant_name=row["assistant_name"],
        short_description=row["short_description"],
        business_scope=json.loads(row["business_scope_json"]),
        business_hours=row["business_hours"],
        public_contact=row["public_contact"],
        welcome_title=row["welcome_title"],
        welcome_description=row["welcome_description"],
        tone=row["tone"],
        handoff_message=row["handoff_message"],
        suggested_questions=json.loads(row["suggested_questions_json"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _row_to_knowledge_source(row: sqlite3.Row) -> KnowledgeSource:
    """将 SQLite 行转换为 KnowledgeSource 实例。"""
    return KnowledgeSource(
        source_id=row["source_id"],
        tenant_id=row["tenant_id"],
        original_filename=row["original_filename"],
        stored_filename=row["stored_filename"],
        file_type=row["file_type"],
        content_hash=row["content_hash"],
        status=row["status"],
        error_message=row["error_message"],
        chunk_count=row["chunk_count"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def save_tenant_profile(
    profile: TenantProfile, db_path: Path | str | None = None, *, create_only: bool = False
) -> None:
    """保存资料；创建模式由数据库唯一约束原子拒绝重复租户。"""
    on_conflict = "DO NOTHING" if create_only else """DO UPDATE SET
        company_name=excluded.company_name,
        brand_name_en=excluded.brand_name_en,
        assistant_name=excluded.assistant_name,
        short_description=excluded.short_description,
        business_scope_json=excluded.business_scope_json,
        business_hours=excluded.business_hours,
        public_contact=excluded.public_contact,
        welcome_title=excluded.welcome_title,
        welcome_description=excluded.welcome_description,
        tone=excluded.tone,
        handoff_message=excluded.handoff_message,
        suggested_questions_json=excluded.suggested_questions_json,
        updated_at=excluded.updated_at"""
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                f"""
                INSERT INTO tenants (
                    tenant_id, company_name, brand_name_en, assistant_name,
                    short_description, business_scope_json, business_hours,
                    public_contact, welcome_title, welcome_description,
                    tone, handoff_message, suggested_questions_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id) {on_conflict};
                """,
                (
                    profile.tenant_id,
                    profile.company_name,
                    profile.brand_name_en,
                    profile.assistant_name,
                    profile.short_description,
                    json.dumps(profile.business_scope, ensure_ascii=False),
                    profile.business_hours,
                    profile.public_contact,
                    profile.welcome_title,
                    profile.welcome_description,
                    profile.tone,
                    profile.handoff_message,
                    json.dumps(
                        profile.suggested_questions, ensure_ascii=False
                    ),
                    profile.created_at.isoformat(),
                    profile.updated_at.isoformat(),
                ),
            )
            if create_only and cursor.rowcount == 0:
                raise Conflict(f"租户 '{profile.tenant_id}' 已存在，请通过资料更新接口修改。")
    finally:
        conn.close()


def load_tenant_profile(
    tenant_id: str, db_path: Path | str | None = None
) -> TenantProfile | None:
    """加载指定租户资料。"""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute(
            "SELECT * FROM tenants WHERE tenant_id = ?",
            (tenant_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return _row_to_tenant_profile(row)
    finally:
        conn.close()


def delete_tenant_profile(
    tenant_id: str, db_path: Path | str | None = None
) -> bool:
    """删除租户资料（级联删除所有知识库数据源）。"""
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                "DELETE FROM tenants WHERE tenant_id = ?",
                (tenant_id,),
            )
            return cursor.rowcount > 0
    finally:
        conn.close()


def save_knowledge_source(
    source: KnowledgeSource, db_path: Path | str | None = None
) -> None:
    """保存或更新知识库文件记录。"""
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO knowledge_sources (
                    source_id, tenant_id, original_filename, stored_filename,
                    file_type, content_hash, status, error_message, chunk_count,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    tenant_id=excluded.tenant_id,
                    original_filename=excluded.original_filename,
                    stored_filename=excluded.stored_filename,
                    file_type=excluded.file_type,
                    content_hash=excluded.content_hash,
                    status=excluded.status,
                    error_message=excluded.error_message,
                    chunk_count=excluded.chunk_count,
                    updated_at=excluded.updated_at;
                """,
                (
                    source.source_id,
                    source.tenant_id,
                    source.original_filename,
                    source.stored_filename,
                    source.file_type,
                    source.content_hash,
                    source.status,
                    source.error_message,
                    source.chunk_count,
                    source.created_at.isoformat(),
                    source.updated_at.isoformat(),
                ),
            )
    finally:
        conn.close()


def load_knowledge_source(
    source_id: str, db_path: Path | str | None = None
) -> KnowledgeSource | None:
    """查询单个知识库文件记录。"""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute(
            "SELECT * FROM knowledge_sources WHERE source_id = ?",
            (source_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return _row_to_knowledge_source(row)
    finally:
        conn.close()


def list_knowledge_sources(
    tenant_id: str, db_path: Path | str | None = None
) -> list[KnowledgeSource]:
    """查询指定租户的所有知识库文件记录。"""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute(
            "SELECT * FROM knowledge_sources WHERE tenant_id = ? ORDER BY created_at DESC",
            (tenant_id,),
        )
        rows = cursor.fetchall()
        return [_row_to_knowledge_source(row) for row in rows]
    finally:
        conn.close()


def delete_knowledge_source_record(
    source_id: str, db_path: Path | str | None = None
) -> bool:
    """删除指定知识库文件记录。"""
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                "DELETE FROM knowledge_sources WHERE source_id = ?",
                (source_id,),
            )
            return cursor.rowcount > 0
    finally:
        conn.close()
