"""兼容旧导入路径；SQL 实现在基础设施持久化层。"""
from back.infrastructure.persistence.tenants import (
    DEFAULT_DB_PATH, get_db_connection, init_db, save_tenant_profile, load_tenant_profile,
    delete_tenant_profile, save_knowledge_source, load_knowledge_source,
    list_knowledge_sources, delete_knowledge_source_record,
)
