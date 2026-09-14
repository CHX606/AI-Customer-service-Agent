"""业务端口的默认适配器。重型依赖仅在对应操作中导入。"""

from back.infrastructure.persistence.errors import storage_operation


class SQLiteTenantRepository:
    @storage_operation
    def create(self, profile):
        from back.infrastructure.persistence.tenants import save_tenant_profile
        save_tenant_profile(profile, create_only=True)

    @storage_operation
    def get(self, tenant_id):
        from back.infrastructure.persistence.tenants import load_tenant_profile
        return load_tenant_profile(tenant_id)

    @storage_operation
    def save(self, profile):
        from back.infrastructure.persistence.tenants import save_tenant_profile
        save_tenant_profile(profile)


class SemanticAnswerCache:
    def revision(self, tenant_id):
        from back.knowledge.retrieval.cache import get_cache_revision
        return get_cache_revision(tenant_id)

    def mutation(self, tenant_id):
        from back.knowledge.retrieval.cache import semantic_cache_mutation
        return semantic_cache_mutation(tenant_id)

    def find(self, tenant_id, question):
        from back.knowledge.retrieval.cache import find_semantic_answer
        return find_semantic_answer(tenant_id, question)

    def store(self, tenant_id, question, answer, *, expected_revision=None):
        from back.knowledge.retrieval.cache import store_semantic_answer
        return store_semantic_answer(tenant_id, question, answer, expected_revision=expected_revision)

    @storage_operation
    def invalidate(self, tenant_id):
        from back.knowledge.retrieval.cache import invalidate_semantic_cache
        return invalidate_semantic_cache(tenant_id)


class CustomerImageInterpreter:
    def describe(self, content, message, tenant_id):
        from back.domain.errors import InvalidRequest, ProcessingFailed
        from back.knowledge.images.query import (
            UserImageValidationError, analyze_user_image, build_image_agent_message,
        )
        try:
            analysis = analyze_user_image(content, user_message=message, tenant_id=tenant_id)
            return build_image_agent_message(user_message=message, analysis=analysis)
        except UserImageValidationError as exc:
            raise InvalidRequest(str(exc)) from exc
        except RuntimeError as exc:
            raise ProcessingFailed("截图识别暂时失败，请重试或直接描述问题。") from exc


class SQLiteKnowledgeRepository:
    def operation(self, tenant_id, source_id):
        from back.infrastructure.operation_locks import source_operation
        return source_operation(tenant_id, source_id)

    @storage_operation
    def get(self, source_id):
        from back.infrastructure.persistence.tenants import load_knowledge_source
        return load_knowledge_source(source_id)

    @storage_operation
    def list(self, tenant_id):
        from back.infrastructure.persistence.tenants import list_knowledge_sources
        return list_knowledge_sources(tenant_id)

    @storage_operation
    def save(self, source):
        from back.infrastructure.persistence.tenants import save_knowledge_source
        save_knowledge_source(source)

    @storage_operation
    def delete(self, source_id):
        from back.infrastructure.persistence.tenants import delete_knowledge_source_record
        return delete_knowledge_source_record(source_id)


class DefaultKnowledgeIndexer:
    def index(self, source, path):
        from back.knowledge.indexing.service import index_knowledge_source
        return index_knowledge_source(source, path)

    def delete(self, tenant_id, source_id):
        from back.infrastructure.search.opensearch import delete_source_documents
        return delete_source_documents(tenant_id, source_id)
