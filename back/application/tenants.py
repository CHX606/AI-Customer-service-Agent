"""企业资料用例，与 HTTP 和 SQL 无关。"""

from datetime import datetime
from back.application.ports import AnswerCache, TenantRepository
from back.domain.errors import NotFound
from back.domain.tenant import TenantProfile, TenantProfileUpdate, TenantPublicProfile


class TenantProfileService:
    def __init__(self, repository: TenantRepository, cache: AnswerCache):
        self.repository = repository
        self.cache = cache

    def get(self, tenant_id: str) -> TenantProfile:
        profile = self.repository.get(tenant_id)
        if profile is None:
            raise NotFound(f"未找到租户 '{tenant_id}' 的配置资料")
        return profile

    def public(self, tenant_id: str) -> TenantPublicProfile:
        return TenantPublicProfile.model_validate(self.get(tenant_id).model_dump())

    def create(self, profile: TenantProfile) -> TenantProfile:
        self.repository.create(profile)
        return profile

    def update(self, request: TenantProfileUpdate) -> TenantProfile:
        current = self.get(request.tenant_id)
        updated = current.model_dump()
        updated.update(request.model_dump(exclude_unset=True))
        updated['updated_at'] = datetime.now()
        profile = TenantProfile.model_validate(updated)
        # 必须先清理旧资料答案；失效失败时不提交资料修改。
        with self.cache.mutation(request.tenant_id):
            self.repository.save(profile)
        return profile
