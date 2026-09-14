"""从业务库登记的原始知识源重建 OpenSearch 索引。"""

import argparse

from back.infrastructure.search.opensearch import (
    ensure_search_backend,
    rebuild_tenant_vector_store,
)
from back.tenant.service import init_tenant_system


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tenant",
        action="append",
        dest="tenants",
        help="要迁移的租户，可重复传入；默认迁移 default",
    )
    args = parser.parse_args()

    init_tenant_system()
    ensure_search_backend()
    for tenant_id in args.tenants or ["default"]:
        store = rebuild_tenant_vector_store(tenant_id)
        count = store.count()
        print(f"{tenant_id}: 已写入 {count} 个 OpenSearch 文档分块")


if __name__ == "__main__":
    main()
