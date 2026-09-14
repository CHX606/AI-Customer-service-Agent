"""验证测试执行期间正式数据库 data/app.db 与生产资源绝对不可变。"""

import hashlib
from pathlib import Path
from fastapi.testclient import TestClient

from back.core.paths import DATA_DIR
from back.interfaces.http.app import app
from back.domain.tenant import TenantProfile
from back.tenant.service import save_tenant_profile


PROD_DB_PATH = DATA_DIR / "app.db"

client = TestClient(app)


def compute_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_production_db_immutability_during_test_suite():
    """断言测试期间对数据库的一切增删改查均作用于 tmp_path，绝不污染 data/app.db。"""
    # 新克隆的仓库不包含运行数据库；测试也不能在默认路径创建它。
    initial_exists = PROD_DB_PATH.exists()
    initial_hash = compute_sha256(PROD_DB_PATH)

    # 1. 模拟大量测试操作：保存新租户
    save_tenant_profile(
        TenantProfile(
            tenant_id="ephemeral_tenant",
            company_name="临时测试租户",
            assistant_name="临时客服",
            short_description="用于验证不可变性",
            welcome_title="你好",
            welcome_description="欢迎",
            tone="简洁",
            handoff_message="转人工",
        )
    )

    # 2. 模拟 API 更新资料与上传
    client.put(
        "/admin/profile",
        headers={"X-Admin-Key": "master-admin-key"},
        json={
            "tenant_id": "default",
            "business_hours": "周一至周日 24小时 (测试修改)",
        },
    )

    # 3. 验证此时正式 data/app.db 哈希完全不变
    current_hash = compute_sha256(PROD_DB_PATH)
    assert PROD_DB_PATH.exists() == initial_exists, "测试不应创建或删除正式数据库"
    assert current_hash == initial_hash, (
        f"正式数据库被测试污染！初始哈希: {initial_hash}, 当前哈希: {current_hash}"
    )
