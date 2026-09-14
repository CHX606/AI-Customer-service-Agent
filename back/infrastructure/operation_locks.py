"""跨线程、跨进程操作互斥；文件锁由操作系统在进程退出时释放。"""
from contextlib import contextmanager
from hashlib import sha256

from filelock import FileLock, Timeout

from back.domain.errors import Conflict
from back.infrastructure.persistence import tenants


def operation_lock(kind: str, tenant_id: str, source_id: str = "") -> FileLock:
    # 跟随业务库路径，测试和不同部署不会共享锁。
    directory = tenants.DEFAULT_DB_PATH.resolve().parent / "operation_locks"
    directory.mkdir(parents=True, exist_ok=True)
    key = sha256(f"{kind}\0{tenant_id}\0{source_id}".encode()).hexdigest()
    return FileLock(str(directory / f"{key}.lock"), is_singleton=True)


@contextmanager
def source_operation(tenant_id: str, source_id: str):
    try:
        with operation_lock("source", tenant_id, source_id).acquire(timeout=0):
            yield
    except Timeout as exc:
        raise Conflict("此知识源正在处理，请完成后重试。") from exc


@contextmanager
def cache_mutation_lock(tenant_id: str):
    try:
        with operation_lock("cache", tenant_id).acquire(timeout=0):
            yield
    except Timeout as exc:
        raise Conflict("此租户的知识或资料正在更新，请完成后重试。") from exc
