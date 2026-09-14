"""知识源本地文件适配器；所有读写限制在指定数据目录。"""

from pathlib import Path
import shutil

from back.domain.errors import InvalidRequest
from back.domain.tenant import validate_tenant_id


class LocalSourceFiles:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def _directory(self, tenant_id: str, source_id: str) -> Path:
        validate_tenant_id(tenant_id)
        if not source_id or any(char in source_id for char in '/\\') or source_id in {'.', '..'}:
            raise InvalidRequest("知识源标识不合法")
        path = (self.root / 'tenants' / tenant_id / 'uploads' / source_id).resolve()
        if not path.is_relative_to(self.root):
            raise InvalidRequest("知识源路径超出允许范围")
        return path

    def write(self, tenant_id, source_id, name, content):
        directory = self._directory(tenant_id, source_id)
        if Path(name).name != name or '/' in name or '\\' in name:
            raise InvalidRequest("文件名不合法")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        path.write_bytes(content)
        return path

    def path(self, source):
        directory = self._directory(source.tenant_id, source.source_id)
        path = (directory / source.stored_filename).resolve()
        if not path.is_relative_to(directory):
            raise InvalidRequest("知识源文件路径不合法")
        return path

    def delete(self, source):
        directory = self._directory(source.tenant_id, source.source_id)
        if directory.exists():
            # 不吞掉磁盘错误：上层保留登记记录，允许后续重试。
            shutil.rmtree(directory)
