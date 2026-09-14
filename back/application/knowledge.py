"""知识源用例：统一文件、登记、索引和缓存之间的操作顺序。"""

from datetime import datetime
import hashlib
import logging
from pathlib import Path
from uuid import uuid4

from back.application.ports import AnswerCache, KnowledgeIndexer, KnowledgeRepository, SourceFiles, TenantReader
from back.domain.errors import ApplicationError, Conflict, DependencyUnavailable, InvalidRequest, NotFound, ProcessingFailed
from back.domain.tenant import KnowledgeSource

MAX_FILE_BYTES = 20 * 1024 * 1024
ALLOWED_EXTENSIONS = {".docx", ".pdf", ".txt", ".md", ".markdown"}
logger = logging.getLogger(__name__)


class KnowledgeService:
    def __init__(self, repository: KnowledgeRepository, files: SourceFiles,
                 indexer: KnowledgeIndexer, cache: AnswerCache, tenants: TenantReader):
        self.repository = repository
        self.files = files
        self.indexer = indexer
        self.cache = cache
        self.tenants = tenants

    def get(self, tenant_id: str, source_id: str) -> KnowledgeSource:
        source = self.repository.get(source_id)
        if source is None or source.tenant_id != tenant_id:
            raise NotFound("未找到指定的知识库数据源")
        return source

    def list(self, tenant_id: str):
        return self.repository.list(tenant_id)

    def upload(self, tenant_id: str, filename: str, content: bytes) -> KnowledgeSource:
        if self.tenants.get(tenant_id) is None:
            raise NotFound(f"未找到租户 '{tenant_id}'，请先创建租户")
        name = Path(filename).name.strip()
        if not name or ".." in filename:
            raise InvalidRequest("文件名不合法，禁止路径穿越")
        extension = Path(name).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            raise InvalidRequest(f"不支持的文件格式：{extension}。仅支持 docx, pdf, txt, md 格式。")
        if not content or len(content) > MAX_FILE_BYTES:
            raise InvalidRequest("文件内容不能为空，且大小不能超过 20MB")
        content_hash = hashlib.sha256(content).hexdigest()
        now = datetime.now()
        source = KnowledgeSource(
            tenant_id=tenant_id, source_id=f"src_{uuid4().hex}", original_filename=name,
            stored_filename=name, file_type=extension.lstrip('.'), content_hash=content_hash,
            status="processing", chunk_count=0, created_at=now, updated_at=now,
        )
        # 在落盘和登记前获得租户变更锁；繁忙请求不留下无人处理的记录。
        try:
            with self.repository.operation(tenant_id, source.source_id), self.cache.mutation(tenant_id):
                # 去重也在锁内完成，避免两个上传先后通过检查后重复入库。
                if any(s.content_hash == content_hash and s.status in {"processing", "ready"}
                       for s in self.repository.list(tenant_id)):
                    raise Conflict("相同内容的知识库文件已经存在")
                return self._save_and_index(source, content)
        except ApplicationError:
            raise
        except Exception as exc:
            logger.exception("知识源上传失败 tenant=%s", tenant_id)
            raise DependencyUnavailable("知识源上传暂不可用，请稍后重试。") from exc

    def _save_and_index(self, source: KnowledgeSource, content: bytes):
        try:
            path = self.files.write(source.tenant_id, source.source_id, source.stored_filename, content)
            self.repository.save(source)
        except Exception as exc:
            try:
                self.files.delete(source)
            except Exception:
                logger.exception("清理未登记的上传文件失败 source=%s", source.source_id)
            raise DependencyUnavailable("知识源保存失败，请稍后重试。") from exc
        return self._index(source, path)

    def reindex(self, tenant_id: str, source_id: str):
        with self.repository.operation(tenant_id, source_id):
            return self._reindex(tenant_id, source_id)

    def _reindex(self, tenant_id: str, source_id: str):
        source = self.get(tenant_id, source_id)
        path = self.files.path(source)
        if not path.exists():
            raise NotFound("源文件已在磁盘丢失")
        return self._index(source, path)

    def _index(self, source, path):
        try:
            return self.indexer.index(source, path)
        except ApplicationError:
            raise
        except Exception as exc:
            logger.exception("知识源索引失败 source=%s", source.source_id)
            raise ProcessingFailed("知识库文档解析或索引失败，可稍后重新索引。") from exc

    def delete(self, tenant_id: str, source_id: str):
        with self.repository.operation(tenant_id, source_id):
            return self._delete(tenant_id, source_id)

    def _delete(self, tenant_id: str, source_id: str):
        source = self.get(tenant_id, source_id)
        try:
            # 变更前失效缓存；不能在删除失败后继续返回旧知识答案。
            with self.cache.mutation(tenant_id):
                self.indexer.delete(tenant_id, source_id)
                self.files.delete(source)
                if not self.repository.delete(source_id):
                    raise NotFound("删除数据源失败")
        except ApplicationError:
            raise
        except Exception as exc:
            # 磁盘/索引失败时保留登记记录，用户可重复请求清理。
            logger.exception("知识源删除未完成 source=%s", source_id)
            raise DependencyUnavailable("知识源删除未完成，请稍后重试。") from exc
