"""The packaged public source must be safe and reindexable from the admin API."""
import pytest

from back.tenant import service


def _stored_path():
    source = service.load_knowledge_source(service.PUBLIC_KNOWLEDGE_SOURCE_ID)
    return service.get_tenant_upload_dir("default", source.source_id) / source.stored_filename


def test_fresh_initialization_seeds_public_markdown_and_external_guide():
    sources = service.list_knowledge_sources("default")
    assert {source.source_id for source in sources} == {
        service.PUBLIC_KNOWLEDGE_SOURCE_ID, service.EXTERNAL_APPS_SOURCE_ID,
    }
    source = service.load_knowledge_source(service.PUBLIC_KNOWLEDGE_SOURCE_ID)
    assert source.file_type == "md"
    assert source.chunk_count == 0
    assert _stored_path().read_bytes() == service.PUBLIC_KNOWLEDGE_PATH.read_bytes()
    assert service.load_knowledge_source("legacy_kelecloud_docx") is None


def test_missing_unchanged_public_source_can_be_restored():
    path = _stored_path()
    path.unlink()
    service.init_tenant_system()
    assert path.read_bytes() == service.PUBLIC_KNOWLEDGE_PATH.read_bytes()


def test_initialization_does_not_overwrite_existing_source_file():
    path = _stored_path()
    path.write_bytes(b"existing content must be preserved")
    service.init_tenant_system()
    assert path.read_bytes() == b"existing content must be preserved"


def test_initialization_does_not_restore_wrong_document_version():
    path = _stored_path()
    path.unlink()
    source = service.load_knowledge_source(service.PUBLIC_KNOWLEDGE_SOURCE_ID)
    service.save_knowledge_source(source.model_copy(update={"content_hash": "different-version"}))
    service.init_tenant_system()
    assert not path.exists()


def test_deleted_public_default_source_is_recreated_without_restoring_private_docx():
    assert service.delete_knowledge_source(service.PUBLIC_KNOWLEDGE_SOURCE_ID, "default")
    service.init_tenant_system()
    assert _stored_path().read_bytes() == service.PUBLIC_KNOWLEDGE_PATH.read_bytes()
    assert service.load_knowledge_source("legacy_kelecloud_docx") is None


@pytest.mark.parametrize("status", ["ready", "processing"])
def test_uploaded_customer_document_prevents_packaged_source_reintroduction(status):
    assert service.delete_knowledge_source(service.PUBLIC_KNOWLEDGE_SOURCE_ID, "default")
    source = service.register_knowledge_source(
        tenant_id="default", source_id="src_customer_faq", original_filename="faq.md",
        stored_filename="faq.md", file_type="md", content_hash="customer-version", status=status,
    )
    path = service.get_tenant_upload_dir("default", source.source_id) / source.stored_filename
    path.write_text("Customer uploaded FAQ", encoding="utf-8")
    service.init_tenant_system()
    assert service.load_knowledge_source(service.PUBLIC_KNOWLEDGE_SOURCE_ID) is None
    assert service.load_knowledge_source("legacy_kelecloud_docx") is None
    assert path.read_text(encoding="utf-8") == "Customer uploaded FAQ"
    assert service.load_knowledge_source(source.source_id).status == status


def test_failed_upload_does_not_prevent_safe_default_registration():
    assert service.delete_knowledge_source(service.PUBLIC_KNOWLEDGE_SOURCE_ID, "default")
    service.register_knowledge_source(
        tenant_id="default", source_id="src_failed", original_filename="failed.md",
        stored_filename="failed.md", file_type="md", content_hash="failed-version", status="failed",
    )
    service.init_tenant_system()
    assert _stored_path().read_bytes() == service.PUBLIC_KNOWLEDGE_PATH.read_bytes()


def test_missing_public_file_is_not_restored_when_manual_source_is_ready():
    path = _stored_path()
    path.unlink()
    service.register_knowledge_source(
        tenant_id="default", source_id="src_customer_faq", original_filename="faq.md",
        stored_filename="faq.md", file_type="md", content_hash="customer-version", status="ready",
    )
    service.init_tenant_system()
    assert not path.exists()


def test_initialization_never_restores_existing_private_builtin_docx():
    source = service.register_knowledge_source(
        tenant_id="default", source_id="legacy_kelecloud_docx",
        original_filename=service.LEGACY_DOCX_PATH.name,
        stored_filename=service.LEGACY_DOCX_PATH.name, file_type="docx",
        content_hash=service.hashlib.sha256(service.LEGACY_DOCX_PATH.read_bytes()).hexdigest(),
        status="ready",
    )
    path = service.get_tenant_upload_dir("default", source.source_id) / source.stored_filename
    assert not path.exists()
    service.init_tenant_system()
    assert not path.exists()
