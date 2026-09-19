"""The built-in Word source must remain reindexable from the management API."""
from back.tenant import service


def _stored_path():
    source = service.load_knowledge_source("legacy_kelecloud_docx")
    return service.get_tenant_upload_dir("default", source.source_id) / source.stored_filename


def test_fresh_initialization_keeps_builtin_docx_available_to_indexer():
    assert _stored_path().read_bytes() == service.LEGACY_DOCX_PATH.read_bytes()


def test_missing_unchanged_builtin_docx_can_be_restored():
    path = _stored_path()
    path.unlink()
    service.init_tenant_system()
    assert path.read_bytes() == service.LEGACY_DOCX_PATH.read_bytes()


def test_initialization_does_not_overwrite_existing_source_file():
    path = _stored_path()
    path.write_bytes(b"existing content must be preserved")
    service.init_tenant_system()
    assert path.read_bytes() == b"existing content must be preserved"


def test_initialization_does_not_restore_wrong_document_version():
    path = _stored_path()
    path.unlink()
    source = service.load_knowledge_source("legacy_kelecloud_docx")
    service.save_knowledge_source(source.model_copy(update={"content_hash": "different-version"}))
    service.init_tenant_system()
    assert not path.exists()
