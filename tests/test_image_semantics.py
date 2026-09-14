"""图片语义预处理、缓存、指纹匹配与异常降级测试。"""

from dataclasses import replace
from io import BytesIO
import json

from langchain_core.documents import Document
from PIL import Image, ImageDraw
import pytest

from back.knowledge.images import semantics as image_semantics
from back.knowledge.ingestion.docx.models import ExtractedImageRecord
from back.knowledge.images.semantics import (
    CustomerImageUnderstanding,
    ImageFingerprint,
    KnowledgeImageSemanticCard,
    build_docx_image_semantic_documents,
    dhash_distance,
    find_matching_image_semantics,
    fingerprint_image,
    format_customer_understanding,
    format_knowledge_card,
    get_or_create_knowledge_semantics,
    load_cached_image_semantic_documents,
    understand_customer_image,
    understand_knowledge_image,
)


def _pattern_image_bytes(
    *,
    size: tuple[int, int] = (320, 180),
    image_format: str = "PNG",
    quality: int = 90,
) -> bytes:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((15, 15, size[0] - 15, 55), fill="navy")
    draw.rectangle((30, 80, size[0] - 80, 115), fill="red")
    draw.line((20, 145, size[0] - 20, 130), fill="black", width=5)
    buffer = BytesIO()
    image.save(buffer, format=image_format, quality=quality)
    image.close()
    return buffer.getvalue()


def _record(tmp_path, *, image_order: int = 7) -> ExtractedImageRecord:
    image_path = tmp_path / f"image_{image_order:03d}.png"
    image_path.write_bytes(_pattern_image_bytes())
    return ExtractedImageRecord(
        source=str(tmp_path / "knowledge.docx"),
        document_name="knowledge.docx",
        document_sha256="a" * 64,
        extracted_path=str(image_path),
        media_path=f"word/media/image{image_order}.png",
        image_name=image_path.name,
        image_sha256="unused",
        relationship_id=f"rId{image_order}",
        position_type="inline",
        image_order=image_order,
        image_index_in_block=1,
        block_index=20 + image_order,
        paragraph_index=20 + image_order,
        section_id="4.3",
        section_title="导入",
        previous_text="导入失败时先确认软件版本。",
        next_text="出现 TLS 错误时更换文档指定的订阅地址。",
    )


def _card(**updates) -> KnowledgeImageSemanticCard:
    data = {
        "should_index": True,
        "image_type": "故障截图",
        "summary": "Clash 导入订阅失败界面",
        "context_role": "展示需要更换订阅地址的典型报错",
        "visible_evidence": ["failed to fetch", "TLS verifier failed"],
        "problem_meaning": "订阅导入请求失败",
        "recommended_action": "按照相邻正文更换订阅地址",
        "search_queries": ["Clash 导入失败怎么办", "TLS 报错"],
        "uncertainty": "",
        "confidence": 0.96,
    }
    data.update(updates)
    return KnowledgeImageSemanticCard(**data)


class _FakeStructuredInvoker:
    def __init__(self, owner, response):
        self.owner = owner
        self.response = response

    def invoke(self, messages):
        self.owner.messages.append(messages)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class _FakeModel:
    def __init__(self, response, name="vision-test"):
        self.response = response
        self.model_name = name
        self.schemas = []
        self.messages = []

    def with_structured_output(self, schema):
        self.schemas.append(schema)
        return _FakeStructuredInvoker(self, self.response)


def _store_data(documents):
    return {
        "documents": [document.page_content for document in documents],
        "metadatas": [document.metadata for document in documents],
    }


class _FakeVectorStore:
    def __init__(self, exact_documents=(), semantic_documents=()):
        self.exact_documents = list(exact_documents)
        self.semantic_documents = list(semantic_documents)
        self.calls = []

    def get(self, **kwargs):
        self.calls.append(kwargs)
        where = kwargs.get("where", {})
        if "image_sha256" in where:
            return _store_data(self.exact_documents)
        return _store_data(self.semantic_documents)


def _patch_vector_store(monkeypatch, store):
    def getter(_tenant_id):
        return store
    monkeypatch.setattr(image_semantics, "_get_image_metadata_collection", getter)


def test_card_cleans_duplicate_and_empty_list_items():
    card = _card(
        visible_evidence=[" 错误 ", "错误", "", "节点全红"],
        search_queries=[" 怎么办 ", "怎么办", ""],
    )
    assert card.visible_evidence == ["错误", "节点全红"]
    assert card.search_queries == ["怎么办"]


def test_card_rejects_out_of_range_confidence():
    with pytest.raises(ValueError):
        _card(confidence=1.2)


def test_fingerprint_exact_hash_changes_but_perceptual_hash_survives_reencoding():
    png_bytes = _pattern_image_bytes(image_format="PNG")
    jpeg_bytes = _pattern_image_bytes(image_format="JPEG", quality=75)
    png_fingerprint = fingerprint_image(png_bytes)
    jpeg_fingerprint = fingerprint_image(jpeg_bytes)

    assert png_fingerprint.sha256 != jpeg_fingerprint.sha256
    assert dhash_distance(png_fingerprint.dhash, jpeg_fingerprint.dhash) <= 4
    assert (
        dhash_distance(
            png_fingerprint.detail_dhash,
            jpeg_fingerprint.detail_dhash,
        )
        <= 24
    )
    assert (png_fingerprint.width, png_fingerprint.height) == (320, 180)


@pytest.mark.parametrize(
    ("left", "right", "message"),
    [
        ("xyz", "0000", "左侧"),
        ("0000", "xyz", "右侧"),
        ("00", "0000", "长度"),
    ],
)
def test_dhash_distance_rejects_invalid_values(left, right, message):
    with pytest.raises(ValueError, match=message):
        dhash_distance(left, right)


def test_understand_knowledge_image_sends_context_and_full_image(tmp_path):
    record = _record(tmp_path)
    model = _FakeModel(_card())
    result = understand_knowledge_image(record, ocr_text="TLS error", model=model)

    assert result.summary == "Clash 导入订阅失败界面"
    assert model.schemas == [KnowledgeImageSemanticCard]
    messages = model.messages[0]
    assert "不得补充" in messages[0].content
    assert "导入失败时先确认软件版本" in messages[1].content[0]["text"]
    assert "TLS error" in messages[1].content[0]["text"]
    assert messages[1].content[1]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,"
    )


def test_understand_customer_image_returns_retrieval_description():
    response = CustomerImageUnderstanding(
        summary="Windows 代理设置页",
        visible_evidence=["代理开关已开启", "端口 10808"],
        likely_problem="需要核对系统代理配置",
        search_queries=["Windows 代理端口"],
        confidence=0.9,
    )
    model = _FakeModel(response)
    result = understand_customer_image(
        _pattern_image_bytes(),
        user_message="还是打不开网页",
        model=model,
    )

    assert result == response
    assert "不要自行给出业务处理方案" in model.messages[0][0].content
    assert "还是打不开网页" in model.messages[0][1].content[0]["text"]


def test_format_knowledge_card_binds_image_to_context(tmp_path):
    text = format_knowledge_card(_card(), _record(tmp_path), ocr_text="TLS error")
    assert "图片与上下文联合资料" in text
    assert "在本段中的作用" in text
    assert "文档支持的处理方式" in text
    assert "图片上文" in text and "图片下文" in text
    assert "TLS error" in text


def test_format_customer_understanding_contains_search_terms():
    text = format_customer_understanding(
        CustomerImageUnderstanding(
            summary="订阅失败弹窗",
            visible_evidence=["failed to fetch"],
            likely_problem="订阅请求失败",
            search_queries=["导入订阅失败"],
            uncertainty="软件版本不清楚",
            confidence=0.8,
        )
    )
    assert "failed to fetch" in text
    assert "导入订阅失败" in text
    assert "软件版本不清楚" in text


def test_semantic_cache_reuses_same_image_context_model(tmp_path):
    record = _record(tmp_path)
    model = _FakeModel(_card())
    first = get_or_create_knowledge_semantics(
        record,
        tenant_id="default",
        source_id="src_1",
        output_root=tmp_path / "cache",
        model=model,
    )
    second = get_or_create_knowledge_semantics(
        record,
        tenant_id="default",
        source_id="src_1",
        output_root=tmp_path / "cache",
        model=model,
    )

    assert len(model.messages) == 1
    assert first[0] == second[0]
    cached = json.loads(first[2].read_text(encoding="utf-8"))
    assert cached["image_sha256"] == first[1].sha256
    assert cached["image_detail_dhash"] == first[1].detail_dhash


def test_semantic_cache_context_change_forces_regeneration(tmp_path):
    record = _record(tmp_path)
    model = _FakeModel(_card())
    kwargs = {
        "tenant_id": "default",
        "source_id": "src_1",
        "output_root": tmp_path / "cache",
        "model": model,
    }
    get_or_create_knowledge_semantics(record, **kwargs)
    changed = replace(record, next_text="新的处理规则")
    get_or_create_knowledge_semantics(changed, **kwargs)
    assert len(model.messages) == 2


def test_semantic_cache_model_change_forces_regeneration(tmp_path):
    record = _record(tmp_path)
    root = tmp_path / "cache"
    first_model = _FakeModel(_card(), name="vision-a")
    second_model = _FakeModel(_card(), name="vision-b")
    get_or_create_knowledge_semantics(
        record,
        tenant_id="default",
        source_id="src_1",
        output_root=root,
        model=first_model,
    )
    get_or_create_knowledge_semantics(
        record,
        tenant_id="default",
        source_id="src_1",
        output_root=root,
        model=second_model,
    )
    assert len(first_model.messages) == 1
    assert len(second_model.messages) == 1


def test_corrupt_semantic_cache_is_regenerated(tmp_path):
    record = _record(tmp_path)
    model = _FakeModel(_card())
    _, _, cache_path = get_or_create_knowledge_semantics(
        record,
        tenant_id="default",
        source_id="src_1",
        output_root=tmp_path / "cache",
        model=model,
    )
    cache_path.write_text("{bad json", encoding="utf-8")
    get_or_create_knowledge_semantics(
        record,
        tenant_id="default",
        source_id="src_1",
        output_root=tmp_path / "cache",
        model=model,
    )
    assert len(model.messages) == 2


def test_force_bypasses_valid_semantic_cache(tmp_path):
    record = _record(tmp_path)
    model = _FakeModel(_card())
    kwargs = {
        "tenant_id": "default",
        "source_id": "src_1",
        "output_root": tmp_path / "cache",
        "model": model,
    }
    get_or_create_knowledge_semantics(record, **kwargs)
    get_or_create_knowledge_semantics(record, force=True, **kwargs)
    assert len(model.messages) == 2


def test_cached_semantic_documents_can_rebuild_index_without_model(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("IMAGE_SEMANTIC_ENABLED", "1")
    record = _record(tmp_path)
    model = _FakeModel(_card())
    get_or_create_knowledge_semantics(
        record,
        tenant_id="default",
        source_id="src_1",
        output_root=tmp_path / "cache",
        model=model,
    )
    documents = load_cached_image_semantic_documents(
        tenant_id="default",
        source_id="src_1",
        content_hash="a" * 64,
        starting_chunk_index=27,
        output_root=tmp_path / "cache",
    )

    assert len(model.messages) == 1
    assert len(documents) == 1
    assert documents[0].metadata["chunk_index"] == 27
    assert documents[0].metadata["content_type"] == "image_semantic"
    assert documents[0].metadata["image_detail_dhash"]
    assert "图片上文" in documents[0].page_content


def test_cached_loader_skips_decorative_and_corrupt_entries(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("IMAGE_SEMANTIC_ENABLED", "1")
    root = tmp_path / "cache"
    record = _record(tmp_path)
    get_or_create_knowledge_semantics(
        record,
        tenant_id="default",
        source_id="src_1",
        output_root=root,
        model=_FakeModel(
            _card(should_index=False, image_type="装饰或无关图片")
        ),
    )
    cache_directory = root / "default" / "src_1" / ("a" * 12)
    (cache_directory / "image_999.json").write_text(
        "{invalid", encoding="utf-8"
    )
    assert load_cached_image_semantic_documents(
        tenant_id="default",
        source_id="src_1",
        content_hash="a" * 64,
        starting_chunk_index=0,
        output_root=root,
    ) == []


def test_cached_loader_rejects_different_document_version(monkeypatch, tmp_path):
    monkeypatch.setenv("IMAGE_SEMANTIC_ENABLED", "1")
    record = _record(tmp_path)
    root = tmp_path / "cache"
    get_or_create_knowledge_semantics(
        record,
        tenant_id="default",
        source_id="src_1",
        output_root=root,
        model=_FakeModel(_card()),
    )
    assert load_cached_image_semantic_documents(
        tenant_id="default",
        source_id="src_1",
        content_hash="b" * 64,
        starting_chunk_index=0,
        output_root=root,
    ) == []


def test_build_documents_indexes_meaningful_images_and_skips_decorations(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("IMAGE_SEMANTIC_ENABLED", "1")
    records = [_record(tmp_path, image_order=7), _record(tmp_path, image_order=8)]
    monkeypatch.setattr(image_semantics, "extract_docx_images", lambda _path: records)

    def fake_semantics(record, **_kwargs):
        card = _card() if record.image_order == 7 else _card(
            should_index=False,
            image_type="装饰或无关图片",
        )
        fingerprint = fingerprint_image(record.extracted_path)
        return card, fingerprint, tmp_path / f"{record.image_order}.json"

    monkeypatch.setattr(
        image_semantics,
        "get_or_create_knowledge_semantics",
        fake_semantics,
    )
    documents = build_docx_image_semantic_documents(
        document_path=tmp_path / "knowledge.docx",
        tenant_id="default",
        source_id="src_1",
        content_hash="b" * 64,
        starting_chunk_index=27,
    )

    assert len(documents) == 1
    assert documents[0].metadata["image_order"] == 7
    assert documents[0].metadata["chunk_index"] == 27
    assert documents[0].metadata["content_type"] == "image_semantic"
    assert "图片下文" in documents[0].page_content


def test_build_documents_single_image_failure_does_not_discard_others(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("IMAGE_SEMANTIC_ENABLED", "1")
    records = [_record(tmp_path, image_order=7), _record(tmp_path, image_order=8)]
    monkeypatch.setattr(image_semantics, "extract_docx_images", lambda _path: records)

    def fake_semantics(record, **_kwargs):
        if record.image_order == 7:
            raise RuntimeError("vision unavailable")
        return (
            _card(),
            fingerprint_image(record.extracted_path),
            tmp_path / "8.json",
        )

    monkeypatch.setattr(
        image_semantics,
        "get_or_create_knowledge_semantics",
        fake_semantics,
    )
    documents = build_docx_image_semantic_documents(
        document_path=tmp_path / "knowledge.docx",
        tenant_id="default",
        source_id="src_1",
        content_hash="b" * 64,
        starting_chunk_index=3,
    )
    assert [document.metadata["image_order"] for document in documents] == [8]


def test_build_documents_stops_batch_after_provider_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("IMAGE_SEMANTIC_ENABLED", "1")
    records = [_record(tmp_path, image_order=7), _record(tmp_path, image_order=8)]
    monkeypatch.setattr(image_semantics, "extract_docx_images", lambda _path: records)
    calls = []

    class ProviderError(RuntimeError):
        status_code = 522

    def fail_provider(record, **_kwargs):
        calls.append(record.image_order)
        raise ProviderError("upstream unavailable")

    monkeypatch.setattr(
        image_semantics,
        "get_or_create_knowledge_semantics",
        fail_provider,
    )
    documents = build_docx_image_semantic_documents(
        document_path=tmp_path / "knowledge.docx",
        tenant_id="default",
        source_id="src_1",
        content_hash="b" * 64,
        starting_chunk_index=0,
    )
    assert documents == []
    assert calls == [7]


def test_build_documents_disabled_does_not_extract_images(monkeypatch, tmp_path):
    monkeypatch.setenv("IMAGE_SEMANTIC_ENABLED", "0")
    extractor = pytest.fail
    monkeypatch.setattr(image_semantics, "extract_docx_images", extractor)
    assert build_docx_image_semantic_documents(
        document_path=tmp_path / "knowledge.docx",
        tenant_id="default",
        source_id="src_1",
        content_hash="b" * 64,
        starting_chunk_index=0,
    ) == []


def test_exact_sha_match_wins_without_loading_perceptual_candidates(
    monkeypatch,
):
    monkeypatch.setenv("IMAGE_SEMANTIC_ENABLED", "1")
    known = Document(
        page_content="已知图片语义",
        metadata={"image_order": 7, "image_sha256": "x"},
    )
    store = _FakeVectorStore(exact_documents=[known])
    _patch_vector_store(monkeypatch, store)
    match = find_matching_image_semantics(
        _pattern_image_bytes(), tenant_id="tenant_a"
    )
    assert match is not None
    assert match.strategy == "exact_sha256"
    assert match.documents[0].page_content == "已知图片语义"
    assert len(store.calls) == 1


def test_reencoded_image_matches_with_coarse_and_detail_hash(monkeypatch):
    monkeypatch.setenv("IMAGE_SEMANTIC_ENABLED", "1")
    source_bytes = _pattern_image_bytes(image_format="PNG")
    reencoded = _pattern_image_bytes(image_format="JPEG", quality=75)
    candidate_fingerprint = fingerprint_image(reencoded)
    candidate = Document(
        page_content="重编码后仍是同一画面",
        metadata={
            "content_type": "image_semantic",
            "image_order": 9,
            "image_dhash": candidate_fingerprint.dhash,
            "image_detail_dhash": candidate_fingerprint.detail_dhash,
            "image_width": candidate_fingerprint.width,
            "image_height": candidate_fingerprint.height,
        },
    )
    store = _FakeVectorStore(semantic_documents=[candidate])
    _patch_vector_store(monkeypatch, store)
    match = find_matching_image_semantics(source_bytes)
    assert match is not None
    assert match.strategy == "perceptual_dhash"
    assert match.documents[0].metadata["image_order"] == 9


def test_detail_hash_prevents_same_layout_false_positive(monkeypatch):
    monkeypatch.setenv("IMAGE_SEMANTIC_ENABLED", "1")
    fingerprint = fingerprint_image(_pattern_image_bytes())
    candidate = Document(
        page_content="不同内容但粗略布局相似",
        metadata={
            "content_type": "image_semantic",
            "image_dhash": fingerprint.dhash,
            "image_detail_dhash": "f" * 256,
            "image_width": fingerprint.width,
            "image_height": fingerprint.height,
        },
    )
    store = _FakeVectorStore(semantic_documents=[candidate])
    _patch_vector_store(monkeypatch, store)
    assert find_matching_image_semantics(_pattern_image_bytes()) is None


def test_aspect_ratio_prevents_perceptual_false_positive(monkeypatch):
    monkeypatch.setenv("IMAGE_SEMANTIC_ENABLED", "1")
    fingerprint = fingerprint_image(_pattern_image_bytes())
    candidate = Document(
        page_content="比例不同",
        metadata={
            "content_type": "image_semantic",
            "image_dhash": fingerprint.dhash,
            "image_detail_dhash": fingerprint.detail_dhash,
            "image_width": 180,
            "image_height": 320,
        },
    )
    store = _FakeVectorStore(semantic_documents=[candidate])
    _patch_vector_store(monkeypatch, store)
    assert find_matching_image_semantics(_pattern_image_bytes()) is None


def test_disabled_matching_never_opens_vector_store(monkeypatch):
    monkeypatch.setenv("IMAGE_SEMANTIC_ENABLED", "0")
    def fail_getter(_tenant_id):
        pytest.fail("不应访问向量库")

    monkeypatch.setattr(
        image_semantics,
        "_get_image_metadata_collection",
        fail_getter,
    )
    assert find_matching_image_semantics(_pattern_image_bytes()) is None
