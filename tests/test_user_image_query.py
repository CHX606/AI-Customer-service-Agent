"""用户截图通用分区策略测试。"""


from io import BytesIO

from PIL import Image
import pytest
from langchain_core.documents import Document

from back.knowledge.images import query as user_image_query
from back.knowledge.images.query import (
    UserImageValidationError,
    _build_query_regions,
    _load_and_validate_image,
)
from back.knowledge.images.text_cleaner import is_ocr_text_sufficient
from back.knowledge.images.semantics import (
    CustomerImageUnderstanding,
    ImageSemanticMatch,
)


def _image_bytes(size=(100, 100), image_format="PNG"):
    buffer = BytesIO()
    Image.new("RGB", size, color="white").save(buffer, format=image_format)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("size", "expected_layout", "expected_count"),
    [
        ((600, 600), "single", 1),
        ((1400, 700), "landscape_columns", 3),
        ((700, 1400), "portrait_rows", 3),
        ((1000, 1000), "square_grid", 4),
    ],
)
def test_build_query_regions(
    size,
    expected_layout,
    expected_count,
):
    """不同宽高比例应进入对应的完整覆盖策略。"""

    layout, regions = _build_query_regions(*size)

    assert layout == expected_layout
    assert len(regions) == expected_count


def test_load_and_validate_image_accepts_png():
    """有效 PNG 应被转换为统一 RGB 图片。"""

    buffer = BytesIO()
    Image.new("RGBA", (100, 100)).save(
        buffer,
        format="PNG",
    )

    image = _load_and_validate_image(
        buffer.getvalue()
    )

    try:
        assert image.mode == "RGB"
        assert image.size == (100, 100)
    finally:
        image.close()


def test_load_and_validate_image_rejects_non_image():
    """伪装成上传内容的普通文本不能进入 OCR。"""

    with pytest.raises(
        UserImageValidationError,
        match="不是有效图片",
    ):
        _load_and_validate_image(b"not an image")


def test_analyze_user_image_limits_generated_tokens(
    monkeypatch,
):
    """用户截图 OCR 必须限制模型生成长度。"""

    buffer = BytesIO()
    Image.new("RGB", (100, 100)).save(
        buffer,
        format="PNG",
    )
    captured_kwargs = {}

    def fake_parse_image(
        image_path,
        output_directory,
        print_result,
        max_new_tokens,
    ):
        captured_kwargs["max_new_tokens"] = max_new_tokens
        return [object()]

    monkeypatch.setattr(
        user_image_query,
        "parse_image",
        fake_parse_image,
    )
    monkeypatch.setattr(
        user_image_query,
        "extract_ocr_text",
        lambda results: "不能获取订阅节点",
    )
    monkeypatch.setattr(
        user_image_query,
        "release_result_memory",
        lambda results: None,
    )

    analysis = user_image_query.analyze_user_image(
        buffer.getvalue()
    )

    assert captured_kwargs["max_new_tokens"] == 512
    assert analysis.ocr_text == "不能获取订阅节点"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("不能获取订阅节点", True),
        ("TLS error", True),
        ("错误", False),
        ("  \n ", False),
        ("１２７.０.０.１", False),
    ],
)
def test_full_image_ocr_sufficiency_heuristic(text, expected):
    assert is_ocr_text_sufficient(text) is expected


def test_large_image_with_sufficient_full_ocr_is_not_split(monkeypatch):
    parse_calls = []
    released = []

    def fake_parse(image_path, **_kwargs):
        with Image.open(image_path) as parsed_image:
            parse_calls.append(parsed_image.size)
        return ["full-result"]

    monkeypatch.setattr(user_image_query, "parse_image", fake_parse)
    monkeypatch.setattr(
        user_image_query,
        "extract_ocr_text",
        lambda _results: "无法获取订阅节点，请检查地址",
    )
    monkeypatch.setattr(
        user_image_query,
        "release_result_memory",
        lambda results: released.append(results),
    )

    analysis = user_image_query.analyze_user_image(_image_bytes((1400, 700)))

    assert parse_calls == [(1400, 700)]
    assert len(released) == 1
    assert analysis.layout == "full_image"
    assert analysis.region_count == 1
    assert analysis.ocr_text == "无法获取订阅节点，请检查地址"


def test_sparse_large_full_ocr_falls_back_to_overlapping_regions(monkeypatch):
    texts = iter(["错误", "左侧订阅信息", "中间错误提示", "右侧节点状态"])
    parse_calls = []

    def fake_parse(image_path, **_kwargs):
        with Image.open(image_path) as parsed_image:
            parse_calls.append(parsed_image.size)
        return [len(parse_calls)]

    monkeypatch.setattr(user_image_query, "parse_image", fake_parse)
    monkeypatch.setattr(
        user_image_query,
        "extract_ocr_text",
        lambda _results: next(texts),
    )
    monkeypatch.setattr(user_image_query, "release_result_memory", lambda _results: None)

    analysis = user_image_query.analyze_user_image(_image_bytes((1400, 700)))

    assert parse_calls[0] == (1400, 700)
    assert parse_calls[1:] == [(560, 700), (560, 700), (560, 700)]
    assert analysis.layout == "full_then_landscape_columns"
    assert analysis.region_count == 4
    assert "错误" in analysis.ocr_text
    assert "左侧订阅信息" in analysis.ocr_text
    assert "右侧节点状态" in analysis.ocr_text


def test_full_image_failure_on_large_image_uses_region_fallback(monkeypatch):
    call_count = 0

    def fake_parse(_image_path, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("full image failed")
        return [call_count]

    monkeypatch.setattr(user_image_query, "parse_image", fake_parse)
    monkeypatch.setattr(
        user_image_query,
        "extract_ocr_text",
        lambda results: f"区域识别结果{results[0]}",
    )
    monkeypatch.setattr(user_image_query, "release_result_memory", lambda _results: None)

    analysis = user_image_query.analyze_user_image(_image_bytes((700, 1400)))

    assert call_count == 4
    assert analysis.layout == "full_then_portrait_rows"
    assert analysis.region_count == 4
    assert "区域识别结果2" in analysis.ocr_text


def test_one_failed_fallback_region_does_not_discard_other_results(monkeypatch):
    call_count = 0

    def fake_parse(_image_path, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            raise RuntimeError("middle region failed")
        return [call_count]

    monkeypatch.setattr(user_image_query, "parse_image", fake_parse)
    monkeypatch.setattr(
        user_image_query,
        "extract_ocr_text",
        lambda results: "短" if results[0] == 1 else f"有效区域文字{results[0]}",
    )
    monkeypatch.setattr(user_image_query, "release_result_memory", lambda _results: None)

    analysis = user_image_query.analyze_user_image(_image_bytes((700, 1400)))

    assert call_count == 4
    assert "有效区域文字2" in analysis.ocr_text
    assert "有效区域文字4" in analysis.ocr_text


def test_fallback_deduplicates_full_and_overlapping_region_text(monkeypatch):
    texts = iter(["错误", "相同的错误提示内容", "相同的错误提示内容", "节点状态正常"])
    monkeypatch.setattr(user_image_query, "parse_image", lambda *_args, **_kwargs: [object()])
    monkeypatch.setattr(user_image_query, "extract_ocr_text", lambda _results: next(texts))
    monkeypatch.setattr(user_image_query, "release_result_memory", lambda _results: None)

    analysis = user_image_query.analyze_user_image(_image_bytes((1400, 700)))

    assert analysis.ocr_text.count("相同的错误提示内容") == 1


def test_small_image_failure_is_not_retried_with_same_full_region(monkeypatch):
    calls = []

    def fail_parse(*_args, **_kwargs):
        calls.append(1)
        raise RuntimeError("small image failed")

    monkeypatch.setattr(user_image_query, "parse_image", fail_parse)

    with pytest.raises(RuntimeError, match="small image failed"):
        user_image_query.analyze_user_image(_image_bytes((100, 100)))

    assert len(calls) == 1


def test_result_memory_is_released_when_text_extraction_fails(monkeypatch):
    results = [object()]
    released = []
    monkeypatch.setattr(user_image_query, "parse_image", lambda *_args, **_kwargs: results)
    monkeypatch.setattr(
        user_image_query,
        "extract_ocr_text",
        lambda _results: (_ for _ in ()).throw(RuntimeError("extract failed")),
    )
    monkeypatch.setattr(
        user_image_query,
        "release_result_memory",
        lambda value: released.append(value),
    )

    with pytest.raises(RuntimeError, match="extract failed"):
        user_image_query.analyze_user_image(_image_bytes((100, 100)))

    assert released == [results]


def test_known_image_match_skips_visual_model_and_ocr(monkeypatch):
    monkeypatch.setenv("IMAGE_SEMANTIC_ENABLED", "1")
    match = ImageSemanticMatch(
        strategy="exact_sha256",
        distance=0,
        documents=(
            Document(
                page_content="图片含义：不能获取订阅节点；处理方式：检查套餐。",
                metadata={"image_order": 7},
            ),
        ),
    )
    monkeypatch.setattr(
        user_image_query,
        "find_matching_image_semantics",
        lambda *_args, **_kwargs: match,
    )
    monkeypatch.setattr(
        user_image_query,
        "understand_customer_image",
        lambda *_args, **_kwargs: pytest.fail("已知图不应调用视觉模型"),
    )
    monkeypatch.setattr(
        user_image_query,
        "parse_image",
        lambda *_args, **_kwargs: pytest.fail("已知图不应调用 OCR"),
    )

    analysis = user_image_query.analyze_user_image(
        _image_bytes(),
        user_message="这个怎么办",
        tenant_id="tenant_a",
    )

    assert analysis.understanding_strategy == "exact_sha256"
    assert analysis.matched_image_order == 7
    assert analysis.ocr_text == ""
    assert "检查套餐" in analysis.semantic_text


def test_unknown_image_uses_visual_semantics_without_ocr(monkeypatch):
    monkeypatch.setenv("IMAGE_SEMANTIC_ENABLED", "1")
    monkeypatch.setattr(
        user_image_query,
        "find_matching_image_semantics",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        user_image_query,
        "understand_customer_image",
        lambda *_args, **_kwargs: CustomerImageUnderstanding(
            summary="Clash 订阅页显示 TLS 报错",
            visible_evidence=["failed to fetch"],
            likely_problem="订阅导入失败",
            search_queries=["Clash TLS 导入失败"],
            confidence=0.91,
        ),
    )
    monkeypatch.setattr(
        user_image_query,
        "parse_image",
        lambda *_args, **_kwargs: pytest.fail("高置信视觉结果不应再 OCR"),
    )
    analysis = user_image_query.analyze_user_image(
        _image_bytes(), user_message="导不进去"
    )
    assert analysis.understanding_strategy == "vision_model"
    assert analysis.layout == "vision_full_image"
    assert "Clash TLS 导入失败" in analysis.semantic_text


def test_low_confidence_visual_result_falls_back_to_ocr(monkeypatch):
    monkeypatch.setenv("IMAGE_SEMANTIC_ENABLED", "1")
    monkeypatch.setattr(
        user_image_query,
        "find_matching_image_semantics",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        user_image_query,
        "understand_customer_image",
        lambda *_args, **_kwargs: CustomerImageUnderstanding(
            summary="看不清的页面",
            confidence=0.1,
        ),
    )
    monkeypatch.setattr(
        user_image_query,
        "_recognize_query_region",
        lambda *_args, **_kwargs: "不能获取订阅节点",
    )
    analysis = user_image_query.analyze_user_image(_image_bytes())
    assert analysis.understanding_strategy == "ocr"
    assert analysis.ocr_text == "不能获取订阅节点"


def test_match_failure_continues_with_visual_model(monkeypatch):
    monkeypatch.setenv("IMAGE_SEMANTIC_ENABLED", "1")

    def fail_match(*_args, **_kwargs):
        raise RuntimeError("vector store unavailable")

    monkeypatch.setattr(
        user_image_query, "find_matching_image_semantics", fail_match
    )
    monkeypatch.setattr(
        user_image_query,
        "understand_customer_image",
        lambda *_args, **_kwargs: CustomerImageUnderstanding(
            summary="Windows 代理设置页",
            confidence=0.8,
        ),
    )
    analysis = user_image_query.analyze_user_image(_image_bytes())
    assert analysis.understanding_strategy == "vision_model"


def test_visual_model_failure_falls_back_to_ocr(monkeypatch):
    monkeypatch.setenv("IMAGE_SEMANTIC_ENABLED", "1")
    monkeypatch.setattr(
        user_image_query,
        "find_matching_image_semantics",
        lambda *_args, **_kwargs: None,
    )

    def fail_vision(*_args, **_kwargs):
        raise RuntimeError("vision unavailable")

    monkeypatch.setattr(user_image_query, "understand_customer_image", fail_vision)
    monkeypatch.setattr(
        user_image_query,
        "_recognize_query_region",
        lambda *_args, **_kwargs: "代理端口设置错误",
    )
    analysis = user_image_query.analyze_user_image(_image_bytes())
    assert analysis.understanding_strategy == "ocr"
    assert analysis.ocr_text == "代理端口设置错误"


def test_visual_provider_failure_temporarily_skips_repeated_slow_calls(monkeypatch):
    monkeypatch.setenv("IMAGE_SEMANTIC_ENABLED", "1")
    monkeypatch.setenv("IMAGE_VISION_FAILURE_COOLDOWN_SECONDS", "120")
    user_image_query.reset_image_vision_circuit()
    monkeypatch.setattr(
        user_image_query,
        "find_matching_image_semantics",
        lambda *_args, **_kwargs: None,
    )
    calls = []

    def fail_vision(*_args, **_kwargs):
        calls.append(1)
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(user_image_query, "understand_customer_image", fail_vision)
    monkeypatch.setattr(
        user_image_query,
        "_recognize_query_region",
        lambda *_args, **_kwargs: "OCR 兜底内容",
    )

    first = user_image_query.analyze_user_image(_image_bytes())
    second = user_image_query.analyze_user_image(_image_bytes())

    assert first.ocr_text == "OCR 兜底内容"
    assert second.ocr_text == "OCR 兜底内容"
    assert len(calls) == 1


def test_semantic_agent_message_uses_meaning_instead_of_empty_ocr():
    analysis = user_image_query.UserImageAnalysis(
        width=320,
        height=180,
        layout="knowledge_image_match",
        region_count=0,
        ocr_text="",
        understanding_strategy="exact_sha256",
        semantic_text="图片含义：订阅导入失败",
        matched_image_order=9,
        match_distance=0,
    )
    message = user_image_query.build_image_agent_message("怎么办", analysis)
    assert "预处理图片语义" in message
    assert "图片含义：订阅导入失败" in message
    assert "截图 OCR 结果" not in message


def test_ocr_agent_message_remains_available_as_fallback():
    analysis = user_image_query.UserImageAnalysis(
        width=320,
        height=180,
        layout="full_image",
        region_count=1,
        ocr_text="failed to fetch",
    )
    message = user_image_query.build_image_agent_message("", analysis)
    assert "截图 OCR 结果" in message
    assert "failed to fetch" in message
