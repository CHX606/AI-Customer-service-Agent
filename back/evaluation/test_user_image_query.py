"""用户截图通用分区策略测试。"""


from io import BytesIO

from PIL import Image
import pytest

from back.rag import user_image_query
from back.rag.user_image_query import (
    UserImageValidationError,
    _build_query_regions,
    _load_and_validate_image,
)


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
