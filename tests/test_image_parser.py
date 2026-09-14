"""PaddleOCR-VL 调用参数测试。"""

import json
from unittest.mock import Mock

import pytest

from back.knowledge.images import parser as image_parser


class _FakeResult:
    def save_to_json(self, save_path):
        pass

    def save_to_markdown(self, save_path):
        pass


class _FakePipeline:
    def __init__(self):
        self.predict_kwargs = None

    def predict(self, **kwargs):
        self.predict_kwargs = kwargs
        return [_FakeResult()]


def test_parse_image_passes_max_new_tokens(
    monkeypatch,
    tmp_path,
):
    """用户截图设置的生成上限必须传给 PaddleOCR-VL。"""

    image_path = tmp_path / "screenshot.png"
    image_path.write_bytes(b"test")
    pipeline = _FakePipeline()
    monkeypatch.setattr(
        image_parser,
        "get_image_parser",
        lambda: pipeline,
    )

    image_parser.parse_image(
        image_path,
        output_directory=tmp_path / "output",
        print_result=False,
        max_new_tokens=512,
    )

    assert pipeline.predict_kwargs == {
        "input": str(image_path.resolve()),
        "max_new_tokens": 512,
    }


def test_parse_image_keeps_default_for_reference_ocr(
    monkeypatch,
    tmp_path,
):
    """参考图批处理未指定上限时保持原来的模型默认值。"""

    image_path = tmp_path / "reference.png"
    image_path.write_bytes(b"test")
    pipeline = _FakePipeline()
    monkeypatch.setattr(
        image_parser,
        "get_image_parser",
        lambda: pipeline,
    )

    image_parser.parse_image(
        image_path,
        output_directory=tmp_path / "output",
        print_result=False,
    )

    assert pipeline.predict_kwargs == {
        "input": str(image_path.resolve()),
    }


def _write_reference_manifest(tmp_path, *, region_count=2):
    manifest_directory = tmp_path / "reference_test"
    manifest_directory.mkdir()
    manifest_path = manifest_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "reference_images": [
                    {
                        "image_order": 7,
                        "category": "subscription_error",
                        "title": "订阅错误",
                        "diagnostic_goal": "识别订阅错误",
                        "full_image_path": str(tmp_path / "full.png"),
                        "section_id": "4.1",
                        "section_title": "订阅",
                        "block_index": 2,
                        "previous_text": "前文",
                        "next_text": "后文",
                        "regions": [
                            {
                                "region_name": f"region_{index}",
                                "region_title": f"区域{index}",
                                "role": "primary",
                                "description": "错误区域",
                                "crop_path": str(tmp_path / f"crop_{index}.png"),
                                "normalized_box": [0, 0, 1, 1],
                                "pixel_box": [0, 0, 100, 100],
                            }
                            for index in range(region_count)
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_reference_ocr_uses_only_full_image_when_text_is_sufficient(
    monkeypatch,
    tmp_path,
):
    manifest = _write_reference_manifest(tmp_path)
    full = Mock(
        return_value={
            "status": "completed",
            "ocr_text": "无法获取订阅节点，请检查订阅地址",
        }
    )
    region = Mock()
    monkeypatch.setattr(image_parser, "_parse_full_reference_image", full)
    monkeypatch.setattr(image_parser, "_parse_manifest_region", region)

    results = image_parser.parse_reference_manifest(
        manifest,
        output_root=tmp_path / "ocr",
    )

    assert len(results) == 1
    assert results[0]["ocr_strategy"] == "full_image"
    assert results[0]["regions"] == []
    assert results[0]["combined_ocr_text"] == "无法获取订阅节点，请检查订阅地址"
    full.assert_called_once()
    region.assert_not_called()


def test_reference_ocr_uses_regions_only_when_full_text_is_sparse(
    monkeypatch,
    tmp_path,
):
    manifest = _write_reference_manifest(tmp_path)
    monkeypatch.setattr(
        image_parser,
        "_parse_full_reference_image",
        Mock(return_value={"status": "completed", "ocr_text": "错误"}),
    )
    region = Mock(
        side_effect=[
            {"role": "primary", "region_title": "区域0", "ocr_text": "订阅地址错误"},
            {"role": "primary", "region_title": "区域1", "ocr_text": "TLS验证失败"},
        ]
    )
    monkeypatch.setattr(image_parser, "_parse_manifest_region", region)

    results = image_parser.parse_reference_manifest(
        manifest,
        output_root=tmp_path / "ocr",
    )

    assert results[0]["ocr_strategy"] == "full_image_then_region_fallback"
    assert len(results[0]["regions"]) == 2
    assert "错误" not in results[0]["combined_ocr_text"].splitlines()
    assert "TLS验证失败" in results[0]["combined_ocr_text"]
    assert region.call_count == 2

    index = json.loads(
        (tmp_path / "ocr" / "reference_test" / "reference_ocr_index.json").read_text(
            encoding="utf-8"
        )
    )
    assert index["full_image_count"] == 1
    assert index["region_count"] == 2


def test_reference_full_image_failure_falls_back_to_regions(monkeypatch, tmp_path):
    manifest = _write_reference_manifest(tmp_path, region_count=1)
    monkeypatch.setattr(
        image_parser,
        "_parse_full_reference_image",
        Mock(side_effect=RuntimeError("full failed")),
    )
    monkeypatch.setattr(
        image_parser,
        "_parse_manifest_region",
        Mock(
            return_value={
                "role": "primary",
                "region_title": "错误区域",
                "ocr_text": "区域兜底识别成功",
            }
        ),
    )

    result = image_parser.parse_reference_manifest(
        manifest,
        output_root=tmp_path / "ocr",
    )[0]

    assert result["ocr_strategy"] == "full_image_then_region_fallback"
    assert result["full_image_result"]["status"] == "failed"
    assert result["combined_ocr_text"] == "区域兜底识别成功"


def test_reference_ocr_raises_when_full_and_regions_have_no_text(monkeypatch, tmp_path):
    manifest = _write_reference_manifest(tmp_path, region_count=1)
    monkeypatch.setattr(
        image_parser,
        "_parse_full_reference_image",
        Mock(return_value={"status": "completed", "ocr_text": ""}),
    )
    monkeypatch.setattr(
        image_parser,
        "_parse_manifest_region",
        Mock(return_value={"role": "primary", "region_title": "空区域", "ocr_text": ""}),
    )

    with pytest.raises(RuntimeError, match="没有识别到"):
        image_parser.parse_reference_manifest(
            manifest,
            output_root=tmp_path / "ocr",
        )


class _SerializableResult:
    @property
    def json(self):
        return {
            "res": {
                "parsing_res_list": [
                    {"block_content": "完整原图文字"},
                ]
            }
        }

    @property
    def markdown(self):
        return {"markdown_texts": "完整原图文字"}


def test_full_reference_result_is_cached_by_image_hash(monkeypatch, tmp_path):
    image_path = tmp_path / "full.png"
    image_path.write_bytes(b"image-v1")
    image_data = {
        "image_order": 7,
        "category": "subscription_error",
        "full_image_path": str(image_path),
    }
    parser = Mock(return_value=[_SerializableResult()])
    release = Mock()
    monkeypatch.setattr(image_parser, "parse_image", parser)
    monkeypatch.setattr(image_parser, "_release_region_result_memory", release)

    first = image_parser._parse_full_reference_image(
        image_data,
        tmp_path / "result",
        force=False,
    )
    second = image_parser._parse_full_reference_image(
        image_data,
        tmp_path / "result",
        force=False,
    )

    assert first["ocr_text"] == "完整原图文字"
    assert second == first
    assert parser.call_count == 1
    release.assert_called_once()


def test_force_reprocesses_cached_full_reference_image(monkeypatch, tmp_path):
    image_path = tmp_path / "full.png"
    image_path.write_bytes(b"image-v1")
    image_data = {
        "image_order": 7,
        "category": "subscription_error",
        "full_image_path": str(image_path),
    }
    parser = Mock(return_value=[_SerializableResult()])
    monkeypatch.setattr(image_parser, "parse_image", parser)
    monkeypatch.setattr(image_parser, "_release_region_result_memory", Mock())

    image_parser._parse_full_reference_image(
        image_data,
        tmp_path / "result",
        force=False,
    )
    image_parser._parse_full_reference_image(
        image_data,
        tmp_path / "result",
        force=True,
    )

    assert parser.call_count == 2


@pytest.mark.parametrize(
    ("text", "term_groups", "expected"),
    [
        ("Clash 导入 failed to fetch，TLS verifier error", [["failed", "失败"], ["tls", "fetch"]], True),
        ("Clash 导入页面显示737433591", [["failed", "失败"], ["tls", "fetch"]], False),
        ("公告中列出了新的订阅地址", [["订阅"], ["地址", "链接"]], True),
        ("公告中列出了新的备用内容", [["订阅"], ["地址", "链接"]], False),
        ("737433591", [], False),
        ("退出登录", [["退出登录", "退出登陆"]], True),
    ],
)
def test_reference_full_text_requires_quality_and_diagnostic_terms(
    text,
    term_groups,
    expected,
):
    image_data = {"required_ocr_term_groups": term_groups}
    assert image_parser._reference_full_text_is_sufficient(image_data, text) is expected
