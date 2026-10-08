"""近似截图必须理解当前开关状态，不能直接复用知识图片里的旧状态。"""

from io import BytesIO
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document
from PIL import Image

from back.knowledge.images import query
from back.knowledge.images.semantics import CustomerImageUnderstanding, ImageSemanticMatch


@pytest.fixture(autouse=True)
def image_semantics_mode(monkeypatch):
    monkeypatch.setenv("IMAGE_FEATURES_ENABLED", "1")
    monkeypatch.setenv("IMAGE_SEMANTIC_ENABLED", "1")
    monkeypatch.setenv("LOCAL_OCR_ENABLED", "1")
    query.reset_image_vision_circuit()


def image_bytes():
    buffer = BytesIO()
    with Image.new("RGB", (160, 120), "white") as image:
        image.save(buffer, format="PNG")
    return buffer.getvalue()


def known_match(strategy="perceptual_dhash", distance=1):
    return ImageSemanticMatch(
        strategy=strategy, distance=distance, documents=(
            Document(
                page_content="历史知识图片：虚拟网卡开关已开启。",
                metadata={"image_order": 7},
            ),
        ),
    )


def current_card(confidence=0.9):
    return CustomerImageUnderstanding(
        summary="当前截图的虚拟网卡开关关闭",
        visible_evidence=["虚拟网卡开关呈灰色，处于关闭状态"],
        likely_problem="当前虚拟网卡未开启",
        search_queries=["虚拟网卡关闭时系统代理无法上网"],
        confidence=confidence,
    )


@pytest.mark.parametrize("distance", [0, 1, 4])
def test_perceptual_match_understands_current_image_instead_of_reusing_old_switch(monkeypatch, distance):
    content = image_bytes()
    match = Mock(return_value=known_match(distance=distance))
    vision = Mock(return_value=current_card())
    monkeypatch.setattr(query, "find_matching_image_semantics", match)
    monkeypatch.setattr(query, "understand_customer_image", vision)
    monkeypatch.setattr(query, "parse_image", Mock(side_effect=AssertionError("高置信视觉无需 OCR")))

    result = query.analyze_user_image(content, user_message="开了就没有网络", tenant_id="tenant_a")

    match.assert_called_once_with(content, tenant_id="tenant_a")
    vision.assert_called_once_with(content, user_message="开了就没有网络")
    assert result.understanding_strategy == "vision_model"
    assert "关闭" in result.semantic_text
    assert "历史知识图片" not in result.semantic_text
    assert "已开启" not in result.semantic_text
    assert result.matched_image_order is None
    assert result.match_distance is None
    assert result.ocr_text == ""


def test_exact_match_keeps_fast_path_even_when_vision_circuit_is_open(monkeypatch):
    monkeypatch.setattr(query, "find_matching_image_semantics", Mock(return_value=
                        known_match("exact_sha256", 0)))
    vision = Mock(side_effect=AssertionError("相同文件无需视觉"))
    ocr = Mock(side_effect=AssertionError("相同文件无需 OCR"))
    monkeypatch.setattr(query, "understand_customer_image", vision)
    monkeypatch.setattr(query, "parse_image", ocr)
    query._record_vision_failure()

    result = query.analyze_user_image(image_bytes())

    assert result.understanding_strategy == "exact_sha256"
    assert result.matched_image_order == 7
    assert result.match_distance == 0
    assert "历史知识图片" in result.semantic_text
    vision.assert_not_called()
    ocr.assert_not_called()


@pytest.mark.parametrize("failure", ["low_confidence", "provider", "circuit"])
def test_near_match_vision_failure_uses_only_current_ocr(monkeypatch, failure):
    monkeypatch.setattr(query, "find_matching_image_semantics", Mock(return_value=known_match()))
    vision = Mock(return_value=current_card(confidence=0.1))
    if failure == "provider":
        vision.side_effect = RuntimeError("vision unavailable")
    elif failure == "circuit":
        query._record_vision_failure()
    monkeypatch.setattr(query, "understand_customer_image", vision)
    recognize = Mock(return_value="当前 OCR：系统代理打开，虚拟网卡文字")
    monkeypatch.setattr(query, "_recognize_query_region", recognize)

    result = query.analyze_user_image(image_bytes())

    assert result.understanding_strategy == "ocr"
    assert result.semantic_text == ""
    assert result.ocr_text == "当前 OCR：系统代理打开，虚拟网卡文字"
    assert result.matched_image_order is None
    assert result.match_distance is None
    recognize.assert_called_once()
    if failure == "circuit":
        vision.assert_not_called()
    else:
        vision.assert_called_once()


@pytest.mark.parametrize("failure", ["low_confidence", "provider", "circuit"])
def test_near_match_without_allowed_ocr_does_not_return_cached_state(monkeypatch, failure):
    monkeypatch.setenv("LOCAL_OCR_ENABLED", "0")
    monkeypatch.setattr(query, "find_matching_image_semantics", Mock(return_value=known_match()))
    vision = Mock(return_value=current_card(confidence=0.1))
    if failure == "provider":
        vision.side_effect = RuntimeError("vision unavailable")
    elif failure == "circuit":
        query._record_vision_failure()
    monkeypatch.setattr(query, "understand_customer_image", vision)
    ocr = Mock(side_effect=AssertionError("未启用本地 OCR"))
    monkeypatch.setattr(query, "parse_image", ocr)

    with pytest.raises(RuntimeError, match="图片 API"):
        query.analyze_user_image(image_bytes())

    ocr.assert_not_called()


def test_low_confidence_near_match_ocr_failure_is_not_replaced_with_old_state(monkeypatch):
    monkeypatch.setattr(query, "find_matching_image_semantics", Mock(return_value=known_match()))
    monkeypatch.setattr(query, "understand_customer_image", Mock(return_value=current_card(0.1)))
    monkeypatch.setattr(query, "_recognize_query_region", Mock(side_effect=RuntimeError("OCR failed")))

    with pytest.raises(RuntimeError, match="OCR failed"):
        query.analyze_user_image(image_bytes())

