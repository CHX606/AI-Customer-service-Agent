"""图片诊断结果转换为 LangChain Document 的回归测试。"""


import json
from io import BytesIO

from PIL import Image

from back.knowledge.images.documents import (
    load_image_documents,
)


def test_load_image_documents(tmp_path):
    """图片正文、URL、token 和扁平 metadata 应完整保留。"""

    reference_directory = (
        tmp_path / "reference_test"
    )
    image_directory = (
        reference_directory / "image_007"
    )
    image_directory.mkdir(parents=True)

    result_path = (
        image_directory / "diagnostic_result.json"
    )
    result_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "image_order": 7,
                "category": "subscription_error",
                "title": "订阅导入失败",
                "diagnostic_goal": "定位订阅导入错误",
                "full_image_path": "image_007.png",
                "section_id": "4.3",
                "section_title": "导入",
                "block_index": 10,
                "previous_text": "上文说明该错误通常和套餐状态有关。",
                "next_text": "下文要求检查浏览器和订阅地址。",
                "full_image_result": {
                    "image_sha256": "abc123",
                },
                "combined_ocr_text": (
                    "不能获取订阅节点\n"
                    "https://example.com/config?token=abc"
                ),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    index_path = (
        reference_directory
        / "reference_ocr_index.json"
    )
    index_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "image_count": 1,
                "region_count": 3,
                "images": [
                    {
                        "image_order": 7,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    documents = load_image_documents(
        index_path=index_path
    )

    assert len(documents) == 1

    document = documents[0]

    assert "订阅导入失败" in document.page_content
    assert "不能获取订阅节点" in document.page_content
    assert "token=abc" in document.page_content
    assert "上文说明" in document.page_content
    assert "下文要求" in document.page_content
    assert document.metadata["content_type"] == (
        "image_diagnostic"
    )
    assert document.metadata["image_order"] == 7
    assert document.metadata["section_id"] == "4.3"
    assert document.metadata["chunk_index"] == 0
    assert document.metadata["image_sha256"] == "abc123"


def test_existing_full_image_adds_perceptual_fingerprints(tmp_path):
    reference_directory = tmp_path / "reference_test"
    image_directory = reference_directory / "image_007"
    image_directory.mkdir(parents=True)
    full_image_path = tmp_path / "image_007.png"
    buffer = BytesIO()
    Image.new("RGB", (160, 100), color="white").save(buffer, format="PNG")
    full_image_path.write_bytes(buffer.getvalue())

    (image_directory / "diagnostic_result.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "image_order": 7,
                "category": "test",
                "title": "测试图片",
                "diagnostic_goal": "测试图片语义",
                "full_image_path": str(full_image_path),
                "combined_ocr_text": "可以用于检索的图片文字",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    index_path = reference_directory / "reference_ocr_index.json"
    index_path.write_text(
        json.dumps(
            {"status": "completed", "image_count": 1, "images": [{"image_order": 7}]}
        ),
        encoding="utf-8",
    )

    document = load_image_documents(index_path=index_path)[0]
    assert len(document.metadata["image_sha256"]) == 64
    assert len(document.metadata["image_dhash"]) == 16
    assert len(document.metadata["image_detail_dhash"]) == 256
    assert document.metadata["image_width"] == 160
    assert document.metadata["image_height"] == 100
