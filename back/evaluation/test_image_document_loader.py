"""图片诊断结果转换为 LangChain Document 的回归测试。"""


import json

from back.rag.image_document_loader import (
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
    assert document.metadata["content_type"] == (
        "image_diagnostic"
    )
    assert document.metadata["image_order"] == 7
    assert document.metadata["section_id"] == "4.3"
    assert document.metadata["chunk_index"] == 0
