"""
测试 DOCX 图片提取功能。

主要确认：
1. 实际提取出 11 张图片。
2. 图片文件真实存在。
3. 图片顺序正确。
4. 图片内容与记录中的 SHA256 一致。
"""


# sha256 用来重新计算图片指纹。
from hashlib import sha256

# Path 用来检查提取后的图片文件。
from pathlib import Path


# 项目当前使用的 Word 文档路径。
from back.rag.loader import DOCUMENT_PATH

# 需要测试的图片提取函数。
from back.rag.docx_image_extractor import (
    extract_docx_images,
)


def test_extract_docx_images(
    tmp_path: Path,
):
    """
    当前 Word 应该提取出 11 张有效图片。

    tmp_path 是 pytest 自动创建的临时目录。
    测试图片不会写入正式数据目录。
    """

    # 在临时目录中准备图片输出位置。
    output_root = (
        tmp_path
        / "docx_images"
    )

    # 真正执行 DOCX 图片提取。
    records = extract_docx_images(
        document_path=DOCUMENT_PATH,
        output_root=output_root,
    )

    # 当前 Word 中一共有 11 张原图。
    assert len(records) == 11

    # 检查图片顺序是否为 1～11。
    image_orders = []

    for record in records:
        image_orders.append(
            record.image_order
        )

    assert image_orders == list(
        range(1, 12)
    )

    # 逐个检查图片记录。
    for record in records:
        # 把字符串路径转换成 Path。
        extracted_path = Path(
            record.extracted_path
        )

        # 图片必须已经保存。
        assert extracted_path.exists()

        # 保存结果必须是文件。
        assert extracted_path.is_file()

        # 图片文件不能是空的。
        assert extracted_path.stat().st_size > 0

        # 必须保留图片的 rId。
        assert record.relationship_id

        # DOCX 内部图片应该位于 word/media。
        assert record.media_path.startswith(
            "word/media/"
        )

        # 重新读取提取后的图片内容。
        image_bytes = extracted_path.read_bytes()

        # 重新计算图片 SHA256。
        actual_sha256 = sha256(
            image_bytes
        ).hexdigest()

        # 实际图片内容应该与记录中的指纹一致。
        assert actual_sha256 == (
            record.image_sha256
        )