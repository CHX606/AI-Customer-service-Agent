"""
测试 document.xml 正文顺序读取功能。

这里使用项目现有的 Word 文档进行测试，
确认正文顺序和图片引用能够被正常读取。
"""


# ZipFile 用来打开 DOCX 压缩包。
from zipfile import ZipFile

# 项目当前使用的 Word 文档路径。
from back.rag.loader import DOCUMENT_PATH

# 这是我们刚刚写完的正文读取函数。
from back.rag.docx_document_reader import (
    read_document_blocks,
)


def test_read_document_blocks():
    """
    document.xml 应该能够生成有序正文块，
    并找到当前 Word 中的 11 张图片引用。
    """

    # 打开项目当前使用的 DOCX 文件。
    with ZipFile(DOCUMENT_PATH) as archive:
        # 真正调用正文读取函数。
        blocks = read_document_blocks(
            archive
        )

    # 正常的 Word 文档应该读取出正文块。
    assert blocks

    # 检查 block_index 是否从 1 开始连续编号。
    block_indexes = [
        block.block_index
        for block in blocks
    ]

    assert block_indexes == list(
        range(1, len(blocks) + 1)
    )

    # 收集所有正文块中的图片引用。
    image_references = []

    for block in blocks:
        for image_reference in block.images:
            image_references.append(
                image_reference
            )

    # 当前 Word 原图一共有 11 张，
    # 所以应该找到 11 个图片引用。
    assert len(image_references) == 11

    # 每张图片都应该存在 rId7 之类的关系编号。
    for image_reference in image_references:
        assert image_reference.relationship_id

    # 当前支持这三种位置类型。
    for image_reference in image_references:
        assert image_reference.position_type in {
            "inline",
            "anchor",
            "unknown",
        }