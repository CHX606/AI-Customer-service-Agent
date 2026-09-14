"""
测试 document.xml.rels 关系读取功能。

主要确认：
document.xml 中找到的图片 rId，
都能在 document.xml.rels 中找到目标路径。
"""


# ZipFile 用来打开项目中的 DOCX 文件。
from zipfile import ZipFile


# 项目当前使用的 Word 文档路径。
from back.knowledge.ingestion.loader import DOCUMENT_PATH

# 读取正文顺序和图片 rId。
from back.knowledge.ingestion.docx.document_reader import (
    read_document_blocks,
)

# 读取 rId 与目标文件的对应关系。
from back.knowledge.ingestion.docx.relationship_reader import (
    read_relationship_targets,
)


def test_image_relationship_targets():
    """
    当前 Word 中的 11 个图片 rId，
    应该都能找到对应的内部图片路径。
    """

    # 打开一次 DOCX，然后交给两个读取模块分别处理。
    with ZipFile(DOCUMENT_PATH) as archive:
        # 从 document.xml 中读取图片 rId。
        blocks = read_document_blocks(
            archive
        )

        # 从 document.xml.rels 中读取关系字典。
        relationships = (
            read_relationship_targets(
                archive
            )
        )

    # 收集 document.xml 中的所有图片引用。
    image_references = []

    for block in blocks:
        for image_reference in block.images:
            image_references.append(
                image_reference
            )

    # 当前 Word 中应该有 11 个图片引用。
    assert len(image_references) == 11

    # 逐个检查图片 rId 是否能找到目标路径。
    for image_reference in image_references:
        relationship_id = (
            image_reference.relationship_id
        )

        # 使用 rId 从关系字典中查找目标。
        target = relationships.get(
            relationship_id
        )

        # 如果得到 None，说明这个 rId 没有对应关系。
        assert target is not None

        # 目标路径不能是空字符串。
        assert target.target

        # 当前 11 张图片应该都保存在 DOCX 内部，
        # 不应该是需要联网下载的外部图片。
        assert (
            target.target_mode.lower()
            != "external"
        )

        # 当前 Word 内部图片的目标路径，
        # 应该以 media/ 开头。
        assert target.target.startswith(
            "media/"
        )