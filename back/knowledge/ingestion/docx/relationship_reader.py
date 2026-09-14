"""
Word document.xml.rels 关系读取模块。

这个文件只负责：
1. 读取 document.xml.rels。
2. 找到 rId 与目标路径的对应关系。
3. 判断目标是 DOCX 内部文件还是外部链接。

这个文件不会：
- 读取 document.xml 正文。
- 提取或保存真实图片。
- 调用 OCR。
- 写入 OpenSearch。
"""


# ZipFile 表示已经打开的 DOCX 压缩包。
from zipfile import ZipFile


# RelationshipTarget 用来保存关系对应的目标信息。
from back.knowledge.ingestion.docx.models import (
    RelationshipTarget,
)

# 使用公共 XML 工具。
#
# XML_NAMESPACES 用来识别 pr:Relationship。
# read_xml_root 负责读取并解析指定的 XML 文件。
from back.knowledge.ingestion.docx.xml import (
    XML_NAMESPACES,
    read_xml_root,
)


# document.xml.rels 在 DOCX 压缩包中的固定位置。
DOCUMENT_RELS_PATH = (
    "word/_rels/document.xml.rels"
)


def read_relationship_targets(
    archive: ZipFile,
) -> dict[str, RelationshipTarget]:
    """
    读取 document.xml.rels 中的关系记录。

    archive：
    已经打开的 DOCX 压缩包。

    返回值：
    一个以 rId 为键的字典。

    例如：
    {
        "rId7": RelationshipTarget(
            target="media/image7.png",
            target_mode="Internal",
        )
    }
    """

    # 调用公共 XML 工具，
    # 读取并解析 document.xml.rels。
    relationships_root = read_xml_root(
        archive,
        DOCUMENT_RELS_PATH,
    )

    # 创建空字典，用来保存：
    #
    # rId → RelationshipTarget
    targets = {}

    # document.xml.rels 中每条关系都是
    # 一个 pr:Relationship 节点。
    relationship_nodes = relationships_root.findall(
        "pr:Relationship",
        XML_NAMESPACES,
    )

    # 每次循环处理一条关系。
    for relationship in relationship_nodes:
        # attrib 是 XML 节点的属性字典。
        #
        # 例如：
        # <Relationship Id="rId7" Target="media/image7.png"/>
        #
        # relationship.attrib 大致是：
        # {
        #     "Id": "rId7",
        #     "Target": "media/image7.png"
        # }
        relationship_id = relationship.attrib.get(
            "Id",
            "",
        )

        target = relationship.attrib.get(
            "Target",
            "",
        )

        # Id 或 Target 缺失时，
        # 这条关系无法正常使用，所以跳过。
        if not relationship_id or not target:
            continue

        # TargetMode 可能不存在。
        #
        # 不存在时通常表示目标位于 DOCX 内部，
        # 所以默认使用 Internal。
        target_mode = relationship.attrib.get(
            "TargetMode",
            "Internal",
        )

        # 使用 rId 作为字典的键，
        # 保存它对应的目标路径和目标类型。
        targets[relationship_id] = (
            RelationshipTarget(
                target=target,
                target_mode=target_mode,
            )
        )

    return targets
