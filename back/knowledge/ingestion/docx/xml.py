"""
DOCX XML 的公共配置。

Word 中不同 XML 文件都会使用命名空间。
统一放在这里，避免每个读取模块重复定义。
"""

# ElementTree 负责把 XML 原始内容解析成节点。
from xml.etree import ElementTree

# ZipFile 表示已经打开的 DOCX 压缩包。
from zipfile import ZipFile

def read_xml_root(
    archive: ZipFile,
    xml_path: str,
) -> ElementTree.Element:
    """
    从 DOCX 压缩包中读取并解析指定的 XML 文件。

    archive：
    已经打开的 DOCX 压缩包。

    xml_path：
    XML 文件在 DOCX 内部的位置，例如：
    word/document.xml
    word/_rels/document.xml.rels

    返回值：
    解析完成后的 XML 根节点。
    """

    try:
        # 根据传入的路径读取 XML 原始内容。
        xml_bytes = archive.read(xml_path)
    except KeyError as error:
        # 路径不存在时，说明 DOCX 文件结构可能不完整。
        raise ValueError(
            f"DOCX 中缺少必要文件：{xml_path}"
        ) from error

    try:
        # 把原始 XML 内容解析成可以查找的节点。
        xml_root = ElementTree.fromstring(
            xml_bytes
        )
    except ElementTree.ParseError as error:
        # 文件存在，但内容不是合法 XML 时进入这里。
        raise ValueError(
            f"DOCX XML 解析失败：{xml_path}"
        ) from error

    return xml_root


# XML 标签所属的命名空间。
#
# 可以简单理解为：
# w、r、a、wp 是下面这些长地址的简称。
XML_NAMESPACES = {
    "w": (
        "http://schemas.openxmlformats.org/"
        "wordprocessingml/2006/main"
    ),
    "r": (
        "http://schemas.openxmlformats.org/"
        "officeDocument/2006/relationships"
    ),
    "a": (
        "http://schemas.openxmlformats.org/"
        "drawingml/2006/main"
    ),
    "wp": (
        "http://schemas.openxmlformats.org/"
        "drawingml/2006/wordprocessingDrawing"
    ),
    "pr": (
        "http://schemas.openxmlformats.org/"
        "package/2006/relationships"
    ),
}

def xml_name(
        prefix:str,
        local_name:str,
) -> str:
    """
    生成 ElementTree 能识别的完整 XML 名称。
    例如 xml_name("w", "p") 表示 Word 段落标签。
    """

    namespace = XML_NAMESPACES[prefix]

    return f"{{{namespace}}}{local_name}"