"""
Word document.xml 正文顺序处理模块。

这个文件只负责：
1. 调用公共 XML 工具读取 document.xml。
2. 按原始顺序处理段落和表格。
3. 提取正文文字。
4. 找出正文中的图片引用编号。

公共的 XML 文件读取、解析和命名空间配置，
统一放在 docx_xml.py 中，避免其他模块重复实现。

这个文件不会：
- 读取 document.xml.rels。
- 提取真实图片文件。
- 调用 OCR。
- 写入 Chroma。
"""


# ElementTree 在这里主要用于说明参数的类型。
#
# document.xml 的实际读取和 XML 解析工作，
# 已经交给 docx_xml.py 中的 read_xml_root()。
from xml.etree import ElementTree

# DOCX 本质上是 ZIP 压缩包。
#
# 当前文件接收一个已经打开的 ZipFile 对象，
# 然后把它交给公共 XML 读取函数。
from zipfile import ZipFile


# 导入正文读取完成后需要返回的数据结构。
#
# DocumentBlock 表示一个有顺序的正文块。
# ImageReference 表示正文中的一张图片引用。
from back.rag.docx_models import (
    DocumentBlock,
    ImageReference,
)

# 导入 DOCX XML 的公共工具。
#
# XML_NAMESPACES：
# 保存 w、r、a、wp 等简称对应的完整 XML 地址。
#
# read_xml_root：
# 从已经打开的 DOCX 中读取并解析指定 XML 文件。
#
# xml_name：
# 生成 ElementTree 能够识别的完整 XML 标签名。
from back.rag.docx_xml import (
    XML_NAMESPACES,
    read_xml_root,
    xml_name,
)


# document.xml 在 DOCX 压缩包中的固定位置。
#
# 当前模块把这个路径交给 read_xml_root()，
# 告诉公共工具具体需要读取哪个 XML 文件。
DOCUMENT_XML_PATH = "word/document.xml"


# w:p 表示 Word 正文中的段落。
PARAGRAPH_TAG = xml_name(
    "w",
    "p",
)


# w:tbl 表示 Word 正文中的表格。
TABLE_TAG = xml_name(
    "w",
    "tbl",
)


# r:embed 保存内嵌图片的关系编号。
#
# 例如：
# <a:blip r:embed="rId7"/>
RELATIONSHIP_EMBED_ATTRIBUTE = xml_name(
    "r",
    "embed",
)


# r:link 保存外部链接图片的关系编号。
RELATIONSHIP_LINK_ATTRIBUTE = xml_name(
    "r",
    "link",
)


def _extract_text(
    element: ElementTree.Element,
) -> str:
    """
    提取一个段落或表格中的所有可见文字。

    参数 element：
    一个段落 w:p 或一个表格 w:tbl。

    返回值：
    合并并清理后的文字。
    如果没有文字，就返回空字符串。
    """

    # Word 中真正保存文字的标签是 w:t。
    #
    # 一个看起来完整的句子，在 XML 中可能被拆成多个 w:t。
    # 所以不能只读取第一个，需要把它们全部找出来。
    text_nodes = element.findall(
        ".//w:t",
        XML_NAMESPACES,
    )

    # 按照 XML 中的原始顺序收集文字。
    #
    # 有些 w:t 节点可能没有内容，
    # 所以使用 if node.text 排除空内容。
    text_parts = [
        node.text
        for node in text_nodes
        if node.text
    ]

    # 把被 Word 拆开的多段文字重新连接起来。
    #
    # 这里不能使用 " ".join(text_parts)，
    # 因为 Word 可能把一个词拆成多个节点。
    # 如果强行添加空格，原文可能会被改变。
    text = "".join(text_parts)

    # strip() 删除文字开头和结尾多余的空白。
    return text.strip()


def _get_image_position_type(
    drawing: ElementTree.Element,
) -> str:
    """
    判断一张图片在 Word 中的位置类型。

    drawing 是 document.xml 中的一张图片节点。

    返回值有三种：
    inline：图片像文字一样嵌入在段落中。
    anchor：图片浮动在文字上方或周围。
    unknown：没有识别出具体类型。
    """

    # 在当前图片节点下面查找 wp:inline。
    #
    # find() 找到节点时返回 XML 节点对象；
    # 没找到时返回 None。
    inline_node = drawing.find(
        ".//wp:inline",
        XML_NAMESPACES,
    )

    # is not None 表示确实找到了 wp:inline。
    if inline_node is not None:
        return "inline"

    # 如果不是 inline，再查找它是不是浮动图片。
    anchor_node = drawing.find(
        ".//wp:anchor",
        XML_NAMESPACES,
    )

    if anchor_node is not None:
        return "anchor"

    # 两种标签都没找到时，不直接报错。
    #
    # 先返回 unknown，方便以后查看是不是遇到了
    # 当前代码还不认识的 Word 图片格式。
    return "unknown"


def _extract_image_references(
    element: ElementTree.Element,
) -> tuple[ImageReference, ...]:
    """
    从一个段落或表格中提取所有图片引用。

    element 是一个段落或表格节点。

    返回值是 ImageReference 元组。
    没有图片时返回空元组。
    """

    # 先创建空列表，用来保存找到的图片引用。
    references = []

    # 一个段落或表格中可能存在多张图片，
    # 所以使用 findall() 找到所有 w:drawing。
    drawings = element.findall(
        ".//w:drawing",
        XML_NAMESPACES,
    )

    # 每次循环处理一张图片。
    for drawing in drawings:
        # 使用前面的函数判断图片位置类型。
        position_type = _get_image_position_type(
            drawing
        )

        # 在当前图片中查找 a:blip。
        #
        # a:blip 节点上保存着 rId7 之类的图片引用编号。
        blip_nodes = drawing.findall(
            ".//a:blip",
            XML_NAMESPACES,
        )

        for blip in blip_nodes:
            # 先尝试读取内嵌图片的 r:embed。
            relationship_id = blip.attrib.get(
                RELATIONSHIP_EMBED_ATTRIBUTE,
                "",
            )

            # 如果没有 r:embed，
            # 再尝试读取外部图片的 r:link。
            if not relationship_id:
                relationship_id = blip.attrib.get(
                    RELATIONSHIP_LINK_ATTRIBUTE,
                    "",
                )

            # 两种属性都没有时，
            # 说明这个节点没有有效图片引用。
            if not relationship_id:
                continue

            # 把找到的关系编号和位置类型保存起来。
            references.append(
                ImageReference(
                    relationship_id=relationship_id,
                    position_type=position_type,
                )
            )

    # 转成元组后返回。
    #
    # 如果没有找到图片，
    # tuple(references) 就是空元组 ()。
    return tuple(references)


def read_document_blocks(
    archive: ZipFile,
) -> list[DocumentBlock]:
    """
    按照 Word 中的原始顺序读取正文内容。

    archive 是已经打开的 DOCX 压缩包。

    返回值是 DocumentBlock 列表。
    列表顺序就是段落和表格在 Word 中的顺序。
    """

    # 调用 docx_xml.py 中的公共函数，
    # 读取并解析 document.xml。
    #
    # 当前文件不需要再重复处理 XML 原始字节
    # 和 XML 格式错误。
    document_root = read_xml_root(
        archive,
        DOCUMENT_XML_PATH,
    )

    # w:body 是 Word 正文的最外层节点。
    #
    # 段落、表格和图片所在的段落，
    # 都放在 body 里面。
    body = document_root.find(
        "w:body",
        XML_NAMESPACES,
    )

    # 正常的 Word 文档应该存在 w:body。
    # 如果没有，说明文档结构可能损坏。
    if body is None:
        raise ValueError(
            "document.xml 中没有找到正文 w:body"
        )

    # 用来保存最终生成的正文块。
    blocks = []

    # block_index 记录正文总体顺序。
    #
    # 段落和表格都会增加这个编号。
    block_index = 0

    # paragraph_index 只记录段落顺序。
    #
    # 遇到表格时，这个编号不会增加。
    paragraph_index = 0

    # 只遍历 body 的直接子节点。
    #
    # 这样才能保留正文顶层的真实顺序，
    # 也不会把表格内部的段落重复拿出来。
    for element in body:
        # 只处理段落 w:p 和表格 w:tbl。
        #
        # Word 中还有 sectPr 等其他配置节点，
        # 它们不是正文内容，所以直接跳过。
        if element.tag not in {
            PARAGRAPH_TAG,
            TABLE_TAG,
        }:
            continue

        # 能运行到这里，说明当前节点是正文内容。
        block_index += 1

        # 判断当前节点是段落还是表格。
        if element.tag == PARAGRAPH_TAG:
            paragraph_index += 1
            current_paragraph_index = (
                paragraph_index
            )
            block_type = "paragraph"
        else:
            # 表格不是顶层段落，
            # 所以段落编号使用 0。
            current_paragraph_index = 0
            block_type = "table"

        # 调用前面写好的函数，
        # 提取当前块中的文字。
        text = _extract_text(element)

        # 调用前面写好的函数，
        # 提取当前块中的图片引用。
        images = _extract_image_references(
            element
        )

        # 把当前块的信息包装成 DocumentBlock。
        current_block = DocumentBlock(
            block_index=block_index,
            paragraph_index=(
                current_paragraph_index
            ),
            block_type=block_type,
            text=text,
            images=images,
        )

        # 把当前块添加到最终列表。
        blocks.append(current_block)

    # 所有正文节点处理完成后，
    # 返回完整的正文块列表。
    return blocks