"""保存读取 document.xml 时使用的数据结构。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageReference:
    """表示正文中发现的一张图片引用。"""

    # 保存 <a:blip> 标签中的 r:embed，
    # 例如 rId7。
    relationship_id: str

    # 保存图片的位置类型：
    # inline 表示图片跟随文字排列；
    # anchor 表示图片浮动在页面上。
    position_type: str


@dataclass(frozen=True)
class DocumentBlock:
    """表示 document.xml 正文中的一个内容块。"""

    # 这个块在整个 Word 正文中的顺序。
    block_index: int

    # 如果是段落，保存它是第几个段落；
    # 如果是表格，可以保存为 0。
    paragraph_index: int

    # 内容类型，目前使用 paragraph 或 table。
    block_type: str

    # 这个段落或表格中的文字。
    text: str

    # 这个内容块中出现的所有图片引用。
    # 没有图片时，这里就是空元组 ()。
    images: tuple[ImageReference, ...]

@dataclass(frozen=True)
class DocumentBlockContext:
    """一个正文块对应的章节和前后文信息。"""

    block_index: int
    section_id: str
    section_title: str
    previous_text: str
    next_text: str


@dataclass(frozen = True)
class RelationshipTarget:
    """
    document.xml.rels 中的一条关系记录。

    target 保存目标路径，例如 media/image7.png。

    target_mode 表示目标在 DOCX 内部还是外部链接。
    常见值是 Internal 或 External。
    """

    target:str
    target_mode:str

@dataclass(frozen=True)
class ExtractedImageRecord:
    """
    一张已经从 DOCX 中提取出来的图片记录。

    这个对象不保存图片本身，
    只保存图片相关的位置、路径和身份信息。
    """

    # 原始 Word 文档的完整路径。
    source: str

    # 原始 Word 文档的文件名。
    document_name: str

    # Word 文档内容的 SHA256。
    #
    # 文档发生变化后，这个值也会变化，
    # 可以用来区分不同版本的 Word。
    document_sha256:str

    # 图片提取到电脑后的完整路径。
    extracted_path: str

    # 图片在 DOCX 压缩包内部的路径。
    #
    # 例如：
    # word/media/image7.png
    media_path: str

    # 提取后使用的图片文件名。
    #
    # 例如：
    # image_007.png
    image_name: str

    # 图片内容的 SHA256。
    #
    # 可以用来判断两张图片内容是否完全相同。
    image_sha256: str

    # document.xml 中保存的图片引用编号。
    #
    # 例如：
    # rId7
    relationship_id: str

    # 图片在 Word 中是 inline、anchor 还是 unknown。
    position_type: str

    # 图片在整份 Word 中的出现顺序。
    #
    # 第一张是 1，第二张是 2。
    image_order: int

    # 图片在当前正文块中的顺序。
    #
    # 一个段落可能有多张图片，
    # 所以需要记录它是当前块中的第几张。
    image_index_in_block: int

    # 图片所在正文块的顺序编号。
    block_index: int

    # 图片所在段落的顺序编号。
    #
    # 如果图片位于表格中，可以使用 0。
    paragraph_index: int

    # 图片所在章节的信息。
    section_id: str
    section_title: str

    # 图片前后最近的正文内容。
    previous_text: str
    next_text: str