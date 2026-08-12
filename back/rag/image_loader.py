"""
Word 图片位置提取模块。

处理流程：
DOCX
→ 读取 document.xml 中的正文顺序
→ 读取 document.xml.rels 中的图片真实路径
→ 提取图片文件
→ 保存章节、block_index 和前后文
→ 返回 ExtractedImageRecord

这个模块暂时不调用 PaddleOCR-VL，也不写入 Chroma。
第一阶段只验证“图片提取”和“Word 位置”是否正确。

原来的 Docx2txtLoader 继续负责纯文本，本模块不会替换它。
"""


from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
import posixpath
from xml.etree import ElementTree
from zipfile import ZipFile

from back.rag.loader import DOCUMENT_PATH
from back.rag.splitter import match_section_heading


# 当前文件位于 back/rag。
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

# 提取后的图片保存在 data/docx_images。
# 每个文档版本会使用独立子目录，避免同名图片互相覆盖。
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "docx_images"

# 图片前后各保存 3 个非空文本块，供后续 OCR 结果理解上下文。
DEFAULT_CONTEXT_BLOCKS = 3


# DOCX 实际上是 ZIP 文件，以下是正文和关系文件的位置。
DOCUMENT_XML_PATH = "word/document.xml"
DOCUMENT_RELS_PATH = "word/_rels/document.xml.rels"


# Word XML 使用命名空间，ElementTree 查询时必须带上这些地址。
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


def _xml_name(prefix: str, local_name: str) -> str:
    """生成 ElementTree 使用的完整 XML 标签名。"""

    namespace = XML_NAMESPACES[prefix]
    return f"{{{namespace}}}{local_name}"


PARAGRAPH_TAG = _xml_name("w", "p")
TABLE_TAG = _xml_name("w", "tbl")
RELATIONSHIP_EMBED_ATTRIBUTE = _xml_name(
    "r",
    "embed",
)
RELATIONSHIP_LINK_ATTRIBUTE = _xml_name(
    "r",
    "link",
)


@dataclass(frozen=True)
class ImageReference:
    """一张图片在 document.xml 中的引用。"""

    relationship_id: str
    position_type: str


@dataclass(frozen=True)
class WordBlock:
    """
    Word 正文中的一个有序逻辑块。

    顶层段落和表格都算一个 block。
    block_index 从 1 开始，代表 Word 正文顺序。
    """

    block_index: int
    paragraph_index: int
    text: str
    section_id: str
    section_title: str
    images: tuple[ImageReference, ...]


@dataclass(frozen=True)
class RelationshipTarget:
    """relationship id 对应的实际目标。"""

    target: str
    target_mode: str


@dataclass(frozen=True)
class ExtractedImageRecord:
    """
    一张已提取图片及其 Word 位置信息。

    extracted_path 后续交给 PaddleOCR-VL。
    其他位置字段后续作为 Chroma metadata。
    """

    source: str
    document_name: str
    document_sha256: str
    extracted_path: str
    media_path: str
    image_name: str
    image_sha256: str
    relationship_id: str
    position_type: str
    image_order: int
    image_index_in_block: int
    block_index: int
    paragraph_index: int
    section_id: str
    section_title: str
    previous_text: str
    next_text: str

    def to_dict(self) -> dict:
        """转换成普通字典，便于打印、测试或保存 JSON。"""

        return asdict(self)

    def to_chroma_metadata(self) -> dict:
        """
        转换成 Chroma 可以接收的扁平 metadata。

        Chroma metadata 不能直接保存 Path、嵌套字典
        或自定义对象，所以这里统一转换成字符串和整数。
        """

        return {
            "source": self.source,
            "document_name": self.document_name,
            "document_sha256": self.document_sha256,
            "content_type": "image",
            "extracted_path": self.extracted_path,
            "media_path": self.media_path,
            "image_name": self.image_name,
            "image_sha256": self.image_sha256,
            "relationship_id": self.relationship_id,
            "position_type": self.position_type,
            "image_order": self.image_order,
            "image_index_in_block": (
                self.image_index_in_block
            ),
            "block_index": self.block_index,
            "paragraph_index": self.paragraph_index,
            "section_id": self.section_id,
            "section_title": self.section_title,
            "previous_text": self.previous_text,
            "next_text": self.next_text,
        }


def _validate_docx_path(
    document_path: str | Path,
) -> Path:
    """确认输入路径存在，并且是 DOCX 文件。"""

    resolved_path = Path(document_path).resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"没有找到 Word 文档：{resolved_path}"
        )

    if not resolved_path.is_file():
        raise ValueError(
            f"Word 文档路径不是文件：{resolved_path}"
        )

    if resolved_path.suffix.lower() != ".docx":
        raise ValueError(
            f"只支持 DOCX 文件：{resolved_path}"
        )

    return resolved_path


def _read_xml(
    archive: ZipFile,
    xml_path: str,
) -> ElementTree.Element:
    """从 DOCX 压缩包读取并解析 XML。"""

    try:
        xml_bytes = archive.read(xml_path)
    except KeyError as error:
        raise ValueError(
            f"DOCX 中缺少必要文件：{xml_path}"
        ) from error

    try:
        return ElementTree.fromstring(xml_bytes)
    except ElementTree.ParseError as error:
        raise ValueError(
            f"DOCX XML 解析失败：{xml_path}"
        ) from error


def _extract_text(element: ElementTree.Element) -> str:
    """合并一个段落或表格中的所有可见文字。"""

    text_parts = [
        node.text
        for node in element.findall(
            ".//w:t",
            XML_NAMESPACES,
        )
        if node.text
    ]

    return "".join(text_parts).strip()


def _extract_image_references(
    element: ElementTree.Element,
) -> tuple[ImageReference, ...]:
    """
    按出现顺序提取一个 block 中的 DrawingML 图片。

    当前知识库的 11 张图片全部是 inline。
    同时保留 anchor 支持，方便以后处理浮动图片。
    """

    references = []

    # drawing 节点在 XML 中的顺序就是图片出现顺序。
    for drawing in element.findall(
        ".//w:drawing",
        XML_NAMESPACES,
    ):
        if drawing.find(
            ".//wp:inline",
            XML_NAMESPACES,
        ) is not None:
            position_type = "inline"
        elif drawing.find(
            ".//wp:anchor",
            XML_NAMESPACES,
        ) is not None:
            position_type = "anchor"
        else:
            position_type = "unknown"

        for blip in drawing.findall(
            ".//a:blip",
            XML_NAMESPACES,
        ):
            # r:embed 是内嵌图片，r:link 是外部链接图片。
            relationship_id = (
                blip.attrib.get(
                    RELATIONSHIP_EMBED_ATTRIBUTE,
                    "",
                )
                or blip.attrib.get(
                    RELATIONSHIP_LINK_ATTRIBUTE,
                    "",
                )
            )

            if relationship_id:
                references.append(
                    ImageReference(
                        relationship_id=(
                            relationship_id
                        ),
                        position_type=position_type,
                    )
                )

    return tuple(references)


def _load_relationship_targets(
    relationships_root: ElementTree.Element,
) -> dict[str, RelationshipTarget]:
    """
    建立 rId 到媒体文件路径的映射。

    document.xml 中只有 rId7 之类的引用，
    document.xml.rels 才保存 media/image1.png。
    """

    targets = {}

    for relationship in relationships_root.findall(
        "pr:Relationship",
        XML_NAMESPACES,
    ):
        relationship_id = relationship.attrib.get(
            "Id",
            "",
        )
        target = relationship.attrib.get(
            "Target",
            "",
        )

        if not relationship_id or not target:
            continue

        targets[relationship_id] = RelationshipTarget(
            target=target,
            target_mode=relationship.attrib.get(
                "TargetMode",
                "Internal",
            ),
        )

    return targets


def _build_word_blocks(
    document_root: ElementTree.Element,
) -> list[WordBlock]:
    """
    把 Word 正文转换成带章节信息的有序 block。

    章节识别复用 splitter.py 的 match_section_heading，
    保证文字 Document 和图片 Document 的章节规则一致。
    """

    body = document_root.find(
        "w:body",
        XML_NAMESPACES,
    )

    if body is None:
        raise ValueError(
            "DOCX document.xml 中没有正文 body"
        )

    # sectPr 等节点不属于可检索正文，不计入 block_index。
    content_elements = [
        element
        for element in list(body)
        if element.tag in {PARAGRAPH_TAG, TABLE_TAG}
    ]

    blocks = []
    paragraph_index = 0
    current_section_id = ""
    current_section_title = ""

    for block_index, element in enumerate(
        content_elements,
        start=1,
    ):
        is_paragraph = element.tag == PARAGRAPH_TAG

        if is_paragraph:
            paragraph_index += 1
            current_paragraph_index = paragraph_index
        else:
            # 表格是一个 block，但不是顶层段落。
            current_paragraph_index = 0

        text = _extract_text(element)

        if is_paragraph and text:
            heading = match_section_heading(text)

            if heading:
                current_section_id = heading[
                    "section_id"
                ]
                current_section_title = heading[
                    "section_title"
                ]

        blocks.append(
            WordBlock(
                block_index=block_index,
                paragraph_index=(
                    current_paragraph_index
                ),
                text=text,
                section_id=current_section_id,
                section_title=current_section_title,
                images=_extract_image_references(
                    element
                ),
            )
        )

    return blocks


def _find_context_text(
    blocks: list[WordBlock],
    current_index: int,
    direction: int,
    context_block_count: int,
) -> str:
    """查找图片之前或之后最近的若干个非空文本块。"""

    if direction not in {-1, 1}:
        raise ValueError(
            "direction 只能是 -1 或 1"
        )

    collected = []
    index = current_index + direction

    while (
        0 <= index < len(blocks)
        and len(collected) < context_block_count
    ):
        text = blocks[index].text.strip()

        if text:
            collected.append(text)

        index += direction

    # 向前查找时是从近到远收集，反转后恢复阅读顺序。
    if direction == -1:
        collected.reverse()

    return "\n".join(collected)


def _resolve_media_path(target: str) -> str:
    """
    把 rels 中的相对路径转换成 DOCX 内部路径。

    media/image1.png 会变成 word/media/image1.png。
    """

    if target.startswith("/"):
        return target.lstrip("/")

    normalized_path = posixpath.normpath(
        posixpath.join("word", target)
    )

    return PurePosixPath(normalized_path).as_posix()


def _calculate_file_sha256(file_path: Path) -> str:
    """分块计算文件哈希，避免一次读取整个 DOCX。"""

    digest = sha256()

    with file_path.open("rb") as file:
        while data := file.read(1024 * 1024):
            digest.update(data)

    return digest.hexdigest()


def extract_docx_images(
    document_path: str | Path = DOCUMENT_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    context_block_count: int = DEFAULT_CONTEXT_BLOCKS,
) -> list[ExtractedImageRecord]:
    """
    提取 DOCX 图片，并返回按 Word 顺序排列的位置记录。

    context_block_count 控制图片前后各保存多少个非空文本块。
    """

    if context_block_count < 0:
        raise ValueError(
            "context_block_count 不能小于 0"
        )

    document_path = _validate_docx_path(
        document_path
    )
    output_root = Path(output_root).resolve()
    document_sha256 = _calculate_file_sha256(
        document_path
    )

    # 文档内容变化时哈希也会变化，因此新旧版本不会混用。
    output_directory = (
        output_root
        / f"{document_path.stem}_{document_sha256[:12]}"
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []
    image_order = 0

    with ZipFile(document_path) as archive:
        document_root = _read_xml(
            archive,
            DOCUMENT_XML_PATH,
        )
        relationships_root = _read_xml(
            archive,
            DOCUMENT_RELS_PATH,
        )

        blocks = _build_word_blocks(document_root)
        relationship_targets = (
            _load_relationship_targets(
                relationships_root
            )
        )

        for current_index, block in enumerate(blocks):
            if not block.images:
                continue

            previous_text = _find_context_text(
                blocks,
                current_index,
                direction=-1,
                context_block_count=context_block_count,
            )
            next_text = _find_context_text(
                blocks,
                current_index,
                direction=1,
                context_block_count=context_block_count,
            )

            for image_index, reference in enumerate(
                block.images,
                start=1,
            ):
                image_order += 1
                target = relationship_targets.get(
                    reference.relationship_id
                )

                if target is None:
                    raise ValueError(
                        "图片引用没有对应关系："
                        f"{reference.relationship_id}"
                    )

                if target.target_mode.lower() == "external":
                    raise ValueError(
                        "暂不支持 DOCX 外部图片："
                        f"{target.target}"
                    )

                media_path = _resolve_media_path(
                    target.target
                )

                try:
                    image_bytes = archive.read(media_path)
                except KeyError as error:
                    raise ValueError(
                        "DOCX 中没有找到图片："
                        f"{media_path}"
                    ) from error

                image_sha256 = sha256(
                    image_bytes
                ).hexdigest()
                suffix = (
                    PurePosixPath(media_path).suffix.lower()
                    or ".bin"
                )
                image_name = (
                    f"image_{image_order:03d}{suffix}"
                )
                extracted_path = (
                    output_directory / image_name
                )

                # 重复运行时覆盖同一编号文件，不产生重复副本。
                extracted_path.write_bytes(image_bytes)

                records.append(
                    ExtractedImageRecord(
                        source=str(document_path),
                        document_name=document_path.name,
                        document_sha256=(
                            document_sha256
                        ),
                        extracted_path=str(
                            extracted_path
                        ),
                        media_path=media_path,
                        image_name=image_name,
                        image_sha256=image_sha256,
                        relationship_id=(
                            reference.relationship_id
                        ),
                        position_type=(
                            reference.position_type
                        ),
                        image_order=image_order,
                        image_index_in_block=image_index,
                        block_index=block.block_index,
                        paragraph_index=(
                            block.paragraph_index
                        ),
                        section_id=block.section_id,
                        section_title=block.section_title,
                        previous_text=previous_text,
                        next_text=next_text,
                    )
                )

    return records


def main():
    """
    第一阶段的命令行检查入口。

    只提取图片并打印位置，不调用 OCR，也不写 Chroma。
    """

    records = extract_docx_images()

    print(
        f"Word 图片提取完成，共 {len(records)} 张"
    )

    for record in records:
        print("\n" + "=" * 70)
        print(f"图片顺序：{record.image_order}")
        print(f"图片文件：{record.image_name}")
        print(f"图片类型：{record.position_type}")
        print(f"Word block：{record.block_index}")
        print(
            f"Word paragraph：{record.paragraph_index}"
        )
        print(
            "所属章节："
            f"{record.section_id} "
            f"{record.section_title}"
        )
        print(f"DOCX 媒体路径：{record.media_path}")
        print(f"提取路径：{record.extracted_path}")
        print("\n前文：")
        print(record.previous_text or "[无]")
        print("\n后文：")
        print(record.next_text or "[无]")


if __name__ == "__main__":
    main()
