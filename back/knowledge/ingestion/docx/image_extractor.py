"""
DOCX 图片提取模块。

这个文件负责：
1. 组合正文图片顺序和 rId 关系表。
2. 找到图片在 DOCX 中的真实路径。
3. 读取图片原始字节。
4. 把图片保存到项目目录。
5. 返回图片的来源和位置信息。

这个文件不会：
- 解析 document.xml 的内部细节。
- 解析 document.xml.rels 的内部细节。
- 修改图片内容。
- 调用 OCR。
"""


# sha256 用来计算 Word 和图片的内容指纹。
from hashlib import sha256

# posixpath 专门处理使用 / 的路径。
#
# DOCX 内部路径固定使用 /，
# 即使当前电脑是 Windows，也不能使用 Windows 路径规则。
import posixpath

# Path 用来处理电脑上的路径。
# PurePosixPath 用来处理 DOCX 压缩包内部路径。
from pathlib import Path, PurePosixPath

# ZipFile 用来打开并读取 DOCX 压缩包。
from zipfile import ZipFile

from back.core.paths import PROJECT_ROOT

# 读取正文顺序和图片 rId。
from back.knowledge.ingestion.docx.document_reader import (
    read_document_blocks,
)

# 提取完成后需要返回的图片记录。
from back.knowledge.ingestion.docx.models import (
    ExtractedImageRecord,
)

# 读取 rId 对应的图片目标路径。
from back.knowledge.ingestion.docx.relationship_reader import (
    read_relationship_targets,
)

from back.knowledge.ingestion.docx.block_context import (
    build_block_contexts,
)


# 提取出来的图片默认保存在这里。
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "docx_images"
)


def _resolve_media_path(
    target: str,
) -> str:
    """
    把关系文件中的目标路径转换成 DOCX 内部路径。

    document.xml.rels 中通常保存：
    media/image7.png

    DOCX 压缩包中的真实位置是：
    word/media/image7.png
    """

    # 如果路径以 / 开头，
    # 表示它已经从 DOCX 根目录开始。
    if target.startswith("/"):
        # ZipFile 内部路径不需要最前面的 /。
        return target.lstrip("/")

    # 普通目标路径是相对于 word/ 目录的，
    # 所以前面需要加上 word。
    media_path = posixpath.join(
        "word",
        target,
    )

    # 整理路径中的 ./ 和 ../。
    normalized_path = posixpath.normpath(
        media_path
    )

    # 确保最终使用 DOCX 内部要求的 / 路径格式。
    return PurePosixPath(
        normalized_path
    ).as_posix()


def _validate_docx_path(
    document_path: str | Path,
) -> Path:
    """
    检查传入的路径是不是可以使用的 DOCX 文件。

    校验成功后返回整理好的绝对路径。
    """

    # 把字符串或相对路径转换成绝对 Path。
    resolved_path = Path(
        document_path
    ).resolve()

    # 路径不存在时直接报错。
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"没有找到 Word 文档：{resolved_path}"
        )

    # 路径必须指向文件，不能是文件夹。
    if not resolved_path.is_file():
        raise ValueError(
            f"Word 文档路径不是文件：{resolved_path}"
        )

    # 当前流程只支持 DOCX 文件。
    if resolved_path.suffix.lower() != ".docx":
        raise ValueError(
            f"只支持 DOCX 文件：{resolved_path}"
        )

    return resolved_path


def _calculate_file_sha256(
    file_path: Path,
) -> str:
    """
    计算一个文件的 SHA256 内容指纹。

    这里分块读取文件，避免一次把整个大文件
    全部加载到内存中。
    """

    # 创建一个空的 SHA256 计算器。
    digest = sha256()

    # 使用二进制方式打开文件。
    with file_path.open("rb") as file:
        while True:
            # 每次读取 1MB。
            data = file.read(
                1024 * 1024
            )

            # 没有读取到内容，
            # 说明文件已经读取完成。
            if not data:
                break

            # 把当前数据加入 SHA256 计算。
            digest.update(data)

    # 返回十六进制格式的 SHA256。
    return digest.hexdigest()


def extract_docx_images(
    document_path: str | Path,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> list[ExtractedImageRecord]:
    """
    从 DOCX 中提取图片，并返回图片记录列表。

    document_path：
    需要处理的 Word 文档路径。

    output_root：
    提取后的图片保存在哪个根目录。

    返回值：
    按照图片在 Word 中的出现顺序排列的
    ExtractedImageRecord 列表。
    """

    # 检查 Word 路径是否合法。
    document_path = _validate_docx_path(
        document_path
    )

    # 把输出根目录转换成绝对路径。
    output_root = Path(
        output_root
    ).resolve()

    # 计算当前 Word 的内容指纹。
    document_sha256 = _calculate_file_sha256(
        document_path
    )

    # 为当前 Word 版本创建独立目录。
    #
    # 例如：
    # data/docx_images/操作文档_a1b2c3d4e5f6/
    output_directory = (
        output_root
        / (
            f"{document_path.stem}_"
            f"{document_sha256[:12]}"
        )
    )

    # 创建输出目录。
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 保存最终生成的所有图片记录。
    records: list[ExtractedImageRecord] = []

    # 记录图片在整份 Word 中的顺序。
    image_order = 0

    # 打开 DOCX 压缩包。
    with ZipFile(document_path) as archive:
        # 从 document.xml 中读取正文顺序和图片 rId。
        blocks = read_document_blocks(
            archive
        )

        # 为所有正文块计算章节和前后文。
        block_contexts = build_block_contexts(
            blocks
        )

        # 从 document.xml.rels 中读取 rId 对应关系。
        relationships = (
            read_relationship_targets(
                archive
            )
        )

        # 按照 Word 正文顺序处理每个块。
        for block in blocks:
            # 一个正文块中可能有多张图片。
            for image_index, image_reference in enumerate(
                block.images,
                start=1,
            ):
                # 整份 Word 的图片顺序增加 1。
                image_order += 1

                # 从 document.xml 中取得图片 rId。
                relationship_id = (
                    image_reference.relationship_id
                )

                # 根据图片所在 block 找到上下文。
                block_context = block_contexts.get(
                    block.block_index
                )

                if block_context is None:
                    raise ValueError(
                        "图片所在正文块没有上下文信息："
                        f"block_index={block.block_index}"
                    )

                # 使用 rId 寻找目标路径。
                target = relationships.get(
                    relationship_id
                )

                # document.xml 中有 rId，
                # 但 .rels 中找不到对应关系时明确报错。
                if target is None:
                    raise ValueError(
                        "图片引用没有对应关系："
                        f"{relationship_id}"
                    )

                # 当前项目不自动下载外部图片。
                if (
                    target.target_mode.lower()
                    == "external"
                ):
                    raise ValueError(
                        "暂不支持 DOCX 外部图片："
                        f"{target.target}"
                    )

                # 转换成 DOCX 内部的完整图片路径。
                media_path = _resolve_media_path(
                    target.target
                )

                try:
                    # 从 DOCX 中读取图片原始字节。
                    image_bytes = archive.read(
                        media_path
                    )
                except KeyError as error:
                    raise ValueError(
                        "DOCX 中没有找到图片："
                        f"{media_path}"
                    ) from error

                # 计算图片内容指纹。
                image_sha256 = sha256(
                    image_bytes
                ).hexdigest()

                # 取得原图片扩展名，例如 .png。
                suffix = PurePosixPath(
                    media_path
                ).suffix.lower()

                # 没有扩展名时使用 .bin。
                if not suffix:
                    suffix = ".bin"

                # 按 Word 图片顺序生成统一文件名。
                image_name = (
                    f"image_{image_order:03d}"
                    f"{suffix}"
                )

                # 生成图片最终保存路径。
                extracted_path = (
                    output_directory
                    / image_name
                )

                # 把图片原始字节保存到电脑。
                extracted_path.write_bytes(
                    image_bytes
                )

                # 创建当前图片的完整记录。
                record = ExtractedImageRecord(
                    source=str(document_path),
                    document_name=(
                        document_path.name
                    ),
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
                        relationship_id
                    ),
                    position_type=(
                        image_reference.position_type
                    ),
                    image_order=image_order,
                    image_index_in_block=(
                        image_index
                    ),
                    block_index=(
                        block.block_index
                    ),
                    paragraph_index=(
                        block.paragraph_index
                    ),
                    section_id=(
                        block_context.section_id
                    ),
                    section_title=(
                        block_context.section_title
                    ),
                    previous_text=(
                        block_context.previous_text
                    ),
                    next_text=(
                        block_context.next_text
                    ),
                )

                # 把当前记录加入最终列表。
                records.append(record)

    # 返回全部图片记录。
    return records
