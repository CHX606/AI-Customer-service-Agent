"""
DOCX 正文块上下文处理模块。

负责：
1. 判断每个正文块属于哪个章节。
2. 找到每个正文块前后的非空文字。

不读取 XML、不提取图片，也不调用 OCR。
"""


from back.knowledge.ingestion.docx.models import (
    DocumentBlock,
    DocumentBlockContext,
)
from back.knowledge.ingestion.splitter import match_section_heading


# 图片前后默认各保留 3 个非空正文块。
DEFAULT_CONTEXT_BLOCKS = 3


def _find_context_text(
    blocks: list[DocumentBlock],
    current_index: int,
    direction: int,
    context_block_count: int,
) -> str:
    """查找当前正文块之前或之后的非空文字。"""

    if direction not in {-1, 1}:
        raise ValueError(
            "direction 只能是 -1 或 1"
        )

    collected: list[str] = []
    index = current_index + direction

    while (
        0 <= index < len(blocks)
        and len(collected) < context_block_count
    ):
        text = blocks[index].text.strip()

        if text:
            collected.append(text)

        index += direction

    # 向前查找时是从近到远收集，
    # 需要反转回来恢复正常阅读顺序。
    if direction == -1:
        collected.reverse()

    return "\n".join(collected)


def build_block_contexts(
    blocks: list[DocumentBlock],
    context_block_count: int = DEFAULT_CONTEXT_BLOCKS,
) -> dict[int, DocumentBlockContext]:
    """
    为所有正文块建立章节和前后文信息。

    返回值以 block_index 为键，
    方便图片记录查询所在正文块的上下文。
    """

    if context_block_count < 0:
        raise ValueError(
            "context_block_count 不能小于 0"
        )

    contexts: dict[
        int,
        DocumentBlockContext,
    ] = {}

    current_section_id = ""
    current_section_title = ""

    for current_index, block in enumerate(
        blocks
    ):
        # 只有普通段落参与章节标题判断。
        if (
            block.block_type == "paragraph"
            and block.text
        ):
            heading = match_section_heading(
                block.text
            )

            if heading:
                current_section_id = heading[
                    "section_id"
                ]
                current_section_title = heading[
                    "section_title"
                ]

        contexts[block.block_index] = (
            DocumentBlockContext(
                block_index=block.block_index,
                section_id=current_section_id,
                section_title=(
                    current_section_title
                ),
                previous_text=_find_context_text(
                    blocks,
                    current_index,
                    direction=-1,
                    context_block_count=(
                        context_block_count
                    ),
                ),
                next_text=_find_context_text(
                    blocks,
                    current_index,
                    direction=1,
                    context_block_count=(
                        context_block_count
                    ),
                ),
            )
        )

    return contexts