"""
这是优化后的结构化切块文件

识别章节标题
→ 按章节切分
→ 保存section_id和section_title
→ 对4.5.2章节敏感字段脱敏
→ 超过600字符的章节再切块，重叠100字符
→ 短章节保持完整
"""

import re

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from back.rag.loader import load_documents


CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


SECTION_HEADING_PATTERN = re.compile(
    r"^(?P<section_id>\d+(?:\.\d+){0,2})"
    r"\s*"
    r"(?P<section_title>.+?)"
    r"\s*$"
)


UNNUMBERED_SECTION_IDS = {
    "账号相关": "2",
    "官网相关": "3",
    "实际使用": "4",
    "安卓端": "4.4",
    "iOS/iPad": "4.5",
    "Windows": "4.6",
    "MacOS": "4.7",
    "卡/慢": "5",
    "解锁与IP质量": "6",
    "可乐云资料": "7",
    "对客服人员的基本要求": "8",
}


SENSITIVE_SECTION_IDS = {
    "4.5.2",
}


SENSITIVE_FIELD_PATTERN = re.compile(
    r"^\s*"
    r"(?P<field>访问网址|账号|密码)"
    r"\s*[：:]"
    r".*$"
)


def match_section_heading(line: str):
    """判断一行文本是否为章节标题。"""

    cleaned_line = line.strip()

    # 处理Word中丢失自动编号的标题。
    if cleaned_line in UNNUMBERED_SECTION_IDS:
        return {
            "section_id": (
                UNNUMBERED_SECTION_IDS[
                    cleaned_line
                ]
            ),
            "section_title": cleaned_line,
        }

    match = SECTION_HEADING_PATTERN.match(
        cleaned_line
    )

    if not match:
        return None

    section_id = match.group("section_id")
    section_title = (
        match.group("section_title").strip()
    )

    # 避免将较长的普通正文识别成标题。
    if len(section_title) > 40:
        return None

    # 标题必须以中文或英文字母开头。
    if not re.match(
        r"[\u4e00-\u9fffA-Za-z]",
        section_title,
    ):
        return None

    # 排除目录中末尾带页码的标题。
    if re.search(
        r"\s+\d+$",
        section_title,
    ):
        return None

    return {
        "section_id": section_id,
        "section_title": section_title,
    }


def split_document_by_sections(
    document: Document,
):
    """按照章节标题将一份长文档拆成多个章节。"""

    sections = []

    current_section_id = None
    current_section_title = None
    current_content = []

    lines = document.page_content.splitlines()

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            continue

        heading = match_section_heading(line)

        if heading:
            # 遇到新标题时，先保存前一个章节。
            if (
                current_section_id is not None
                and current_content
            ):
                section_text = "\n".join(
                    current_content
                )

                sections.append(
                    Document(
                        page_content=section_text,
                        metadata={
                            **document.metadata,
                            "section_id": (
                                current_section_id
                            ),
                            "section_title": (
                                current_section_title
                            ),
                        },
                    )
                )

            current_section_id = (
                heading["section_id"]
            )

            current_section_title = (
                heading["section_title"]
            )

            current_content = []

            continue

        # 目录和封面等标题前内容暂时不加入知识块。
        if current_section_id is not None:
            current_content.append(line)

    # 保存最后一个章节。
    if (
        current_section_id is not None
        and current_content
    ):
        section_text = "\n".join(
            current_content
        )

        sections.append(
            Document(
                page_content=section_text,
                metadata={
                    **document.metadata,
                    "section_id": (
                        current_section_id
                    ),
                    "section_title": (
                        current_section_title
                    ),
                },
            )
        )

    return sections


def redact_sensitive_content(
    text: str,
) -> str:
    """替换文本中的网址、账号和密码。"""

    safe_lines = []

    for line in text.splitlines():
        match = SENSITIVE_FIELD_PATTERN.match(
            line
        )

        if match:
            field_name = match.group("field")

            safe_lines.append(
                f"{field_name}：[敏感信息已隐藏]"
            )

            continue

        safe_lines.append(line)

    return "\n".join(safe_lines)


def sanitize_section(
    section: Document,
) -> Document:
    """对指定知识库章节进行脱敏。"""

    section_id = section.metadata.get(
        "section_id"
    )

    # 非敏感章节保持原样。
    if section_id not in SENSITIVE_SECTION_IDS:
        return section

    safe_content = redact_sensitive_content(
        section.page_content
    )

    return Document(
        page_content=safe_content,
        metadata={
            **section.metadata,
            "redacted": True,
        },
    )


def split_long_section(
    section: Document,
):
    """对过长章节进行二次切块。"""

    section_id = section.metadata[
        "section_id"
    ]

    section_title = section.metadata[
        "section_title"
    ]

    section_heading = (
        f"{section_id} {section_title}"
    )

    # 短章节不再切割，但把标题放回正文。
    if len(section.page_content) <= CHUNK_SIZE:
        return [
            Document(
                page_content=(
                    f"{section_heading}\n"
                    f"{section.page_content}"
                ),
                metadata={
                    **section.metadata,
                    "chunk_index": 0,
                },
            )
        ]

    text_splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
    )

    child_chunks = text_splitter.split_text(
        section.page_content
    )

    results = []

    for index, child_text in enumerate(
        child_chunks
    ):
        results.append(
            Document(
                page_content=(
                    f"{section_heading}\n"
                    f"{child_text}"
                ),
                metadata={
                    **section.metadata,
                    "chunk_index": index,
                },
            )
        )

    return results


def split_documents(documents):
    """按章节切分、清理敏感信息并二次切块。"""

    chunks = []

    for document in documents:
        sections = split_document_by_sections(
            document
        )

        for section in sections:
            # 生成最终chunk前进行脱敏。
            safe_section = sanitize_section(
                section
            )

            section_chunks = split_long_section(
                safe_section
            )

            chunks.extend(section_chunks)

    return chunks


if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)

    print(
        "原始 Document 数量：",
        len(documents),
    )

    print(
        "结构化 chunk 数量：",
        len(chunks),
    )

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        print(
            f"\n----- 第 {index} 个文本块 -----"
        )

        print(
            "字符数量：",
            len(chunk.page_content),
        )

        print(
            "metadata：",
            chunk.metadata,
        )

        print(
            chunk.page_content[:500]
        )