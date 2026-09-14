"""知识库文档切块模块。

支持：
1. 具备章节编号的文档：按章节提取标题并精准切块与脱敏。
2. 无章节编号/自由排版文档：自适应回退为基于字符重叠的通用分块。
3. 统一注入租户、源文件与切块编号元数据。
"""

import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from back.knowledge.ingestion.loader import load_documents


# CrossEncoder 的 512 token 上限还要容纳问题、章节标题和特殊 token。
# 中文字符通常接近一个 token，因此索引块需要明显小于 512 字符。
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

FALLBACK_CHUNK_SIZE = 800
FALLBACK_CHUNK_OVERLAP = 120


SECTION_HEADING_PATTERN = re.compile(
    r"^(?P<section_id>\d+(?:\.\d+){0,2})"
    r"\s*"
    r"(?P<section_title>.+?)"
    r"\s*$"
)


UNNUMBERED_SECTION_IDS = {
    "订单相关": "1",
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

    if cleaned_line in UNNUMBERED_SECTION_IDS:
        return {
            "section_id": UNNUMBERED_SECTION_IDS[cleaned_line],
            "section_title": cleaned_line,
        }

    match = SECTION_HEADING_PATTERN.match(cleaned_line)
    if not match:
        return None

    section_id = match.group("section_id")
    section_title = match.group("section_title").strip()

    if len(section_title) > 40:
        return None

    if not re.match(r"[\u4e00-\u9fffA-Za-z]", section_title):
        return None

    if re.search(r"\s+\d+$", section_title):
        return None

    return {
        "section_id": section_id,
        "section_title": section_title,
    }


def split_document_by_sections(document: Document) -> list[Document]:
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
            if current_section_id is not None and current_content:
                section_text = "\n".join(current_content)
                sections.append(
                    Document(
                        page_content=section_text,
                        metadata={
                            **document.metadata,
                            "section_id": current_section_id,
                            "section_title": current_section_title,
                        },
                    )
                )

            current_section_id = heading["section_id"]
            current_section_title = heading["section_title"]
            current_content = []
            continue

        if current_section_id is not None:
            current_content.append(line)

    if current_section_id is not None and current_content:
        section_text = "\n".join(current_content)
        sections.append(
            Document(
                page_content=section_text,
                metadata={
                    **document.metadata,
                    "section_id": current_section_id,
                    "section_title": current_section_title,
                },
            )
        )

    return sections


def redact_sensitive_content(text: str) -> str:
    """替换文本中的网址、账号和密码。"""
    safe_lines = []
    for line in text.splitlines():
        match = SENSITIVE_FIELD_PATTERN.match(line)
        if match:
            field_name = match.group("field")
            safe_lines.append(f"{field_name}：[敏感信息已隐藏]")
            continue
        safe_lines.append(line)
    return "\n".join(safe_lines)


def sanitize_section(section: Document) -> Document:
    """对指定知识库章节进行脱敏。"""
    section_id = section.metadata.get("section_id")
    if section_id not in SENSITIVE_SECTION_IDS:
        return section

    safe_content = redact_sensitive_content(section.page_content)
    return Document(
        page_content=safe_content,
        metadata={
            **section.metadata,
            "redacted": True,
        },
    )


def split_long_section(section: Document) -> list[Document]:
    """对过长章节进行二次切块。"""
    section_id = section.metadata.get("section_id", "未知")
    section_title = section.metadata.get("section_title", "正文")
    section_heading = f"{section_id} {section_title}".strip()

    if len(section.page_content) <= CHUNK_SIZE:
        content = (
            f"{section_heading}\n{section.page_content}"
            if section_heading
            else section.page_content
        )
        return [
            Document(
                page_content=content,
                metadata={
                    **section.metadata,
                    "chunk_index": 0,
                },
            )
        ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    child_chunks = text_splitter.split_text(section.page_content)

    results = []
    for index, child_text in enumerate(child_chunks):
        content = (
            f"{section_heading}\n{child_text}"
            if section_heading
            else child_text
        )
        results.append(
            Document(
                page_content=content,
                metadata={
                    **section.metadata,
                    "chunk_index": index,
                },
            )
        )
    return results


def split_prebuilt_documents(
    documents: list[Document],
    starting_chunk_index: int = 0,
) -> list[Document]:
    """切分图片语义等已带元数据的文档，并重新分配连续块编号。"""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    results = []
    for document in documents:
        parts = (
            text_splitter.split_text(document.page_content)
            if len(document.page_content) > CHUNK_SIZE
            else [document.page_content]
        )
        section_id = str(document.metadata.get("section_id", "")).strip()
        section_title = str(
            document.metadata.get("section_title", "")
        ).strip()
        heading = f"{section_id} {section_title}".strip()
        parent_chunk_index = document.metadata.get("chunk_index")

        for part in parts:
            content = part
            if heading and not part.lstrip().startswith(heading):
                content = f"{heading}\n{part}"
            metadata = {
                **document.metadata,
                "chunk_index": starting_chunk_index + len(results),
            }
            if len(parts) > 1:
                metadata["parent_chunk_index"] = parent_chunk_index
            results.append(Document(page_content=content, metadata=metadata))
    return results


def split_documents(documents: list[Document]) -> list[Document]:
    """按章节切分；无章节时自适应回退为 RecursiveCharacterTextSplitter。"""
    chunks = []
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=FALLBACK_CHUNK_SIZE,
        chunk_overlap=FALLBACK_CHUNK_OVERLAP,
    )

    for document in documents:
        sections = split_document_by_sections(document)

        if sections:
            # 成功按章节切分
            for section in sections:
                safe_section = sanitize_section(section)
                section_chunks = split_long_section(safe_section)
                chunks.extend(section_chunks)
        else:
            # 回退机制：无法识别章节结构时使用通用切块
            raw_text = document.page_content.strip()
            if not raw_text:
                continue

            raw_chunks = fallback_splitter.split_text(raw_text)
            for index, chunk_text in enumerate(raw_chunks):
                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata={
                            **document.metadata,
                            "section_id": "通用",
                            "section_title": document.metadata.get(
                                "filename", "通用文档"
                            ),
                            "chunk_index": index,
                        },
                    )
                )

    # 重新规范化同批次的 chunk_index
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index

    return chunks


if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)
    print("原始 Document 数量：", len(documents))
    print("结构化 chunk 数量：", len(chunks))
