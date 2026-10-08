"""给精排命中的文本块补齐同源、同版本、同章节的相邻上下文。"""

from typing import Any

from langchain_core.documents import Document

from back.infrastructure.search.opensearch import get_vector_store


WINDOW = 1
MAX_CHARS = 1400
MAX_OVERLAP = 120
MIN_OVERLAP = 8

ChunkKey = tuple[str, str, str, int]


def _chunk_key(metadata: dict[str, Any], tenant_id: str) -> ChunkKey | None:
    """图片、跨租户或缺少可用定位元数据的块保持原样。"""
    if str(metadata.get("tenant_id", tenant_id)) != tenant_id:
        return None
    if str(metadata.get("content_type") or "").lower().startswith("image"):
        return None
    source_id = str(metadata.get("source_id") or "").strip()
    section_id = str(metadata.get("section_id") or "").strip()
    if not source_id or not section_id:
        return None
    try:
        chunk_index = int(str(metadata.get("chunk_index", "")))
    except (TypeError, ValueError):
        return None
    if chunk_index < 0:
        return None
    return source_id, section_id, str(metadata.get("content_hash") or ""), chunk_index


def _join_chunks(chunks: list[str], metadata: dict[str, Any]) -> str:
    heading = " ".join(
        str(metadata.get(field) or "").strip()
        for field in ("section_id", "section_title")
    ).strip()
    # 通用分块没有标题，不能把正文的第一行当作标题删掉。
    has_heading = bool(heading and chunks[0].partition("\n")[0] == heading)
    bodies = [
        chunk.partition("\n")[2]
        if has_heading and chunk.partition("\n")[0] == heading
        else chunk
        for chunk in chunks
    ]
    text = bodies[0]
    for body in bodies[1:]:
        overlap = 0
        for size in range(min(len(text), len(body), MAX_OVERLAP), MIN_OVERLAP - 1, -1):
            if text[-size:] == body[:size]:
                overlap = size
                break
        text += body[overlap:] if overlap else f"\n{body}"
    return f"{heading}\n{text}" if has_heading else text


def expand_neighbor_chunks(documents: list[Document], tenant_id: str) -> list[Document]:
    """最多补齐前后各一块，保持精排的数量、顺序和命中块元数据。"""
    keys = [_chunk_key(document.metadata, tenant_id) for document in documents]
    targets = [key for key in keys if key is not None]
    if not targets:
        return documents

    wanted = {
        (*key[:3], index)
        for key in targets
        for index in range(max(0, key[3] - WINDOW), key[3] + WINDOW + 1)
    }
    result = get_vector_store(tenant_id).get(where={
        "source_id": {"$in": sorted({key[0] for key in targets})},
        "section_id": {"$in": sorted({key[1] for key in targets})},
        "content_hash": {"$in": sorted({key[2] for key in targets})},
        # 只查询需要的窗口，避免整章读取触及 get 的 10,000 条上限。
        "chunk_index": {"$in": sorted({str(key[3]) for key in wanted})},
    })
    chunks: dict[ChunkKey, str] = {}
    for text, metadata in zip(result.get("documents", []), result.get("metadatas", [])):
        key = _chunk_key(metadata, tenant_id)
        if key in wanted and isinstance(text, str) and text.strip():
            chunks[key] = text
    # 检索和扩展之间可能发生重索；命中正文始终以精排时的内容为准。
    for document, key in zip(documents, keys):
        if key is not None:
            chunks[key] = document.page_content

    expanded = []
    used: set[ChunkKey] = set()
    for document, key in zip(documents, keys):
        if key is None:
            expanded.append(document)
            continue
        span = [
            (*key[:3], index)
            for index in range(max(0, key[3] - WINDOW), key[3] + WINDOW + 1)
            if (*key[:3], index) in chunks
        ]
        if len(span) <= 1 or any(item in used for item in span):
            expanded.append(document)
            continue
        text = _join_chunks([chunks[item] for item in span], document.metadata)
        if len(text) > MAX_CHARS:
            expanded.append(document)
            continue
        used.update(span)
        expanded.append(Document(
            page_content=text,
            metadata={**document.metadata, "expanded_chunks": [item[3] for item in span]},
        ))
    return expanded
