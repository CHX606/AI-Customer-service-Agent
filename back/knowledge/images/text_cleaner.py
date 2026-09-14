"""图片 OCR 文本清理工具。"""


from difflib import SequenceMatcher
import re
import unicodedata
from typing import Any


# 只有较长文本才进行近似判断，避免把“账户”“设置”之类
# 含义不同但很短的界面标签误删。
MIN_FUZZY_LENGTH = 12
FUZZY_DUPLICATE_THRESHOLD = 0.94
MIN_INFORMATIVE_OCR_CHARS = 6


def _normalize_for_comparison(text: str) -> str:
    """生成只用于比较的文本，不改变最终保存的原文。"""

    normalized = unicodedata.normalize(
        "NFKC",
        text,
    ).casefold()

    # OCR 可能把同一句识别成不同的空格形式，例如：
    # “端口 10808”和“端口10808”。比较时忽略这些差别。
    normalized = re.sub(r"\s+", "", normalized)

    return normalized.strip()


def is_ocr_text_sufficient(text: str) -> bool:
    """判断整图 OCR 是否已包含可用于诊断的有效文字。"""
    normalized = unicodedata.normalize("NFKC", text)
    informative_chars = re.findall(
        r"[\w\u4e00-\u9fff]",
        normalized,
        flags=re.UNICODE,
    )
    semantic_chars = re.findall(
        r"[A-Za-z\u4e00-\u9fff]",
        normalized,
    )
    return (
        len(informative_chars) >= MIN_INFORMATIVE_OCR_CHARS
        and bool(semantic_chars)
    )


def _is_duplicate(
    normalized_text: str,
    seen_texts: list[str],
) -> bool:
    """判断一行是否与已经保留的内容重复。"""

    if not normalized_text:
        return True

    if normalized_text in seen_texts:
        return True

    if len(normalized_text) < MIN_FUZZY_LENGTH:
        return False

    for seen_text in seen_texts:
        if len(seen_text) < MIN_FUZZY_LENGTH:
            continue

        similarity = SequenceMatcher(
            None,
            normalized_text,
            seen_text,
        ).ratio()

        if similarity >= FUZZY_DUPLICATE_THRESHOLD:
            return True

    return False


def deduplicate_text_lines(texts: list[str]) -> str:
    """
    按出现顺序合并多段 OCR 文本并删除重复行。

    最终保留第一次出现的原始文字；标准化文本只用于判断，
    因此网址、订阅参数、大小写和标点不会在输出中被改写。
    """

    kept_lines = []
    seen_texts = []

    for text in texts:
        for raw_line in text.splitlines():
            line = raw_line.strip()

            if not line:
                continue

            normalized_line = _normalize_for_comparison(
                line
            )

            if _is_duplicate(
                normalized_line,
                seen_texts,
            ):
                continue

            kept_lines.append(line)
            seen_texts.append(normalized_line)

    return "\n".join(kept_lines)


def deduplicate_region_texts(
    regions: list[dict[str, Any]],
) -> str:
    """
    合并同一张参考图的区域文字。

    primary 区域包含直接用于故障判断的字段，因此优先保留；
    context 区域随后补充没有出现过的上下文，减少重叠裁剪
    造成的大段重复内容。
    """

    primary_texts = [
        str(region.get("ocr_text", ""))
        for region in regions
        if region.get("role") == "primary"
    ]
    context_texts = [
        str(region.get("ocr_text", ""))
        for region in regions
        if region.get("role") == "context"
    ]
    other_texts = [
        str(region.get("ocr_text", ""))
        for region in regions
        if region.get("role")
        not in {"primary", "context"}
    ]

    return deduplicate_text_lines(
        primary_texts
        + context_texts
        + other_texts
    )
