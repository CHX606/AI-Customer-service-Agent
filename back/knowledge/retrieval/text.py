"""OpenSearch 中文关键词字段的统一预分词。"""

import jieba


def tokenize_chinese(text: str) -> list[str]:
    """使用 jieba 搜索模式生成稳定的 BM25 词项。"""
    words = jieba.lcut_for_search((text or "").lower())
    return [word.strip() for word in words if word.strip()]


def build_keyword_terms(text: str) -> str:
    """生成供 OpenSearch whitespace analyzer 索引和查询的文本。"""
    return " ".join(tokenize_chinese(text))
