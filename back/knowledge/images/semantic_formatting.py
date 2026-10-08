"""把图片语义组织成检索正文，保持既有知识图缓存格式。"""
from back.knowledge.images.semantic_models import (
    MAX_CONTEXT_CHARS, CustomerImageUnderstanding, KnowledgeImageSemanticCard, _clean_text,
)
from back.knowledge.ingestion.docx.models import ExtractedImageRecord

def format_knowledge_card(
    card: KnowledgeImageSemanticCard,
    record: ExtractedImageRecord,
    *,
    ocr_text: str = "",
) -> str:
    """生成检索与最终回答共用的图片-上下文联合正文。"""

    lines = [
        "[图片与上下文联合资料]",
        f"所属章节：{record.section_id} {record.section_title}".strip(),
        f"图片类型：{card.image_type}",
        f"图片含义：{card.summary}",
        f"在本段中的作用：{card.context_role}",
    ]
    if card.visible_evidence:
        lines.append("可识别画面：" + "；".join(card.visible_evidence))
    if card.problem_meaning:
        lines.append("该画面意味着：" + card.problem_meaning)
    if card.recommended_action:
        lines.append("文档支持的处理方式：" + card.recommended_action)
    if card.search_queries:
        lines.append("客户可能的问法：" + "；".join(card.search_queries))
    if record.previous_text.strip():
        lines.append("图片上文：\n" + _clean_text(record.previous_text, MAX_CONTEXT_CHARS))
    if record.next_text.strip():
        lines.append("图片下文：\n" + _clean_text(record.next_text, MAX_CONTEXT_CHARS))
    if ocr_text.strip():
        lines.append("图片文字（辅助证据）：\n" + _clean_text(ocr_text, MAX_CONTEXT_CHARS))
    if card.uncertainty:
        lines.append("识别限制：" + card.uncertainty)
    return "\n".join(line for line in lines if line.strip())

def format_customer_understanding(card: CustomerImageUnderstanding) -> str:
    lines = ["图片概述：" + card.summary]
    if card.visible_evidence:
        lines.append("可见证据：" + "；".join(card.visible_evidence))
    if card.likely_problem:
        lines.append("可能的问题表现：" + card.likely_problem)
    if card.search_queries:
        lines.append("建议检索词：" + "；".join(card.search_queries))
    if card.uncertainty:
        lines.append("不确定信息：" + card.uncertainty)
    return "\n".join(lines)
