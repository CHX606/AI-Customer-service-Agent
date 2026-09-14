"""客服问题生命周期与上下文边界的确定性规则。"""

import re

from back.domain.conversation import ActiveIssue, IntentType, IssueStatus, RelationType


CLOSED_ISSUE_STATUSES = {
    "resolved",
    "handed_off",
}

RESOLUTION_PHRASES = (
    "问题解决了",
    "已经解决了",
    "已经恢复了",
    "恢复正常了",
    "现在可以用了",
    "现在能用了",
    "现在好了",
    "已经好了",
)

_RESOLUTION_CONFIRMATION = re.compile(
    r"(?:好的|好|嗯|谢谢你|谢谢|感谢|太好了|我这边|我的|这个|问题)*"
    rf"(?:{'|'.join(re.escape(phrase) for phrase in RESOLUTION_PHRASES)})"
    r"(?:啦|谢谢你的帮助|谢谢你|谢谢|非常感谢|感谢|辛苦了|太好了|好的)*"
)


def decide_relation_with_lifecycle(
    *,
    model_relation: RelationType,
    active_issue: ActiveIssue | None,
    is_self_contained: bool,
    references_active_issue: bool,
    answers_last_question: bool,
    explicit_new_issue: bool,
) -> RelationType:
    """结合模型语义信号和问题状态，确定最终上下文关系。"""

    if active_issue is None:
        return "new_issue"

    if explicit_new_issue:
        return "new_issue"

    if references_active_issue:
        if model_relation == "correction":
            return "correction"

        return "continue"

    status = active_issue.get("status", "open")

    if status == "awaiting_user" and answers_last_question:
        if model_relation == "correction":
            return "correction"

        return "continue"

    if status in CLOSED_ISSUE_STATUSES and is_self_contained:
        return "new_issue"

    return model_relation


def start_or_continue_issue(
    active_issue: ActiveIssue | None,
    relation: RelationType | None,
    summary: str,
) -> ActiveIssue | None:
    """根据上下文关系创建新问题，或继续更新当前问题。"""

    if relation not in {"continue", "new_issue", "correction"}:
        return active_issue

    if relation == "new_issue" or active_issue is None:
        return {
            "summary": summary,
            "status": "open",
            "intent": None,
            "last_clarifying_question": None,
        }

    updated_issue: ActiveIssue = {
        **active_issue,
        "summary": summary,
        "status": "open",
        "last_clarifying_question": None,
    }

    return updated_issue


def set_issue_intent(
    active_issue: ActiveIssue | None,
    intent: IntentType,
) -> ActiveIssue | None:
    """记录当前问题的业务意图。"""

    if active_issue is None:
        return None

    return {
        **active_issue,
        "intent": intent,
    }


def transition_issue(
    active_issue: ActiveIssue | None,
    status: IssueStatus,
    *,
    last_clarifying_question: str | None = None,
) -> ActiveIssue | None:
    """将当前问题转换到指定生命周期状态。"""

    if active_issue is None:
        return None

    return {
        **active_issue,
        "status": status,
        "last_clarifying_question": last_clarifying_question,
    }


def user_confirms_resolution(message: str) -> bool:
    """只接受完整肯定句；否定、疑问或附带新诉求时继续正常分析。"""
    # 保留问号；“已经解决了？”不能被当成用户确认。
    normalized_message = re.sub(r"[\s，。！、,.!；;：:]+", "", message)
    return _RESOLUTION_CONFIRMATION.fullmatch(normalized_message) is not None
