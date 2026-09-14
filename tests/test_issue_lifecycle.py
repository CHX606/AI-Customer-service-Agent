"""测试客服问题生命周期和上下文边界规则。"""

from back.agent.workflow.lifecycle import (
    decide_relation_with_lifecycle,
    set_issue_intent,
    start_or_continue_issue,
    transition_issue,
    user_confirms_resolution,
)


def main():
    active_issue = start_or_continue_issue(
        active_issue=None,
        relation="new_issue",
        summary="续费后流量没有恢复",
    )

    assert active_issue is not None
    assert active_issue["status"] == "open"

    active_issue = set_issue_intent(
        active_issue,
        "traffic",
    )
    active_issue = transition_issue(
        active_issue,
        "answered",
    )

    assert active_issue is not None
    assert active_issue["intent"] == "traffic"
    assert active_issue["status"] == "answered"

    # answered 状态下，用户追问同义/更具体问题（model_relation="continue"），保持 continue
    continue_relation_after_answered = decide_relation_with_lifecycle(
        model_relation="continue",
        active_issue=active_issue,
        is_self_contained=True,
        references_active_issue=False,
        answers_last_question=False,
        explicit_new_issue=False,
    )

    assert continue_relation_after_answered == "continue"

    # answered 状态下，真正的新问题（model_relation="new_issue"），进入 new_issue
    new_relation_after_answered = decide_relation_with_lifecycle(
        model_relation="new_issue",
        active_issue=active_issue,
        is_self_contained=True,
        references_active_issue=False,
        answers_last_question=False,
        explicit_new_issue=False,
    )

    assert new_relation_after_answered == "new_issue"

    # resolved 状态下，独立新问题切换为 new_issue
    resolved_issue = transition_issue(active_issue, "resolved")
    new_relation_after_resolved = decide_relation_with_lifecycle(
        model_relation="continue",
        active_issue=resolved_issue,
        is_self_contained=True,
        references_active_issue=False,
        answers_last_question=False,
        explicit_new_issue=False,
    )
    assert new_relation_after_resolved == "new_issue"

    active_issue = start_or_continue_issue(
        active_issue=active_issue,
        relation="new_issue",
        summary="我购买了套餐，但是不能用",
    )
    active_issue = transition_issue(
        active_issue,
        "awaiting_user",
        last_clarifying_question="套餐已经到账了吗？",
    )

    assert active_issue is not None
    assert active_issue["status"] == "awaiting_user"

    continue_relation = decide_relation_with_lifecycle(
        model_relation="new_issue",
        active_issue=active_issue,
        is_self_contained=False,
        references_active_issue=False,
        answers_last_question=True,
        explicit_new_issue=False,
    )

    assert continue_relation == "continue"

    reopened_relation = decide_relation_with_lifecycle(
        model_relation="new_issue",
        active_issue=transition_issue(active_issue, "answered"),
        is_self_contained=False,
        references_active_issue=True,
        answers_last_question=False,
        explicit_new_issue=False,
    )

    assert reopened_relation == "continue"
    assert user_confirms_resolution("已经解决了，谢谢")
    assert not user_confirms_resolution("谢谢，我先试一下")

    print("问题生命周期创建与状态转移测试通过")
    print("已回答问题与新问题的边界规则测试通过")
    print("等待补充信息后的继续对话规则测试通过")
    print("用户确认解决识别测试通过")


def test_issue_lifecycle():
    main()


if __name__ == "__main__":
    main()
