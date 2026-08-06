"""测试业务范围和信息充分性路由。"""

from back.agent.graph import (
    respond_clarify,
    route_after_intent,
    route_after_scope,
)


def main():
    scope_cases = {
        "in_scope": "analyze_intent",
        "chitchat": "chitchat",
        "out_of_scope": "out_of_scope",
        "uncertain": "scope_uncertain",
    }

    for scope, expected_node in scope_cases.items():
        actual_node = route_after_scope(
            {
                "scope": scope,
            }
        )

        assert actual_node == expected_node, (
            f"scope={scope}时应进入{expected_node}，"
            f"实际进入{actual_node}"
        )

    intent_cases = {
        "retrieve": "rewrite_query",
        "clarify": "clarify",
    }

    for action, expected_node in intent_cases.items():
        actual_node = route_after_intent(
            {
                "action": action,
            }
        )

        assert actual_node == expected_node, (
            f"action={action}时应进入{expected_node}，"
            f"实际进入{actual_node}"
        )

    clarification = "请问套餐是否已经显示到账？"
    response = respond_clarify(
        {
            "clarifying_question": clarification,
        }
    )

    assert response["messages"][0].content == clarification

    print("业务范围路由测试通过")
    print("意图动作路由测试通过")
    print("针对性追问返回测试通过")


if __name__ == "__main__":
    main()
