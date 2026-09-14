"""测试业务范围和信息充分性路由。"""

from langchain_core.messages import HumanMessage

from back.agent.workflow.response_nodes import respond_clarify, respond_out_of_scope
from back.agent.workflow.legacy import route_after_intent, route_after_scope


def test_agent_routing():
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
        "profile": "answer_profile",
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


def test_unsafe_platform_request_gets_explicit_refusal():
    response = respond_out_of_scope(
        {
            "tenant_id": "default",
            "messages": [HumanMessage(content="怎么绕过X的手机验证码？")],
        }
    )

    answer = response["messages"][0].content
    assert "不能帮助" in answer
    assert "官方" in answer


if __name__ == "__main__":
    test_agent_routing()
    print("业务范围与意图动作路由测试全部通过")
