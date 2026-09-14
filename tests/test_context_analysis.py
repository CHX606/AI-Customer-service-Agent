"""上下文生命周期规则的离线验证。"""
import json
from unittest.mock import Mock
import pytest
from back.agent.analysis.context import ContextAnalysis, analyze_context

@pytest.mark.parametrize("issue, explicit, expected", [(None, False, "new_issue"),
    ({"summary": "旧问题", "status": "awaiting_user"}, False, "continue"),
    ({"summary": "旧问题", "status": "answered"}, True, "new_issue")])
def test_context_boundary_uses_lifecycle(issue, explicit, expected):
    output=ContextAnalysis(relation="continue", resolved_query="旧问题加当前问题", decision_reason="模型判断",
        is_self_contained=explicit, references_active_issue=False, answers_last_question=not explicit,
        explicit_new_issue=explicit)
    llm=Mock()
    llm.with_structured_output.return_value.invoke.return_value=output
    result=analyze_context(llm, "当前问题", [], issue)
    assert result.relation == expected
    if expected == "new_issue":
        assert result.resolved_query == "当前问题"
    payload=json.loads(llm.with_structured_output.return_value.invoke.call_args.args[0][1].content)
    assert payload["current_query"] == "当前问题"
    assert payload["active_issue"] == issue
