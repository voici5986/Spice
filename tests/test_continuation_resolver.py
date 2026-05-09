from __future__ import annotations

import unittest

from spice.llm.core import LLMRequest, LLMResponse
from spice.runtime.continuation_resolver import (
    resolve_continuation,
    resolve_continuation_with_llm,
    selected_candidate_execution_text,
    update_frame_selected_candidate,
)


class _FakeContinuationClient:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.requests: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            provider_id="openai",
            model_id="gpt-test",
            output_text=self.output_text,
            raw_payload={},
            finish_reason="stop",
            usage={},
            latency_ms=1,
            request_id="continuation-test",
        )


class ContinuationResolverTests(unittest.TestCase):
    def test_resolves_option_labels_and_choice_phrases(self) -> None:
        frame = _frame()

        direct = resolve_continuation("B", frame)
        phrase = resolve_continuation("go with B", frame)
        chinese = resolve_continuation("选第二个", frame)

        self.assertTrue(direct.is_continuation)
        self.assertEqual(direct.action, "choose_option")
        self.assertEqual(direct.candidate_id, "candidate.b")
        self.assertEqual(phrase.candidate_id, "candidate.b")
        self.assertEqual(chinese.candidate_id, "candidate.b")

    def test_resolves_execute_and_refine_followups(self) -> None:
        frame = _frame()

        execute = resolve_continuation("execute selected", frame)
        refine = resolve_continuation("refine that to be lower risk", frame)

        self.assertTrue(execute.is_continuation)
        self.assertEqual(execute.action, "execute_selected")
        self.assertEqual(execute.candidate_id, "candidate.a")
        self.assertTrue(refine.is_continuation)
        self.assertEqual(refine.action, "refine")
        self.assertEqual(refine.text, "to be lower risk")

    def test_yes_on_approval_frame_means_approve_and_execute(self) -> None:
        frame = _frame()
        frame["approval_id"] = "approval.test"

        result = resolve_continuation("y", frame)

        self.assertTrue(result.is_continuation)
        self.assertEqual(result.action, "approve_execute")

    def test_non_continuation_stays_new_intent(self) -> None:
        result = resolve_continuation("My CLI has low daily active users.", _frame())

        self.assertFalse(result.is_continuation)
        self.assertEqual(result.action, "new_intent")

    def test_llm_fallback_resolves_semantic_option_choice(self) -> None:
        client = _FakeContinuationClient(
            '{"is_continuation": true, "action": "choose_option", '
            '"candidate_label": "B", "reason": "User chose the demo option."}'
        )

        result = resolve_continuation_with_llm("就选改 demo 那个", _frame(), client=client)

        self.assertTrue(result.is_continuation)
        self.assertEqual(result.action, "choose_option")
        self.assertEqual(result.candidate_id, "candidate.b")
        self.assertEqual(result.label, "B")
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(client.requests[0].response_format_hint, "json_object")

    def test_llm_fallback_resolves_natural_execute_request(self) -> None:
        client = _FakeContinuationClient(
            '{"is_continuation": true, "action": "execute_selected", '
            '"reason": "User asked to start implementing the selected decision."}'
        )

        result = resolve_continuation_with_llm("那就去干吧", _frame(), client=client)

        self.assertTrue(result.is_continuation)
        self.assertEqual(result.action, "execute_selected")
        self.assertEqual(result.candidate_id, "candidate.a")

    def test_llm_fallback_rejects_unknown_candidate_choice(self) -> None:
        client = _FakeContinuationClient(
            '{"is_continuation": true, "action": "choose_option", '
            '"candidate_id": "candidate.missing", "reason": "Bad match."}'
        )

        result = resolve_continuation_with_llm("选那个不存在的", _frame(), client=client)

        self.assertFalse(result.is_continuation)
        self.assertEqual(result.action, "new_intent")
        self.assertIn("unknown candidate", result.reason)

    def test_updates_selected_candidate_in_frame(self) -> None:
        updated = update_frame_selected_candidate(_frame(), "candidate.b")

        self.assertEqual(updated["selected_candidate_id"], "candidate.b")
        self.assertEqual(updated["selected"]["label"], "B")
        self.assertTrue(updated["candidates"][1]["is_selected"])
        self.assertFalse(updated["candidates"][0]["is_selected"])

    def test_selected_candidate_execution_text_uses_executor_task_first(self) -> None:
        self.assertEqual(
            selected_candidate_execution_text(_frame()),
            "Create the onboarding fix plan.",
        )


def _frame() -> dict[str, object]:
    return {
        "selected_candidate_id": "candidate.a",
        "selected": {
            "label": "A",
            "candidate_id": "candidate.a",
            "title": "Fix onboarding",
            "executor_task": "Create the onboarding fix plan.",
            "recommended_action": "Prioritize onboarding.",
            "intent": "Fix first-run onboarding.",
            "is_selected": True,
        },
        "candidates": [
            {
                "label": "A",
                "candidate_id": "candidate.a",
                "title": "Fix onboarding",
                "executor_task": "Create the onboarding fix plan.",
                "is_selected": True,
            },
            {
                "label": "B",
                "candidate_id": "candidate.b",
                "title": "Improve demo",
                "executor_task": "Create the demo polish plan.",
                "is_selected": False,
            },
        ],
    }


if __name__ == "__main__":
    unittest.main()
