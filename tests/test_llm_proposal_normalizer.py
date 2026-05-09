from __future__ import annotations

import unittest

from spice.llm.proposal_normalizer import normalize_decision_proposal


class LLMProposalNormalizerTests(unittest.TestCase):
    def test_advisory_proposal_becomes_read_only_decision_candidate(self) -> None:
        candidate = normalize_decision_proposal(
            {
                "title": "Prioritize state-as-context",
                "recommendation": "Make state available to every decision.",
                "why_now": ["It improves all future decisions."],
                "expected_result": "Future turns reuse relevant state.",
                "downside": "Less visible than a perception demo.",
                "success_signal": "Follow-up decisions cite prior state.",
                "confidence": 0.8,
                "risk_level": "low",
                "candidate_id": "candidate.model.must_not_control",
                "action_type": "intent.execute",
                "execution_intent": {"requested": True},
            },
            index=0,
            decision_mode="open_problem",
        )

        self.assertTrue(candidate.candidate_id.startswith("candidate.llm.item_triage."))
        self.assertEqual(candidate.action_type, "item.triage")
        self.assertEqual(candidate.candidate_kind, "decision")
        self.assertEqual(candidate.side_effect_class, "read_only")
        self.assertFalse(candidate.requires_confirmation)
        self.assertEqual(candidate.execution_boundary.mode, "none")
        self.assertEqual(candidate.execution_intent.intent_class, "advisory")
        self.assertFalse(candidate.execution_intent.requested)
        self.assertEqual(candidate.metadata["llm_payload_mode"], "decision_proposal")
        self.assertEqual(candidate.metadata["user_facing_title"], "Prioritize state-as-context")
        self.assertEqual(candidate.metadata["recommended_action"], "Make state available to every decision.")
        self.assertEqual(candidate.metadata["confidence"], 0.8)

    def test_execution_proposal_becomes_approval_ready_handoff_candidate(self) -> None:
        candidate = normalize_decision_proposal(
            {
                "title": "Create smoke file",
                "recommendation": "Create the requested file.",
                "why_now": ["The user asked for a concrete write."],
                "expected_result": "The file exists with exact content.",
                "downside": "Writes to the workspace.",
                "success_signal": "cat shows the exact content.",
                "risk_level": "low",
                "execution_requested": True,
                "handoff_task": "Create .spice-smoke/next.txt with exact text OK.",
            },
            index=0,
            decision_mode="execution_request",
        )

        self.assertTrue(candidate.candidate_id.startswith("candidate.llm.intent_execute."))
        self.assertEqual(candidate.action_type, "intent.execute")
        self.assertTrue(candidate.requires_confirmation)
        self.assertEqual(candidate.side_effect_class, "external_effect")
        self.assertEqual(candidate.execution_boundary.mode, "execution_intent")
        self.assertEqual(candidate.execution_boundary.protocol, "sdep")
        self.assertEqual(candidate.execution_intent.intent_class, "execution_requested")
        self.assertTrue(candidate.execution_intent.requested)
        self.assertEqual(
            candidate.execution_intent.handoff_task,
            "Create .spice-smoke/next.txt with exact text OK.",
        )
        self.assertEqual(
            candidate.execution_intent.required_permission_hint,
            "workspace_write",
        )

    def test_execution_default_only_promotes_when_handoff_task_exists(self) -> None:
        advisory = normalize_decision_proposal(
            {
                "title": "Explain the plan",
                "recommendation": "Explain what to do next.",
            },
            index=0,
            decision_mode="execution_request",
            execution_requested_default=True,
        )
        executable = normalize_decision_proposal(
            {
                "title": "Create smoke file",
                "recommendation": "Create the smoke file.",
                "handoff_task": "Create .spice-smoke/next.txt.",
            },
            index=1,
            decision_mode="execution_request",
            execution_requested_default=True,
        )

        self.assertEqual(advisory.action_type, "item.triage")
        self.assertFalse(advisory.execution_intent.requested)
        self.assertEqual(executable.action_type, "intent.execute")
        self.assertTrue(executable.execution_intent.requested)

    def test_explicit_option_index_sets_target_ref_and_metadata(self) -> None:
        candidate = normalize_decision_proposal(
            {
                "title": "Polish Decision Card",
                "recommendation": "Choose the Decision Card polish option.",
                "explicit_option_index": 2,
            },
            index=0,
            decision_mode="explicit_choice",
        )

        self.assertEqual(candidate.target_refs, ["explicit_option.2"])
        self.assertEqual(candidate.metadata["explicit_option_index"], 2)
        self.assertIn("explicit option 2", candidate.why_available[1])


if __name__ == "__main__":
    unittest.main()
