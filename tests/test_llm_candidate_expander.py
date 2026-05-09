from __future__ import annotations

import json
import unittest

from spice.decision.general.candidates import GenericCandidate
from spice.decision.general.state import GeneralDecisionState
from spice.decision.general.types import Intent
from spice.llm.candidate_expander import (
    _build_expansion_prompt,
    _parse_expansion_payload,
    build_explicit_option_candidates,
    expand_candidates_from_runtime_config,
    expand_candidates_with_llm,
    extract_explicit_options,
    merge_expanded_candidates,
)
from spice.llm.core import LLMClient, LLMModelConfig, LLMRouter, LLMTaskHook, ProviderRegistry
from spice.llm.decision_proposal import (
    LLMDecisionProposal,
    LLM_DECISION_PROPOSAL_FIELDS,
    RUNTIME_CANDIDATE_FIELDS,
)
from spice.llm.providers import DeterministicLLMProvider


class LLMCandidateExpanderTests(unittest.TestCase):
    def test_lightweight_decision_proposal_schema_excludes_runtime_fields(self) -> None:
        payload = LLMDecisionProposal.response_schema()["decisions"][0]

        self.assertEqual(set(payload), set(LLM_DECISION_PROPOSAL_FIELDS))
        self.assertFalse(set(payload) & RUNTIME_CANDIDATE_FIELDS)

    def test_lightweight_decision_proposal_normalizes_only_semantic_fields(self) -> None:
        proposal = LLMDecisionProposal.from_payload(
            {
                "title": "Prioritize state-as-context",
                "recommendation": "Make state a first-class decision context input.",
                "why_now": ["It improves every future decision.", ""],
                "expected_result": "Follow-up decisions reuse prior state.",
                "downside": "Less visible than a perception demo.",
                "success_signal": "A follow-up turn can use prior decision state.",
                "confidence": 1.4,
                "risk_level": "LOW",
                "explicit_option_index": "2",
                "execution_requested": "false",
                "handoff_task": "ignored unless execution is requested",
                "candidate_id": "candidate.model.should_not_control",
                "action_type": "context.prepare",
                "execution_intent": {"intent_class": "execution_requested"},
                "side_effect_class": "external_effect",
                "requires_confirmation": True,
                "estimated_cost": {"time_minutes": 99},
            }
        )

        payload = proposal.to_payload()

        self.assertEqual(set(payload), set(LLM_DECISION_PROPOSAL_FIELDS))
        self.assertEqual(payload["title"], "Prioritize state-as-context")
        self.assertEqual(payload["why_now"], ["It improves every future decision."])
        self.assertEqual(payload["confidence"], 1.0)
        self.assertEqual(payload["risk_level"], "low")
        self.assertEqual(payload["explicit_option_index"], 2)
        self.assertFalse(payload["execution_requested"])
        self.assertFalse(set(payload) & RUNTIME_CANDIDATE_FIELDS)

    def test_expander_validates_and_builds_generic_candidates(self) -> None:
        response = {
            "candidates": [
                {
                    "action_type": "task.split",
                    "intent": "Split the failing-test work into diagnosis and fix steps.",
                    "user_facing_title": "Break the failing test fix into diagnosis and patch steps",
                    "recommended_action": "First identify the failing assertion, then make the smallest patch.",
                    "why_now": ["The failure blocks CI."],
                    "expected_result": "A short execution plan for the failing test.",
                    "executor_task": "Inspect the failing test and propose the smallest safe patch.",
                    "target_refs": ["intent.main"],
                    "side_effect_class": "state_change",
                    "requires_confirmation": False,
                    "estimated_cost": {"time_minutes": 10, "attention": "medium"},
                    "risk_profile": {"level": "low", "summary": "Low risk planning step."},
                    "expected_state_delta": {"summary": "Creates a smaller plan."},
                    "why_available": ["The intent is broad enough to split."],
                    "execution_intent": {
                        "intent_class": "advisory",
                        "requested": False,
                        "reason": "Planning candidate only.",
                        "side_effect_class": "state_change",
                    },
                },
                {
                    "action_type": "not.allowed",
                    "intent": "Invalid option.",
                },
            ]
        }
        result = expand_candidates_with_llm(
            client=_client(response),
            state=_state(),
            intent_text="Fix the failing test.",
            rule_candidates=[_rule_candidate()],
            model_provider="deterministic",
            model_id="deterministic.v1",
        )

        self.assertEqual(result.status, "expanded")
        self.assertEqual(result.proposed_count, 2)
        self.assertEqual(result.accepted_count, 1)
        self.assertEqual(result.rejected_count, 1)
        candidate = result.candidates[0]
        self.assertTrue(candidate.candidate_id.startswith("candidate.llm.task_split."))
        self.assertEqual(candidate.action_type, "task.split")
        self.assertEqual(candidate.side_effect_class, "state_change")
        self.assertTrue(candidate.requires_confirmation)
        self.assertEqual(candidate.execution_boundary.side_effect_class, "state_change")
        self.assertTrue(candidate.metadata["llm_generated"])
        self.assertEqual(candidate.candidate_kind, "decision")
        self.assertEqual(candidate.execution_intent.intent_class, "advisory")
        self.assertFalse(candidate.execution_intent.requested)
        self.assertEqual(candidate.execution_intent.reason, "Planning candidate only.")
        self.assertEqual(candidate.metadata["candidate_kind"], "decision")
        self.assertEqual(candidate.metadata["candidate_source"], "llm_generator")
        self.assertEqual(candidate.metadata["execution_intent"]["intent_class"], "advisory")
        self.assertEqual(candidate.metadata["execution_intent"]["requested"], False)
        self.assertEqual(
            candidate.metadata["execution_intent"]["reason"],
            "Planning candidate only.",
        )
        self.assertEqual(candidate.metadata["decision_mode"], "execution_request")
        self.assertEqual(
            candidate.metadata["user_facing_title"],
            "Break the failing test fix into diagnosis and patch steps",
        )
        self.assertEqual(
            candidate.metadata["recommended_action"],
            "First identify the failing assertion, then make the smallest patch.",
        )
        self.assertEqual(candidate.metadata["why_now"], ["The failure blocks CI."])
        self.assertEqual(
            candidate.metadata["executor_task"],
            "Inspect the failing test and propose the smallest safe patch.",
        )

    def test_expander_accepts_lightweight_decision_proposals(self) -> None:
        result = expand_candidates_with_llm(
            client=_client(
                {
                    "decisions": [
                        {
                            "title": "Prioritize state-as-context",
                            "recommendation": "Make state a first-class decision context input.",
                            "why_now": ["It improves every future decision."],
                            "expected_result": "Follow-up turns reuse prior decision state.",
                            "downside": "Less visually obvious than a perception demo.",
                            "success_signal": "A follow-up decision references prior state.",
                            "confidence": 0.82,
                            "risk_level": "low",
                            "candidate_id": "candidate.model.must_not_control",
                            "action_type": "intent.execute",
                            "execution_intent": {"intent_class": "execution_requested"},
                        }
                    ]
                }
            ),
            state=_state(),
            intent_text="What should we prioritize next?",
            rule_candidates=[_rule_candidate()],
        )

        self.assertEqual(result.status, "expanded")
        self.assertEqual(result.proposed_count, 1)
        self.assertEqual(result.accepted_count, 1)
        candidate = result.candidates[0]
        self.assertTrue(candidate.candidate_id.startswith("candidate.llm.item_triage."))
        self.assertEqual(candidate.action_type, "item.triage")
        self.assertEqual(candidate.candidate_kind, "decision")
        self.assertEqual(candidate.side_effect_class, "read_only")
        self.assertFalse(candidate.requires_confirmation)
        self.assertEqual(candidate.execution_intent.intent_class, "advisory")
        self.assertFalse(candidate.execution_intent.requested)
        self.assertEqual(candidate.metadata["llm_payload_mode"], "decision_proposal")
        self.assertEqual(candidate.metadata["user_facing_title"], "Prioritize state-as-context")
        self.assertEqual(candidate.metadata["confidence"], 0.82)
        self.assertFalse(set(candidate.metadata) & {"candidate_id", "action_type"})

    def test_minimax_style_options_payload_normalizes_to_lightweight_candidates(self) -> None:
        result = expand_candidates_with_llm(
            client=_client(
                {
                    "options": [
                        {
                            "title": "Prioritize state-as-context",
                            "next_step": "Make state a first-class context input.",
                            "why": "It improves every future decision.",
                            "expected_outcome": "Follow-up turns reuse prior state.",
                            "risk": "low risk",
                        },
                        {
                            "name": "Deepen executor handoff",
                            "recommendation": "Improve the approval-to-execution path.",
                            "risks": ["medium integration risk"],
                        },
                    ]
                }
            ),
            state=_state(),
            intent_text="Which path should Spice prioritize next?",
            rule_candidates=[_rule_candidate()],
        )

        self.assertEqual(result.status, "expanded")
        self.assertEqual(result.proposed_count, 2)
        self.assertEqual(result.accepted_count, 2)
        first = result.candidates[0]
        second = result.candidates[1]
        self.assertEqual(first.metadata["llm_payload_mode"], "decision_proposal")
        self.assertEqual(first.metadata["user_facing_title"], "Prioritize state-as-context")
        self.assertEqual(
            first.metadata["recommended_action"],
            "Make state a first-class context input.",
        )
        self.assertEqual(first.metadata["why_now"], ["It improves every future decision."])
        self.assertEqual(first.metadata["expected_result"], "Follow-up turns reuse prior state.")
        self.assertEqual(first.risk_profile.level, "low")
        self.assertEqual(second.metadata["user_facing_title"], "Deepen executor handoff")
        self.assertEqual(second.risk_profile.level, "medium")

    def test_english_and_chinese_intents_accept_lightweight_proposals(self) -> None:
        cases = [
            (
                "What should Spice prioritize next?",
                "en",
                "Prioritize perception",
                "Improve proactive perception first.",
            ),
            (
                "Spice 下一步应该优先做什么？",
                "zh",
                "优先改进感知",
                "先改进主动感知。",
            ),
        ]

        for intent_text, display_language, title, recommendation in cases:
            with self.subTest(display_language=display_language):
                result = expand_candidates_with_llm(
                    client=_client(
                        {
                            "decisions": [
                                {
                                    "title": title,
                                    "recommendation": recommendation,
                                    "why_now": ["This is the highest leverage path."],
                                    "expected_result": "Better decisions.",
                                    "risk_level": "low",
                                }
                            ]
                        }
                    ),
                    state=_state(),
                    intent_text=intent_text,
                    rule_candidates=[_rule_candidate()],
                    display_language=display_language,
                )

                self.assertEqual(result.status, "expanded")
                self.assertEqual(result.accepted_count, 1)
                self.assertEqual(result.candidates[0].metadata["user_facing_title"], title)
                self.assertEqual(result.candidates[0].metadata["recommended_action"], recommendation)

    def test_execution_request_proposal_becomes_approval_ready_candidate(self) -> None:
        result = expand_candidates_with_llm(
            client=_client(
                {
                    "decisions": [
                        {
                            "title": "Create the smoke note",
                            "recommendation": "Create the requested smoke note file.",
                            "why_now": ["The user requested a concrete file write."],
                            "expected_result": "The smoke file exists with the requested text.",
                            "downside": "It writes to the workspace.",
                            "success_signal": "cat shows the exact requested content.",
                            "confidence": 0.9,
                            "risk_level": "low",
                            "execution_requested": True,
                            "handoff_task": "Create .spice-smoke/next.txt with exact text OK.",
                        }
                    ]
                }
            ),
            state=_state(),
            intent_text="Create .spice-smoke/next.txt with exact text OK.",
            rule_candidates=[_rule_candidate()],
        )

        candidate = result.candidates[0]

        self.assertEqual(candidate.action_type, "intent.execute")
        self.assertTrue(candidate.requires_confirmation)
        self.assertEqual(candidate.execution_boundary.mode, "execution_intent")
        self.assertEqual(candidate.execution_boundary.protocol, "sdep")
        self.assertEqual(candidate.execution_intent.intent_class, "execution_requested")
        self.assertTrue(candidate.execution_intent.requested)
        self.assertEqual(
            candidate.execution_intent.handoff_task,
            "Create .spice-smoke/next.txt with exact text OK.",
        )
        self.assertEqual(
            candidate.metadata["execution_intent"]["required_permission_hint"],
            "workspace_write",
        )

    def test_parser_repairs_markdown_fenced_json_and_alias_fields(self) -> None:
        payload = _parse_expansion_payload(
            """
            Here is the decision JSON:
            ```json
            {
              "recommendations": [
                {
                  "title": "Improve perception",
                  "next_step": "Prioritize OpenChronicle perception",
                  "why": "It differentiates Spice from reactive agents.",
                  "expected_outcome": "Spice can trigger decisions proactively.",
                  "risk": "medium risk"
                }
              ]
            }
            ```
            """
        )

        decision = payload["decisions"][0]

        self.assertEqual(decision["title"], "Improve perception")
        self.assertEqual(decision["recommendation"], "Prioritize OpenChronicle perception")
        self.assertEqual(
            decision["why_now"],
            ["It differentiates Spice from reactive agents."],
        )
        self.assertEqual(
            decision["expected_result"],
            "Spice can trigger decisions proactively.",
        )
        self.assertEqual(decision["risk_level"], "medium")

    def test_parser_repairs_top_level_list_and_string_items(self) -> None:
        payload = _parse_expansion_payload(
            """
            [
              {"option": "State as context", "why": ["Improves every decision"]},
              "Improve executor handoff"
            ]
            """
        )

        self.assertEqual(len(payload["decisions"]), 2)
        self.assertEqual(payload["decisions"][0]["title"], "State as context")
        self.assertEqual(payload["decisions"][0]["recommendation"], "State as context")
        self.assertEqual(payload["decisions"][1]["title"], "Improve executor handoff")
        self.assertEqual(
            payload["decisions"][1]["recommendation"],
            "Improve executor handoff",
        )

    def test_parser_supports_options_and_items_keys(self) -> None:
        options_payload = _parse_expansion_payload(
            '{"options": [{"title": "Add retry", "next_step": "Add LLM retry"}]}'
        )
        items_payload = _parse_expansion_payload(
            '{"items": [{"name": "Polish card", "expected_outcome": "Cleaner demo"}]}'
        )

        self.assertEqual(options_payload["decisions"][0]["recommendation"], "Add LLM retry")
        self.assertEqual(items_payload["decisions"][0]["title"], "Polish card")
        self.assertEqual(items_payload["decisions"][0]["expected_result"], "Cleaner demo")

    def test_parser_preserves_legacy_candidates_payload(self) -> None:
        payload = _parse_expansion_payload(
            json.dumps(
                {
                    "candidates": [
                        {
                            "action_type": "item.triage",
                            "intent": "Triage the work.",
                        }
                    ]
                }
            )
        )

        self.assertIn("candidates", payload)
        self.assertNotIn("decisions", payload)
        self.assertEqual(payload["candidates"][0]["action_type"], "item.triage")

    def test_unrepairable_output_records_raw_output_in_result(self) -> None:
        raw_output = "I cannot produce JSON today."
        result = expand_candidates_with_llm(
            client=_client_text(raw_output),
            state=_state(),
            intent_text="What should we prioritize next?",
            rule_candidates=[_rule_candidate()],
        )

        self.assertEqual(result.status, "fallback")
        self.assertIn("valid JSON", result.error)
        self.assertEqual(result.raw_output, raw_output)
        self.assertEqual(result.to_payload()["raw_output"], raw_output)

    def test_missing_decisions_list_falls_back_with_raw_output(self) -> None:
        raw_output = '{"notes": ["no candidates here"]}'
        result = expand_candidates_with_llm(
            client=_client_text(raw_output),
            state=_state(),
            intent_text="What should we prioritize next?",
            rule_candidates=[_rule_candidate()],
        )

        self.assertEqual(result.status, "fallback")
        self.assertEqual(result.error, "missing decisions list")
        self.assertEqual(result.raw_output, raw_output)

    def test_expander_treats_llm_triage_as_approval_gated_state_change(self) -> None:
        result = expand_candidates_with_llm(
            client=_client(
                {
                    "candidates": [
                        {
                            "action_type": "item.triage",
                            "intent": "Inspect the merge conflict and identify the safest resolution.",
                            "user_facing_title": "Resolve the merge conflict first",
                            "recommended_action": "Ask the executor to inspect the conflicted files and propose the minimal safe resolution.",
                        }
                    ]
                }
            ),
            state=_state(),
            intent_text="I hit a merge conflict.",
            rule_candidates=[_rule_candidate()],
        )

        candidate = result.candidates[0]

        self.assertEqual(candidate.action_type, "item.triage")
        self.assertEqual(candidate.side_effect_class, "state_change")
        self.assertEqual(candidate.execution_boundary.side_effect_class, "state_change")
        self.assertTrue(candidate.requires_confirmation)
        self.assertEqual(
            candidate.metadata["user_facing_title"],
            "Resolve the merge conflict first",
        )

    def test_merge_expanded_candidates_keeps_llm_decision_candidates_first(self) -> None:
        rule_candidate = _rule_candidate()
        result = expand_candidates_with_llm(
            client=_client(
                {
                    "candidates": [
                        {
                            "action_type": "user.clarify",
                            "intent": "Ask the user which test is failing.",
                            "side_effect_class": "read_only",
                            "requires_confirmation": False,
                        }
                    ]
                }
            ),
            state=_state(),
            intent_text="Fix the failing test.",
            rule_candidates=[rule_candidate],
        )

        merged = merge_expanded_candidates([rule_candidate], result)

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].metadata["source"], "llm_candidate_expander")
        self.assertEqual(merged[0].metadata["candidate_kind"], "decision")
        self.assertEqual(merged[1].candidate_id, rule_candidate.candidate_id)

    def test_runtime_config_disabled_returns_disabled_result(self) -> None:
        result = expand_candidates_from_runtime_config(
            config={"llm_candidate_expand": "false", "llm_provider": "deterministic"},
            state=_state(),
            intent_text="Fix the failing test.",
            rule_candidates=[_rule_candidate()],
        )

        self.assertFalse(result.enabled)
        self.assertEqual(result.status, "disabled")
        self.assertEqual(result.candidates, [])

    def test_runtime_config_missing_model_falls_back_without_raising(self) -> None:
        result = expand_candidates_from_runtime_config(
            config={"llm_candidate_expand": "true", "llm_provider": "openai"},
            state=_state(),
            intent_text="Fix the failing test.",
            rule_candidates=[_rule_candidate()],
        )

        self.assertTrue(result.enabled)
        self.assertEqual(result.status, "fallback")
        self.assertIn("llm_model", result.error)

    def test_extract_explicit_options_from_choice_intent(self) -> None:
        options = extract_explicit_options(
            "Compare these 3 next steps for Spice: add LLM retry, polish Decision Card, "
            "add JSON output. Which should we do first?"
        )

        self.assertEqual(
            options,
            ["add LLM retry", "polish Decision Card", "add JSON output"],
        )

    def test_extract_explicit_options_from_decide_list_without_pick_keyword(self) -> None:
        options = extract_explicit_options(
            "I need to decide: add LLM retry, polish Decision Card, add JSON output"
        )

        self.assertEqual(
            options,
            ["add LLM retry", "polish Decision Card", "add JSON output"],
        )

    def test_extract_explicit_options_ignores_empty_and_non_choice_text(self) -> None:
        self.assertEqual(extract_explicit_options(""), [])
        self.assertEqual(extract_explicit_options("just a normal sentence"), [])
        self.assertEqual(extract_explicit_options("choose add LLM retry"), [])
        self.assertEqual(
            extract_explicit_options(
                "I have a failing test, a pending PR review, and a meeting in 45 minutes"
            ),
            [],
        )

    def test_explicit_option_candidates_use_user_options_directly(self) -> None:
        candidates = build_explicit_option_candidates(
            intent_text="I need to decide: add LLM retry, polish Decision Card, add JSON output",
            state=_state(),
        )

        self.assertEqual(len(candidates), 3)
        self.assertEqual(
            [candidate.metadata["user_facing_title"] for candidate in candidates],
            ["add LLM retry", "polish Decision Card", "add JSON output"],
        )
        self.assertTrue(all(candidate.action_type == "item.triage" for candidate in candidates))
        self.assertTrue(all(candidate.candidate_kind == "decision" for candidate in candidates))
        self.assertTrue(all(candidate.execution_intent.intent_class == "advisory" for candidate in candidates))
        self.assertTrue(all(not candidate.execution_intent.requested for candidate in candidates))
        self.assertTrue(
            all(candidate.metadata["execution_intent"]["requested"] is False for candidate in candidates)
        )
        self.assertTrue(all(not candidate.requires_confirmation for candidate in candidates))
        self.assertTrue(all(candidate.metadata["candidate_kind"] == "decision" for candidate in candidates))
        self.assertTrue(
            all(candidate.metadata["candidate_source"] == "explicit_options" for candidate in candidates)
        )

    def test_explicit_choice_mode_rejects_meta_actions(self) -> None:
        result = expand_candidates_with_llm(
            client=_client(
                {
                    "candidates": [
                        {
                            "action_type": "user.clarify",
                            "explicit_option_index": 1,
                            "intent": "Ask the user to clarify release priorities.",
                            "user_facing_title": "Clarify release priorities",
                            "recommended_action": "Ask the user what matters most.",
                        },
                        {
                            "action_type": "item.triage",
                            "explicit_option_index": 2,
                            "intent": "Compare whether polish Decision Card should be first.",
                            "user_facing_title": "polish Decision Card",
                            "recommended_action": "Choose polish Decision Card as the first release task.",
                            "why_now": ["It improves demo clarity."],
                        },
                    ]
                }
            ),
            state=_state(),
            intent_text=(
                "Compare these 3 next steps for Spice: add LLM retry, polish Decision Card, "
                "add JSON output. Which should we do first?"
            ),
            rule_candidates=[_rule_candidate()],
        )

        self.assertEqual(result.status, "expanded")
        self.assertEqual(result.accepted_count, 1)
        self.assertEqual(result.rejected_count, 1)
        self.assertEqual(result.candidates[0].metadata["user_facing_title"], "polish Decision Card")
        self.assertIn("meta action_type", result.rejected[0]["reason"])

    def test_explicit_choice_mode_accepts_paraphrase_with_option_index(self) -> None:
        result = expand_candidates_with_llm(
            client=_client(
                {
                    "candidates": [
                        {
                            "action_type": "item.triage",
                            "explicit_option_index": 1,
                            "intent": "Implement retry logic for LLM API calls.",
                            "user_facing_title": "Add retry handling for LLM calls",
                            "recommended_action": "Prioritize retry handling for model calls.",
                            "why_now": ["It improves runtime reliability."],
                        },
                    ]
                }
            ),
            state=_state(),
            intent_text=(
                "Compare these 3 next steps for Spice: add LLM retry, polish Decision Card, "
                "add JSON output. Which should we do first?"
            ),
            rule_candidates=[_rule_candidate()],
        )

        self.assertEqual(result.status, "expanded")
        self.assertEqual(result.accepted_count, 1)
        self.assertEqual(result.rejected_count, 0)
        self.assertEqual(result.candidates[0].metadata["explicit_option_index"], 1)

    def test_explicit_choice_proposal_accepts_option_index(self) -> None:
        result = expand_candidates_with_llm(
            client=_client(
                {
                    "decisions": [
                        {
                            "title": "Improve Decision Card rendering",
                            "recommendation": "Choose the Decision Card polish option first.",
                            "why_now": ["It makes the product value visible in demos."],
                            "expected_result": "The next release feels clearer to users.",
                            "downside": "It does not improve model reliability.",
                            "success_signal": "Users understand the card without explanation.",
                            "confidence": 0.75,
                            "risk_level": "low",
                            "explicit_option_index": 2,
                        }
                    ]
                }
            ),
            state=_state(),
            intent_text=(
                "Compare these 3 next steps for Spice: add LLM retry, polish Decision Card, "
                "add JSON output. Which should we do first?"
            ),
            rule_candidates=[_rule_candidate()],
        )

        self.assertEqual(result.status, "expanded")
        self.assertEqual(result.accepted_count, 1)
        self.assertEqual(result.rejected_count, 0)
        self.assertEqual(result.candidates[0].metadata["explicit_option_index"], 2)
        self.assertEqual(result.candidates[0].action_type, "item.triage")

    def test_explicit_choice_proposal_rejects_missing_option_index(self) -> None:
        result = expand_candidates_with_llm(
            client=_client(
                {
                    "decisions": [
                        {
                            "title": "Improve Decision Card rendering",
                            "recommendation": "Choose the Decision Card polish option first.",
                            "why_now": ["It makes the product value visible in demos."],
                        }
                    ]
                }
            ),
            state=_state(),
            intent_text=(
                "Compare these 3 next steps for Spice: add LLM retry, polish Decision Card, "
                "add JSON output. Which should we do first?"
            ),
            rule_candidates=[_rule_candidate()],
        )

        self.assertEqual(result.status, "no_valid_candidates")
        self.assertEqual(result.accepted_count, 0)
        self.assertEqual(result.rejected_count, 1)
        self.assertIn("missing explicit_option_index", result.rejected[0]["reason"])

    def test_explicit_choice_mode_rejects_missing_option_index(self) -> None:
        result = expand_candidates_with_llm(
            client=_client(
                {
                    "candidates": [
                        {
                            "action_type": "item.triage",
                            "intent": "Compare whether polish Decision Card should be first.",
                            "user_facing_title": "polish Decision Card",
                            "recommended_action": "Choose polish Decision Card as the first release task.",
                        },
                    ]
                }
            ),
            state=_state(),
            intent_text=(
                "Compare these 3 next steps for Spice: add LLM retry, polish Decision Card, "
                "add JSON output. Which should we do first?"
            ),
            rule_candidates=[_rule_candidate()],
        )

        self.assertEqual(result.status, "no_valid_candidates")
        self.assertEqual(result.accepted_count, 0)
        self.assertEqual(result.rejected_count, 1)
        self.assertIn("missing explicit_option_index", result.rejected[0]["reason"])

    def test_explicit_choice_mode_rejects_out_of_range_option_index(self) -> None:
        result = expand_candidates_with_llm(
            client=_client(
                {
                    "candidates": [
                        {
                            "action_type": "item.triage",
                            "explicit_option_index": 9,
                            "intent": "Compare whether polish Decision Card should be first.",
                            "user_facing_title": "polish Decision Card",
                            "recommended_action": "Choose polish Decision Card as the first release task.",
                        },
                    ]
                }
            ),
            state=_state(),
            intent_text=(
                "Compare these 3 next steps for Spice: add LLM retry, polish Decision Card, "
                "add JSON output. Which should we do first?"
            ),
            rule_candidates=[_rule_candidate()],
        )

        self.assertEqual(result.status, "no_valid_candidates")
        self.assertEqual(result.accepted_count, 0)
        self.assertEqual(result.rejected_count, 1)
        self.assertIn("out of range", result.rejected[0]["reason"])

    def test_expansion_prompt_uses_compiled_context_as_primary_context(self) -> None:
        prompt = _build_expansion_prompt(
            state=_state(),
            intent_text="What should I focus on next?",
            rule_candidates=[_rule_candidate()],
            display_language="en",
            decision_context={
                "id": "decision-ctx-test",
                "context_type": "decision",
                "current_intent": {"text": "What should I focus on next?"},
                "active_decision_frame": {
                    "decision_id": "decision.previous",
                    "selected_candidate_id": "candidate.previous",
                    "candidates": [
                        {
                            "label": "A",
                            "candidate_id": "candidate.previous",
                            "title": "Polish Decision Card",
                            "recommended_action": "Polish the card.",
                            "extra_noise": "drop me",
                        }
                    ],
                },
                "recent_decisions": [{"decision_id": "decision.previous"}],
                "executor_affordance": {"executor": "codex", "permission_mode": "workspace_write"},
                "retrieved_memory": [{"id": "memory.previous", "summary": "Previous lesson"}],
            },
        )

        payload = json.loads(prompt)

        self.assertEqual(payload["context_usage"]["primary_context"], "compiled_context")
        self.assertEqual(
            payload["decision_proposal_policy"]["output_contract"],
            "lightweight_decision_proposals",
        )
        self.assertEqual(payload["decision_proposal_policy"]["top_level_field"], "decisions")
        self.assertTrue(payload["decision_proposal_policy"]["runtime_normalizes"])
        self.assertFalse(payload["decision_proposal_policy"]["execution_requested_default"])
        self.assertIn(
            "execution_intent",
            payload["decision_proposal_policy"]["forbidden_runtime_fields"],
        )
        self.assertEqual(
            payload["proposal_schema"],
            LLMDecisionProposal.response_schema(),
        )
        self.assertNotIn("candidate_schema", payload)
        self.assertNotIn("allowed_action_types", payload)
        self.assertNotIn("execution_intent_policy", payload)
        self.assertEqual(payload["state_summary"]["role"], "legacy_fallback_only")
        self.assertEqual(payload["compiled_context"]["id"], "decision-ctx-test")
        self.assertEqual(
            payload["compiled_context"]["current_intent"]["text"],
            "What should I focus on next?",
        )
        self.assertEqual(
            payload["compiled_context"]["active_decision_frame"]["candidates"][0]["title"],
            "Polish Decision Card",
        )
        self.assertNotIn(
            "extra_noise",
            payload["compiled_context"]["active_decision_frame"]["candidates"][0],
        )
        self.assertEqual(
            payload["compiled_context"]["retrieved_memory"][0]["summary"],
            "Previous lesson",
        )

    def test_execution_request_prompt_defaults_to_execution_requested(self) -> None:
        prompt = _build_expansion_prompt(
            state=_state(),
            intent_text="Create .spice-smoke/next.txt with the chosen next step.",
            rule_candidates=[_rule_candidate()],
            display_language="en",
        )

        payload = json.loads(prompt)

        self.assertEqual(payload["decision_mode"], "execution_request")
        self.assertTrue(payload["decision_proposal_policy"]["execution_requested_default"])


def _client(response_payload: dict[str, object]) -> LLMClient:
    return _client_text(json.dumps(response_payload))


def _client_text(output_text: str) -> LLMClient:
    provider = DeterministicLLMProvider(
        responses={LLMTaskHook.DECISION_PROPOSE: output_text}
    )
    model = LLMModelConfig(
        provider_id="deterministic",
        model_id="deterministic.v1",
        response_format_hint="json_object",
    )
    return LLMClient(
        registry=ProviderRegistry.empty().register(provider),
        router=LLMRouter(hook_defaults={LLMTaskHook.DECISION_PROPOSE: model}),
    )


def _state() -> GeneralDecisionState:
    return GeneralDecisionState(
        state_id="state.test",
        intents=[
            Intent(
                intent_id="intent.main",
                summary="Fix the failing test.",
                urgency="high",
            )
        ],
    )


def _rule_candidate() -> GenericCandidate:
    return GenericCandidate(
        candidate_id="candidate.rule.context",
        action_type="context.prepare",
        intent="Prepare context before acting.",
        target_refs=["intent.main"],
        requires_confirmation=False,
        side_effect_class="none",
        why_available=["Rule candidate."],
        availability_status="available",
    )


if __name__ == "__main__":
    unittest.main()
