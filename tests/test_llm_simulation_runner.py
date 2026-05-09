from __future__ import annotations

import json
import unittest

from spice.decision.general.candidates import GenericCandidate
from spice.decision.general.state import GeneralDecisionState
from spice.decision.general.types import Constraint, Intent, WorkItem
from spice.llm.core import (
    LLMClient,
    LLMModelConfig,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMRouter,
    LLMTaskHook,
    ProviderRegistry,
)
from spice.llm.providers import DeterministicLLMProvider
from spice.llm.simulation_runner import (
    SIMULATION_OUTPUT_MALFORMED_ERROR,
    _build_simulation_prompt,
    simulate_candidates_from_runtime_config,
    simulate_candidates_with_llm,
)


class LLMSimulationRunnerTests(unittest.TestCase):
    def test_simulation_metadata_is_attached_to_matching_candidates(self) -> None:
        candidates = [_candidate("candidate.one"), _candidate("candidate.two")]
        result = simulate_candidates_with_llm(
            client=_client(
                {
                    "simulations": [
                        {
                            "candidate_id": "candidate.one",
                            "expected_outcome": "The task is clarified before execution.",
                            "downside": "Adds one extra turn.",
                            "success_signal": "The user confirms the missing test context.",
                            "time_fit": "fits",
                            "likely_benefits": ["Less rework"],
                            "likely_risks": ["Adds one extra turn"],
                            "estimated_time_minutes": 5,
                            "failure_modes": ["User may not respond"],
                            "confidence": 0.8,
                        }
                    ]
                }
            ),
            state=_state(),
            intent_text="Fix the failing test.",
            candidates=candidates,
            model_provider="deterministic",
            model_id="deterministic.v1",
        )

        self.assertEqual(result.status, "simulated")
        self.assertEqual(result.proposed_count, 1)
        self.assertEqual(result.applied_count, 1)
        self.assertEqual(result.rejected_count, 0)
        self.assertEqual(
            result.simulation_target_ids,
            ["candidate.one", "candidate.two"],
        )
        self.assertEqual(result.simulation_target_count, 2)
        simulated = result.candidates[0].metadata["llm_simulation"]
        self.assertEqual(simulated["candidate_id"], "candidate.one")
        self.assertEqual(simulated["expected_outcome"], "The task is clarified before execution.")
        self.assertEqual(simulated["simulated_outcome"], "The task is clarified before execution.")
        self.assertEqual(simulated["downside"], "Adds one extra turn.")
        self.assertEqual(simulated["success_signal"], "The user confirms the missing test context.")
        self.assertEqual(simulated["time_fit"], "fits")
        self.assertEqual(simulated["likely_benefits"], ["Less rework"])
        self.assertEqual(simulated["estimated_time_minutes"], 5)
        self.assertEqual(simulated["confidence"], 0.8)
        self.assertNotIn("llm_simulation", result.candidates[1].metadata)
        payload = result.to_payload()
        self.assertEqual(payload["simulations"][0]["candidate_id"], "candidate.one")
        self.assertEqual(
            payload["simulation_target_ids"],
            ["candidate.one", "candidate.two"],
        )
        self.assertEqual(payload["simulation_target_count"], 2)
        json.dumps(payload)

    def test_legacy_simulation_payload_is_normalized_to_structured_fields(self) -> None:
        result = simulate_candidates_with_llm(
            client=_client(
                {
                    "simulations": [
                        {
                            "candidate_id": "candidate.one",
                            "simulated_outcome": "Legacy outcome text.",
                            "likely_risks": ["Risk one", "Risk two"],
                            "confidence": 0.5,
                        }
                    ]
                }
            ),
            state=_state(),
            intent_text="Fix the failing test.",
            candidates=[_candidate("candidate.one")],
        )

        simulation = result.candidates[0].metadata["llm_simulation"]
        self.assertEqual(simulation["expected_outcome"], "Legacy outcome text.")
        self.assertEqual(simulation["downside"], "Risk one; Risk two")
        self.assertEqual(simulation["time_fit"], "unknown")

    def test_unknown_candidate_id_is_rejected_without_mutating_candidates(self) -> None:
        candidates = [_candidate("candidate.one")]
        result = simulate_candidates_with_llm(
            client=_client(
                {
                    "simulations": [
                        {
                            "candidate_id": "candidate.missing",
                            "simulated_outcome": "Unknown candidate.",
                            "confidence": 0.5,
                        }
                    ]
                }
            ),
            state=_state(),
            intent_text="Fix the failing test.",
            candidates=candidates,
        )

        self.assertEqual(result.status, "no_valid_simulations")
        self.assertEqual(result.applied_count, 0)
        self.assertEqual(result.rejected_count, 1)
        self.assertEqual(result.matched_simulation_ids, [])
        self.assertEqual(result.unmatched_simulation_ids, ["candidate.missing"])
        self.assertEqual(result.rejected[0]["reason"], "unknown candidate_id")
        self.assertNotIn("llm_simulation", result.candidates[0].metadata)
        self.assertNotIn("llm_simulation", candidates[0].metadata)
        payload = result.to_payload()
        self.assertEqual(payload["matched_simulation_ids"], [])
        self.assertEqual(payload["unmatched_simulation_ids"], ["candidate.missing"])

    def test_mixed_matching_and_unknown_candidate_ids_are_recorded(self) -> None:
        result = simulate_candidates_with_llm(
            client=_client(
                {
                    "simulations": [
                        {
                            "candidate_id": "candidate.one",
                            "expected_outcome": "Known candidate.",
                            "confidence": 0.6,
                        },
                        {
                            "candidate_id": "candidate.missing",
                            "expected_outcome": "Unknown candidate.",
                            "confidence": 0.4,
                        },
                    ]
                }
            ),
            state=_state(),
            intent_text="Fix the failing test.",
            candidates=[_candidate("candidate.one")],
        )

        self.assertEqual(result.status, "simulated")
        self.assertEqual(result.applied_count, 1)
        self.assertEqual(result.rejected_count, 1)
        self.assertEqual(result.matched_simulation_ids, ["candidate.one"])
        self.assertEqual(result.unmatched_simulation_ids, ["candidate.missing"])
        self.assertEqual(
            result.candidates[0].metadata["llm_simulation"]["expected_outcome"],
            "Known candidate.",
        )
        payload = result.to_payload()
        self.assertEqual(payload["matched_simulation_ids"], ["candidate.one"])
        self.assertEqual(payload["unmatched_simulation_ids"], ["candidate.missing"])

    def test_runtime_config_disabled_returns_original_candidates(self) -> None:
        candidates = [_candidate("candidate.one")]
        result = simulate_candidates_from_runtime_config(
            config={"llm_simulation": "false", "llm_provider": "deterministic"},
            state=_state(),
            intent_text="Fix the failing test.",
            candidates=candidates,
        )

        self.assertFalse(result.enabled)
        self.assertEqual(result.status, "disabled")
        self.assertEqual(result.candidates, candidates)
        self.assertEqual(result.simulation_target_ids, ["candidate.one"])
        self.assertEqual(result.simulation_target_count, 1)

    def test_runtime_config_missing_model_falls_back_without_raising(self) -> None:
        result = simulate_candidates_from_runtime_config(
            config={"llm_simulation": "true", "llm_provider": "openai"},
            state=_state(),
            intent_text="Fix the failing test.",
            candidates=[_candidate("candidate.one")],
        )

        self.assertTrue(result.enabled)
        self.assertEqual(result.status, "fallback")
        self.assertIn("llm_model", result.error)

    def test_missing_simulations_list_falls_back_with_raw_output(self) -> None:
        raw_output = '{"notes": ["no simulations here"]}'

        result = simulate_candidates_with_llm(
            client=_client_text(raw_output),
            state=_state(),
            intent_text="Fix the failing test.",
            candidates=[_candidate("candidate.one")],
        )

        self.assertTrue(result.enabled)
        self.assertEqual(result.status, "fallback")
        self.assertEqual(result.error, "missing simulations list")
        self.assertEqual(result.raw_output, raw_output)
        self.assertEqual(result.to_payload()["raw_output"], raw_output)

    def test_truncated_simulation_json_falls_back_with_malformed_error(self) -> None:
        raw_output = (
            '{"simulations": [{"candidate_id": "candidate.one", '
            '"expected_outcome": "Partial output", "likely_risks": ['
        )

        result = simulate_candidates_with_llm(
            client=_client_text(raw_output),
            state=_state(),
            intent_text="Fix the failing test.",
            candidates=[_candidate("candidate.one")],
        )

        self.assertTrue(result.enabled)
        self.assertEqual(result.status, "fallback")
        self.assertEqual(result.error, SIMULATION_OUTPUT_MALFORMED_ERROR)
        self.assertEqual(result.raw_output, raw_output)
        self.assertEqual(result.simulation_target_ids, ["candidate.one"])
        self.assertEqual(result.simulation_target_count, 1)
        payload = result.to_payload()
        self.assertEqual(payload["raw_output"], raw_output)
        self.assertEqual(payload["simulation_target_ids"], ["candidate.one"])
        self.assertEqual(payload["simulation_target_count"], 1)

    def test_simulation_prompt_uses_compiled_context_as_primary_context(self) -> None:
        prompt = _build_simulation_prompt(
            state=_state(),
            intent_text="What should I focus on next?",
            candidates=[_candidate("candidate.one")],
            display_language="en",
            simulation_context={
                "id": "simulation-ctx-test",
                "context_type": "simulation",
                "decision_context_ref": "decision-ctx-test",
                "current_intent": {"text": "What should I focus on next?"},
                "candidate_decisions": [
                    {
                        "candidate_id": "candidate.one",
                        "intent": "Polish Decision Card",
                        "metadata": {"expected_result": "Better demo clarity"},
                    }
                ],
                "active_decision_frame": {
                    "decision_id": "decision.previous",
                    "selected_candidate_id": "candidate.previous",
                    "candidates": [
                        {
                            "label": "A",
                            "candidate_id": "candidate.previous",
                            "title": "Add retry",
                            "extra_noise": "drop me",
                        }
                    ],
                },
                "executor_affordance": {"executor": "codex", "permission_mode": "workspace_write"},
                "historical_analogs": [{"id": "memory.similar", "summary": "Similar choice"}],
            },
        )

        payload = json.loads(prompt)

        self.assertEqual(payload["context_usage"]["primary_context"], "compiled_context")
        self.assertEqual(payload["state_summary"]["role"], "legacy_fallback_only")
        self.assertEqual(payload["compiled_context"]["id"], "simulation-ctx-test")
        self.assertEqual(
            payload["compiled_context"]["decision_context_ref"],
            "decision-ctx-test",
        )
        self.assertEqual(
            payload["compiled_context"]["candidate_decisions"][0]["intent"],
            "Polish Decision Card",
        )
        self.assertEqual(
            payload["compiled_context"]["active_decision_frame"]["candidates"][0]["title"],
            "Add retry",
        )
        self.assertNotIn(
            "extra_noise",
            payload["compiled_context"]["active_decision_frame"]["candidates"][0],
        )

    def test_simulation_system_prompt_requests_only_slim_fields(self) -> None:
        from spice.llm.simulation_runner import _system_prompt

        prompt = _system_prompt("en")

        self.assertIn("candidate_id", prompt)
        self.assertIn("expected_outcome", prompt)
        self.assertIn("downside", prompt)
        self.assertIn("success_signal", prompt)
        self.assertIn("confidence", prompt)
        self.assertIn("Compress likely risks", prompt)
        self.assertIn("Do not include extra fields", prompt)
        self.assertNotIn("time_fit", prompt)
        self.assertNotIn("likely_benefits", prompt)
        self.assertNotIn("likely_risks", prompt)
        self.assertNotIn("estimated_time_minutes", prompt)
        self.assertNotIn("failure_modes", prompt)

    def test_simulation_request_uses_dynamic_token_budget(self) -> None:
        for candidate_count, expected_tokens in ((1, 700), (2, 900), (3, 1100), (4, 1200)):
            with self.subTest(candidate_count=candidate_count):
                provider = _RecordingProvider(
                    {
                        "simulations": [
                            {
                                "candidate_id": "candidate.0",
                                "expected_outcome": "Candidate is simulated.",
                            }
                        ]
                    }
                )
                simulate_candidates_with_llm(
                    client=_client_for_provider(provider),
                    state=_state(),
                    intent_text="Fix the failing test.",
                    candidates=[_candidate(f"candidate.{index}") for index in range(candidate_count)],
                )

                self.assertIsNotNone(provider.last_request)
                self.assertEqual(provider.last_request.max_tokens, expected_tokens)


def _client(response_payload: dict[str, object]) -> LLMClient:
    provider = DeterministicLLMProvider(
        responses={LLMTaskHook.SIMULATION_ADVISE: json.dumps(response_payload)}
    )
    model = LLMModelConfig(
        provider_id="deterministic",
        model_id="deterministic.v1",
        response_format_hint="json_object",
    )
    return LLMClient(
        registry=ProviderRegistry.empty().register(provider),
        router=LLMRouter(hook_defaults={LLMTaskHook.SIMULATION_ADVISE: model}),
    )


def _client_text(output_text: str) -> LLMClient:
    provider = DeterministicLLMProvider(
        responses={LLMTaskHook.SIMULATION_ADVISE: output_text}
    )
    model = LLMModelConfig(
        provider_id="deterministic",
        model_id="deterministic.v1",
        response_format_hint="json_object",
    )
    return LLMClient(
        registry=ProviderRegistry.empty().register(provider),
        router=LLMRouter(hook_defaults={LLMTaskHook.SIMULATION_ADVISE: model}),
    )


def _client_for_provider(provider: LLMProvider) -> LLMClient:
    model = LLMModelConfig(
        provider_id=provider.provider_id,
        model_id="recording.v1",
        response_format_hint="json_object",
    )
    return LLMClient(
        registry=ProviderRegistry.empty().register(provider),
        router=LLMRouter(hook_defaults={LLMTaskHook.SIMULATION_ADVISE: model}),
    )


class _RecordingProvider(LLMProvider):
    provider_id = "recording"

    def __init__(self, response_payload: dict[str, object]) -> None:
        self.response_payload = response_payload
        self.last_request: LLMRequest | None = None

    def generate(self, request: LLMRequest, model: LLMModelConfig) -> LLMResponse:
        self.last_request = request
        output_text = json.dumps(self.response_payload)
        return LLMResponse(
            provider_id=self.provider_id,
            model_id=model.model_id,
            output_text=output_text,
            raw_payload={},
            finish_reason="stop",
            usage={},
            latency_ms=0,
            request_id="recording-request",
        )


def _state() -> GeneralDecisionState:
    return GeneralDecisionState(
        state_id="state.test",
        intents=[Intent(intent_id="intent.main", summary="Fix the failing test.", urgency="high")],
        work_items=[
            WorkItem(
                work_item_id="work.fix_test",
                title="Fix failing test",
                urgency="high",
                status="open",
            )
        ],
        constraints=[
            Constraint(
                constraint_id="constraint.keep_safe",
                kind="safety",
                description="Do not execute without approval.",
                severity="hard",
            )
        ],
    )


def _candidate(candidate_id: str) -> GenericCandidate:
    return GenericCandidate(
        candidate_id=candidate_id,
        action_type="user.clarify",
        intent="Ask which failing test should be fixed first.",
        target_refs=["intent.main"],
        requires_confirmation=False,
        side_effect_class="read_only",
        why_available=["Clarifying is safe."],
        availability_status="available",
    )


if __name__ == "__main__":
    unittest.main()
