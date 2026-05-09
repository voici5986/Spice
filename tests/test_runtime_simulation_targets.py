from __future__ import annotations

import unittest

from spice.decision.general.candidates import GenericCandidate
from spice.llm.simulation_runner import LLMSimulationResult
from spice.runtime.simulation_targets import (
    merge_simulation_result_candidates,
    select_simulation_targets,
)


class RuntimeSimulationTargetTests(unittest.TestCase):
    def test_explicit_options_are_selected_before_runtime_guardrails(self) -> None:
        candidates = [
            _candidate("candidate.context", "context.prepare"),
            _candidate(
                "candidate.option.1",
                "item.triage",
                candidate_kind="decision",
                metadata={
                    "source": "explicit_options",
                    "candidate_source": "explicit_options",
                    "explicit_option_index": 1,
                },
            ),
            _candidate(
                "candidate.option.2",
                "item.triage",
                candidate_kind="decision",
                metadata={
                    "source": "explicit_options",
                    "candidate_source": "explicit_options",
                    "explicit_option_index": 2,
                },
            ),
            _candidate("candidate.record", "state.record"),
        ]

        targets = select_simulation_targets(candidates, explicit_options=["A", "B"])

        self.assertEqual(
            [candidate.candidate_id for candidate in targets],
            ["candidate.option.1", "candidate.option.2"],
        )

    def test_llm_decision_candidates_are_limited_to_three(self) -> None:
        candidates = [
            _candidate("candidate.context", "context.prepare"),
            *[
                _candidate(
                    f"candidate.llm.{index}",
                    "item.triage",
                    candidate_kind="decision",
                    metadata={
                        "candidate_kind": "decision",
                        "candidate_source": "llm_generator",
                    },
                )
                for index in range(4)
            ],
        ]

        targets = select_simulation_targets(candidates)

        self.assertEqual(len(targets), 3)
        self.assertEqual(
            [candidate.candidate_id for candidate in targets],
            ["candidate.llm.0", "candidate.llm.1", "candidate.llm.2"],
        )

    def test_runtime_guardrails_do_not_enter_simulation_targets(self) -> None:
        candidates = [
            _candidate("candidate.context", "context.prepare"),
            _candidate("candidate.record", "state.record"),
            _candidate("candidate.defer", "time.defer"),
            _candidate("candidate.approval", "approval.request"),
            _candidate("candidate.decision", "item.triage", candidate_kind="decision"),
        ]

        targets = select_simulation_targets(candidates)

        self.assertEqual(
            [candidate.candidate_id for candidate in targets],
            ["candidate.decision"],
        )

    def test_only_guardrails_returns_no_simulation_targets(self) -> None:
        candidates = [
            _candidate("candidate.context", "context.prepare"),
            _candidate("candidate.record", "state.record"),
            _candidate("candidate.defer", "time.defer"),
            _candidate("candidate.approval", "approval.request"),
        ]

        self.assertEqual(select_simulation_targets(candidates), [])

    def test_merge_simulation_result_is_best_effort_by_candidate_id(self) -> None:
        candidates = [
            _candidate("candidate.one", "item.triage"),
            _candidate("candidate.two", "item.triage"),
        ]
        simulated = _candidate(
            "candidate.two",
            "item.triage",
            metadata={
                "llm_simulation": {
                    "candidate_id": "candidate.two",
                    "expected_outcome": "Second candidate is simulated.",
                }
            },
        )
        unknown = _candidate(
            "candidate.missing",
            "item.triage",
            metadata={
                "llm_simulation": {
                    "candidate_id": "candidate.missing",
                    "expected_outcome": "Should be ignored.",
                }
            },
        )
        result = LLMSimulationResult(
            enabled=True,
            status="simulated",
            candidates=[simulated, unknown],
            proposed_count=2,
            applied_count=1,
        )

        merged = merge_simulation_result_candidates(candidates, result)

        self.assertEqual(len(merged.candidates), 2)
        self.assertNotIn("llm_simulation", merged.candidates[0].metadata)
        self.assertEqual(
            merged.candidates[1].metadata["llm_simulation"]["expected_outcome"],
            "Second candidate is simulated.",
        )


def _candidate(
    candidate_id: str,
    action_type: str,
    *,
    candidate_kind: str = "runtime_action",
    metadata: dict[str, object] | None = None,
) -> GenericCandidate:
    return GenericCandidate(
        candidate_id=candidate_id,
        action_type=action_type,
        intent=candidate_id,
        candidate_kind=candidate_kind,
        availability_status="available",
        metadata=dict(metadata or {}),
    )


if __name__ == "__main__":
    unittest.main()
