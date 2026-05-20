from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from examples.decision_hub_demo.run_demo import build_compare_artifact
from examples.decision_hub_demo.trace import TRACE_REGISTRY
from spice.decision.compare import render_compare_json, render_compare_text
from spice.decision.compare_rich import render_compare_rich
from spice.decision.compare_payload import (
    build_compare_payload_from_trace,
    load_compare_payload,
    normalize_compare_payload,
)
from spice.entry.cli import main
from tests.helpers import repo_root


REPO_ROOT = repo_root()
FIXTURE_DIR = REPO_ROOT / "examples" / "decision_hub_demo" / "compare_artifacts"
LEGACY_FIXTURE_PATH = FIXTURE_DIR / "meetings_vs_pr_conflict.json"
NOW = datetime(2026, 4, 17, 6, 0, 0, tzinfo=timezone.utc)


class DecisionCompareTests(unittest.TestCase):
    def test_compare_payload_from_decision_hub_demo_trace(self) -> None:
        compare_payload = build_compare_artifact(now=NOW)

        self.assertIn("decision_id", compare_payload)
        self.assertIn("trace_ref", compare_payload)
        self.assertIn("decision_relevant_state_summary", compare_payload)
        self.assertIn("candidate_decisions", compare_payload)
        self.assertIn("score_breakdown", compare_payload)
        self.assertIn("selected_recommendation", compare_payload)
        self.assertIn("why_not_the_others", compare_payload)
        self.assertTrue(compare_payload["candidate_decisions"])
        candidate_ids = {
            candidate["candidate_id"]
            for candidate in compare_payload["candidate_decisions"]
        }
        self.assertIn(
            compare_payload["selected_recommendation"]["candidate_id"],
            candidate_ids,
        )
        self.assertTrue(
            compare_payload["selected_recommendation"]["candidate_id"].startswith(
                "candidate."
            )
        )

    def test_compare_reads_standard_payload_and_renders_text(self) -> None:
        payload = load_compare_payload(LEGACY_FIXTURE_PATH)

        output = render_compare_text(payload, show_execution=True)

        self.assertIn("DECISION COMPARISON", output)
        self.assertIn("decision_id:", output)
        self.assertIn("trace_ref:", output)
        self.assertIn("cand.delegate_to_executor", output)
        self.assertIn("WHY NOT OTHERS", output)
        self.assertIn("EXECUTION BOUNDARY", output)

    def test_compare_rich_renders_decision_card_sections(self) -> None:
        payload = load_compare_payload(LEGACY_FIXTURE_PATH)

        output = render_compare_rich(payload, show_execution=True, use_bars=True)

        self.assertIn("SPICE DECISION CARD", output)
        self.assertIn("GENERAL STATE", output)
        self.assertIn("CANDIDATE DECISIONS", output)
        self.assertIn("SELECTED DECISION", output)
        self.assertIn("WHY NOT OTHERS", output)
        self.assertIn("EXECUTION BOUNDARY", output)

    def test_compare_rich_respects_no_bars(self) -> None:
        payload = load_compare_payload(LEGACY_FIXTURE_PATH)

        output = render_compare_rich(payload, use_bars=False)

        self.assertIn("SPICE DECISION CARD", output)
        self.assertNotIn("█", output)
        self.assertNotIn("░", output)

    def test_compare_rich_respects_render_width(self) -> None:
        payload = load_compare_payload(LEGACY_FIXTURE_PATH)

        output = render_compare_rich(payload, use_bars=False, width=60)

        self.assertIn("SPICE DECISION CARD", output)
        self.assertTrue(all(len(line) <= 60 for line in output.splitlines()))

    def test_compare_rich_defaults_to_selected_plus_payload_order_candidates(self) -> None:
        candidates = []
        scores = {
            "cand.1": 0.1,
            "cand.2": 0.2,
            "cand.3": 0.3,
            "cand.4": 0.8,
            "cand.5": 0.9,
        }
        for index in range(1, 6):
            candidate_id = f"cand.{index}"
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "title": f"Candidate {index}",
                    "action": f"action.{index}",
                    "intent": f"Candidate {index} intent.",
                    "enabled_reason": "available",
                    "expected_effect": {},
                    "is_selected": candidate_id == "cand.3",
                }
            )
        payload = normalize_compare_payload(
            {
                "decision_id": "decision.top3",
                "trace_ref": "trace.top3",
                "decision_relevant_state_summary": {
                    "active_commitments": [],
                    "open_work_items": [],
                    "active_conflicts": [],
                    "executor_available": False,
                },
                "candidate_decisions": candidates,
                "score_breakdown": {
                    "candidates": {
                        candidate_id: {
                            "score_total": score,
                            "dimensions": [],
                            "constraints": [],
                            "vetoes": [],
                            "tradeoff_rules": [],
                        }
                        for candidate_id, score in scores.items()
                    }
                },
                "selected_recommendation": {
                    "candidate_id": "cand.3",
                    "action": "action.3",
                    "title": "Candidate 3",
                    "selection_reason": "selected from compare payload",
                    "decision_basis": [],
                },
                "why_not_the_others": [
                    {
                        "candidate_id": f"cand.{index}",
                        "title": f"Candidate {index}",
                        "reasons": [{"kind": "score", "summary": "lower score"}],
                    }
                    for index in (1, 2, 4, 5)
                ],
                "expected_outcome_or_risk": {},
            }
        )

        output = render_compare_rich(payload, use_bars=False)

        self.assertIn("Candidate 3", output)
        self.assertIn("Candidate 1", output)
        self.assertIn("Candidate 2", output)
        self.assertNotIn("Candidate 4 intent.", output)
        self.assertNotIn("Candidate 5 intent.", output)
        self.assertIn("lower-priority candidates hidden", output)
        self.assertIn("lower-priority alternatives hidden", output)

    def test_compare_rich_falls_back_silently_without_rich(self) -> None:
        import builtins
        from unittest.mock import patch

        payload = load_compare_payload(LEGACY_FIXTURE_PATH)
        expected = render_compare_text(payload, show_execution=True, use_bars=False)
        original_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "rich" or name.startswith("rich."):
                raise ImportError("rich unavailable")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=blocked_import):
            output = render_compare_rich(payload, show_execution=True, use_bars=False)

        self.assertEqual(output, expected)

    def test_compare_text_renders_product_candidate_language_before_internal_action(self) -> None:
        payload = normalize_compare_payload(
            {
                "decision_id": "decision.demo",
                "trace_ref": "trace.demo",
                "decision_relevant_state_summary": {
                    "active_commitments": [],
                    "open_work_items": [],
                    "active_conflicts": [],
                    "executor_available": True,
                },
                "candidate_decisions": [
                    {
                        "candidate_id": "cand.a",
                        "title": "Fix the failing test before the meeting",
                        "action": "intent.execute",
                        "intent": "Act on the failing test.",
                        "recommended_action": "Ask Codex to inspect the failure and make the smallest safe patch.",
                        "why_now": ["The failure blocks CI."],
                        "expected_result": "The focused test passes.",
                        "executor_task": "Fix the failing test and do not touch unrelated files.",
                        "key_constraints": [],
                        "expected_effect": {},
                        "is_selected": True,
                    }
                ],
                "score_breakdown": {
                    "candidates": {
                        "cand.a": {
                            "score_total": 0.9,
                            "dimensions": [],
                            "constraints": [],
                            "vetoes": [],
                            "tradeoff_rules": [],
                        }
                    }
                },
                "selected_recommendation": {
                    "candidate_id": "cand.a",
                    "action": "intent.execute",
                    "title": "Fix the failing test before the meeting",
                    "human_summary": "Ask Codex to inspect the failure and make the smallest safe patch.",
                    "selection_reason": "selected",
                    "decision_basis": [],
                },
                "why_not_the_others": [],
                "expected_outcome_or_risk": {},
            }
        )

        output = render_compare_text(payload, use_bars=False)
        rich_output = render_compare_rich(payload, use_bars=False)

        self.assertIn("Fix the failing test before the meeting", output)
        self.assertIn("recommendation: Ask Codex to inspect the failure", output)
        self.assertIn("why now:", output)
        self.assertIn("expected outcome if chosen: The focused test passes.", output)
        self.assertIn("expected outcome if chosen:", rich_output)
        self.assertIn("executor task: Fix the failing test", output)
        self.assertIn("internal action: intent.execute", output)

    def test_compare_text_and_rich_render_advisory_execution_affordance(self) -> None:
        payload = _execution_affordance_payload(
            {
                "candidate_executable": False,
                "blocked": True,
                "blocked_reason": (
                    "Candidate is advisory; execution_intent.intent_class is not "
                    "execution_requested."
                ),
                "executor": {"executor_id": "hermes"},
                "permission": {"configured": "read_only", "required": "read_only"},
                "approval": {"required": False},
            }
        )

        text_output = render_compare_text(payload, use_bars=False)
        rich_output = render_compare_rich(payload, use_bars=False)

        self.assertIn("not executable; advisory only; no executor handoff requested", text_output)
        self.assertIn("approval not required", text_output)
        self.assertNotIn("execution: blocked", text_output)
        self.assertIn("not executable; advisory only; no executor handoff requested", rich_output)

    def test_compare_text_and_rich_render_executable_execution_affordance(self) -> None:
        payload = _execution_affordance_payload(
            {
                "candidate_executable": True,
                "executable": True,
                "blocked": False,
                "required_capability": "code_edit",
                "executor_capability_source": "static_baseline",
                "capability": {
                    "required_capability": "code_edit",
                    "executor_has_required_capability": True,
                    "source": "static_baseline",
                    "matched_capability": "code_edit",
                    "limitations": ["Static baseline, not live tool inventory."],
                },
                "executor": {"executor_id": "codex"},
                "permission": {"configured": "read_only", "required": "workspace_write"},
                "approval": {"required": True},
            }
        )

        text_output = render_compare_text(payload, use_bars=False)
        rich_output = render_compare_rich(payload, use_bars=False)

        self.assertIn("ready for approval via codex", text_output)
        self.assertIn("permission=read_only->workspace_write", text_output)
        self.assertIn("required_capability=code_edit", text_output)
        self.assertIn("capability_source=static_baseline", text_output)
        self.assertIn("limitations=Static baseline", text_output)
        self.assertIn("approval required", text_output)
        self.assertIn("ready for approval via codex", rich_output)
        self.assertIn("required capability code_edit", rich_output)

    def test_compare_text_and_rich_render_missing_execution_capability(self) -> None:
        payload = _execution_affordance_payload(
            {
                "candidate_executable": False,
                "executable": False,
                "blocked": True,
                "blocked_reason": "Executor lacks required capability: github_work",
                "required_capability": "github_work",
                "executor_capability_source": "static_baseline",
                "capability": {
                    "required_capability": "github_work",
                    "executor_has_required_capability": False,
                    "source": "static_baseline",
                    "matched_capability": "",
                    "limitations": ["Static baseline, not live tool inventory."],
                },
                "executor": {"executor_id": "codex"},
                "permission": {"configured": "workspace_write", "required": "workspace_write"},
                "approval": {"required": True},
            }
        )

        text_output = render_compare_text(payload, use_bars=False)
        rich_output = render_compare_rich(payload, use_bars=False)

        self.assertIn("Executor lacks required capability: github_work", text_output)
        self.assertIn("missing capability=github_work", text_output)
        self.assertIn("capability_source=static_baseline", text_output)
        self.assertIn("missing capability github_work", rich_output)

    def test_compare_text_and_rich_render_candidate_simulation_metadata(self) -> None:
        payload = normalize_compare_payload(
            {
                "decision_id": "decision.sim",
                "trace_ref": "trace.sim",
                "decision_relevant_state_summary": {
                    "active_commitments": [],
                    "open_work_items": [],
                    "active_conflicts": [],
                    "executor_available": False,
                },
                "candidate_decisions": [
                    {
                        "candidate_id": "cand.sim",
                        "title": "Simulated Candidate",
                        "action": "context.prepare",
                        "intent": "Prepare the context first.",
                        "enabled_reason": "baseline",
                        "expected_effect": {},
                        "simulation": {
                            "candidate_id": "cand.sim",
                            "expected_outcome": "Context is collected before execution.",
                            "downside": "Adds a short delay",
                            "success_signal": "The next execution has fewer unknowns.",
                            "time_fit": "fits",
                            "likely_benefits": ["Lower execution risk"],
                            "likely_risks": ["Adds a short delay"],
                            "estimated_time_minutes": 4,
                            "failure_modes": ["Context still incomplete"],
                            "confidence": 0.72,
                            "source": "llm_simulation_runner",
                        },
                        "is_selected": True,
                    }
                ],
                "score_breakdown": {
                    "candidates": {
                        "cand.sim": {
                            "score_total": 0.8,
                            "dimensions": [],
                            "constraints": [],
                            "vetoes": [],
                            "tradeoff_rules": [],
                        }
                    }
                },
                "selected_recommendation": {
                    "candidate_id": "cand.sim",
                    "action": "context.prepare",
                    "title": "Simulated Candidate",
                    "selection_reason": "selected from compare payload",
                    "decision_basis": [],
                },
                "why_not_the_others": [],
                "expected_outcome_or_risk": {},
            }
        )

        text_output = render_compare_text(payload)
        rich_output = render_compare_rich(payload, use_bars=False)

        self.assertIn("LLM simulation", text_output)
        self.assertIn("Context is collected before execution.", text_output)
        self.assertIn("downside: Adds a short delay", text_output)
        self.assertIn("success signal: The next execution has fewer unknowns.", text_output)
        self.assertIn("time fit: fits", text_output)
        self.assertIn("likely risks: Adds a short delay", text_output)
        self.assertIn("confidence: 0.72", text_output)
        self.assertIn("simulation:", rich_output)
        self.assertIn("Context is collected before execution.", rich_output)
        self.assertIn("downside:", rich_output)
        self.assertIn("success:", rich_output)
        self.assertIn("time fit:", rich_output)

    def test_compare_text_and_rich_render_candidate_history_metadata(self) -> None:
        payload = normalize_compare_payload(
            {
                "decision_id": "decision.history",
                "trace_ref": "trace.history",
                "decision_relevant_state_summary": {
                    "active_commitments": [],
                    "open_work_items": [],
                    "active_conflicts": [],
                    "executor_available": False,
                },
                "candidate_decisions": [
                    {
                        "candidate_id": "cand.history",
                        "title": "History Candidate",
                        "action": "item.triage",
                        "intent": "Use the historically successful action.",
                        "enabled_reason": "baseline",
                        "expected_effect": {},
                        "history": {
                            "action_type": "item.triage",
                            "similar_outcome_count": 3,
                            "success_count": 2,
                            "failure_count": 1,
                            "partial_count": 0,
                            "historical_score": 0.67,
                            "recent_outcome_ids": ["outcome.1", "outcome.2", "outcome.3"],
                        },
                        "is_selected": True,
                    }
                ],
                "score_breakdown": {
                    "candidates": {
                        "cand.history": {
                            "score_total": 0.8,
                            "dimensions": [],
                            "constraints": [],
                            "vetoes": [],
                            "tradeoff_rules": [],
                        }
                    }
                },
                "selected_recommendation": {
                    "candidate_id": "cand.history",
                    "action": "item.triage",
                    "title": "History Candidate",
                    "selection_reason": "selected from compare payload",
                    "decision_basis": [],
                },
                "why_not_the_others": [],
                "expected_outcome_or_risk": {},
            }
        )

        text_output = render_compare_text(payload)
        rich_output = render_compare_rich(payload, use_bars=False)

        self.assertIn("history:", text_output)
        self.assertIn("2/3 success", text_output)
        self.assertIn("1 failed", text_output)
        self.assertIn("history:", rich_output)
        self.assertIn("2/3 success", rich_output)

    def test_compare_json_output_is_stable(self) -> None:
        payload = load_compare_payload(LEGACY_FIXTURE_PATH)

        rendered = json.loads(render_compare_json(payload))

        self.assertEqual(rendered["decision_id"], payload["decision_id"])
        self.assertEqual(rendered["trace_ref"], payload["trace_ref"])
        self.assertEqual(
            rendered["selected_recommendation"]["candidate_id"],
            payload["selected_recommendation"]["candidate_id"],
        )

    def test_veto_candidate_is_explained(self) -> None:
        payload = load_compare_payload(LEGACY_FIXTURE_PATH)

        output = render_compare_text(payload)

        self.assertIn("Vetoed by no_commitment_endangerment", output)
        self.assertIn("Vetoed by no_silent_blocker_ignore", output)
        self.assertIn("guided score: 0.43 (blocked by veto)", output)

    def test_tradeoff_rule_candidate_is_explained(self) -> None:
        payload = load_compare_payload(LEGACY_FIXTURE_PATH)

        output = render_compare_text(payload)

        self.assertIn(
            "The selected candidate was preferred by trade-off rule prefer_delegate_when_executor_available_and_time_pressure",
            output,
        )

    def test_missing_optional_fields_do_not_crash(self) -> None:
        payload = normalize_compare_payload(
            {
                "decision_id": "decision.demo",
                "trace_ref": "trace.demo",
                "decision_relevant_state_summary": {
                    "now": "2026-04-17T06:00:00+00:00",
                    "available_window_minutes": 15,
                    "active_commitments": [],
                    "open_work_items": [],
                    "active_conflicts": [],
                    "executor_available": False,
                },
                "candidate_decisions": [
                    {
                        "candidate_id": "cand.a",
                        "title": "Candidate A",
                        "action": "handle_now",
                        "intent": "Take the action now.",
                        "enabled_reason": "baseline",
                        "key_constraints": [],
                        "expected_effect": {},
                        "is_selected": True,
                    },
                    {
                        "candidate_id": "cand.b",
                        "title": "Candidate B",
                        "action": "ignore_temporarily",
                        "intent": "Do nothing for now.",
                        "enabled_reason": "baseline",
                        "key_constraints": [],
                        "expected_effect": {},
                        "is_selected": False,
                    },
                ],
                "score_breakdown": {
                    "candidates": {
                        "cand.a": {
                            "score_total": 0.9,
                            "dimensions": [],
                            "constraints": [],
                            "vetoes": [],
                            "tradeoff_rules": [],
                        },
                        "cand.b": {
                            "score_total": 0.2,
                            "dimensions": [],
                            "constraints": [],
                            "vetoes": [],
                            "tradeoff_rules": [],
                        },
                    }
                },
                "selected_recommendation": {
                    "candidate_id": "cand.a",
                    "action": "handle_now",
                    "title": "Candidate A",
                    "selection_reason": "selected from compare payload",
                    "decision_basis": [],
                },
                "why_not_the_others": [{"candidate_id": "cand.b", "title": "Candidate B", "reasons": []}],
                "expected_outcome_or_risk": {},
            }
        )

        output = render_compare_text(payload)

        self.assertIn("Candidate A", output)
        self.assertIn("Candidate B", output)
        self.assertIn("No explicit compare evidence was recorded", output)

    def test_compare_does_not_depend_on_live_runtime_registry(self) -> None:
        TRACE_REGISTRY.clear()
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = main(
                [
                    "decision",
                    "compare",
                    "--input",
                    str(LEGACY_FIXTURE_PATH),
                ]
            )

        self.assertEqual(code, 0)
        self.assertIn("DECISION COMPARISON", stdout.getvalue())

    def test_show_execution_is_optional(self) -> None:
        payload = load_compare_payload(LEGACY_FIXTURE_PATH)

        without_execution = render_compare_text(payload, show_execution=False)
        with_execution = render_compare_text(payload, show_execution=True)

        self.assertNotIn("EXECUTION BOUNDARY", without_execution)
        self.assertIn("EXECUTION BOUNDARY", with_execution)

    def test_why_not_is_not_invented_by_renderer(self) -> None:
        payload = normalize_compare_payload(
            {
                "decision_id": "decision.demo",
                "trace_ref": "trace.demo",
                "decision_relevant_state_summary": {
                    "now": "2026-04-17T06:00:00+00:00",
                    "available_window_minutes": 15,
                    "active_commitments": [],
                    "open_work_items": [],
                    "active_conflicts": [],
                    "executor_available": False,
                },
                "candidate_decisions": [
                    {
                        "candidate_id": "cand.selected",
                        "title": "Selected Candidate",
                        "action": "handle_now",
                        "intent": "Act immediately.",
                        "enabled_reason": "baseline",
                        "key_constraints": [],
                        "expected_effect": {},
                        "is_selected": True,
                    },
                    {
                        "candidate_id": "cand.other",
                        "title": "Other Candidate",
                        "action": "ignore_temporarily",
                        "intent": "Wait.",
                        "enabled_reason": "baseline",
                        "key_constraints": [],
                        "expected_effect": {},
                        "is_selected": False,
                    },
                ],
                "score_breakdown": {
                    "candidates": {
                        "cand.selected": {
                            "score_total": 0.5,
                            "dimensions": [],
                            "constraints": [],
                            "vetoes": [],
                            "tradeoff_rules": [],
                        },
                        "cand.other": {
                            "score_total": 0.4,
                            "dimensions": [],
                            "constraints": [],
                            "vetoes": [],
                            "tradeoff_rules": [],
                        },
                    }
                },
                "selected_recommendation": {
                    "candidate_id": "cand.selected",
                    "action": "handle_now",
                    "title": "Selected Candidate",
                    "selection_reason": "selected from compare payload",
                    "decision_basis": [],
                },
                "why_not_the_others": [
                    {
                        "candidate_id": "cand.other",
                        "title": "Other Candidate",
                        "reasons": [],
                    }
                ],
                "expected_outcome_or_risk": {},
            }
        )

        output = render_compare_text(payload)

        self.assertIn("No explicit compare evidence was recorded for this candidate.", output)
        self.assertNotIn("Vetoed by", output)
        self.assertNotIn("trade-off rule", output)

    def test_decision_hub_demo_artifact_golden(self) -> None:
        payload = load_compare_payload(LEGACY_FIXTURE_PATH)

        self.assertEqual(payload["selected_recommendation"]["candidate_id"], "cand.delegate_to_executor")
        self.assertEqual(
            payload["execution_boundary"]["execution_path"],
            "SDEP -> Hermes/Codex",
        )
        self.assertTrue(
            any(
                item["candidate_id"] == "cand.quick_triage_then_defer"
                for item in payload["why_not_the_others"]
            )
        )

    def test_cli_json_output(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = main(
                [
                    "decision",
                    "compare",
                    "--input",
                    str(LEGACY_FIXTURE_PATH),
                    "--json",
                ]
            )

        self.assertEqual(code, 0)
        rendered = json.loads(stdout.getvalue())
        self.assertEqual(rendered["decision_id"], load_compare_payload(LEGACY_FIXTURE_PATH)["decision_id"])


def _execution_affordance_payload(affordance: dict[str, object]) -> dict[str, object]:
    return normalize_compare_payload(
        {
            "decision_id": "decision.execution.copy",
            "trace_ref": "trace.execution.copy",
            "decision_relevant_state_summary": {
                "active_commitments": [],
                "open_work_items": [],
                "active_conflicts": [],
                "executor_available": True,
            },
            "candidate_decisions": [
                {
                    "candidate_id": "cand.execution.copy",
                    "title": "Improve state-as-context",
                    "action": "item.triage",
                    "intent": "Prioritize the next development direction.",
                    "enabled_reason": "available",
                    "expected_effect": {},
                    "execution_affordance": affordance,
                    "is_selected": True,
                }
            ],
            "score_breakdown": {
                "candidates": {
                    "cand.execution.copy": {
                        "score_total": 0.8,
                        "dimensions": [],
                        "constraints": [],
                        "vetoes": [],
                        "tradeoff_rules": [],
                    }
                }
            },
            "selected_recommendation": {
                "candidate_id": "cand.execution.copy",
                "action": "item.triage",
                "title": "Improve state-as-context",
                "selection_reason": "selected",
                "decision_basis": [],
            },
            "why_not_the_others": [],
            "expected_outcome_or_risk": {},
        }
    )


if __name__ == "__main__":
    unittest.main()
