from __future__ import annotations

import json
import tempfile
import unittest

from spice.llm.core import LLMRequest, LLMResponse
from spice.memory import FileMemoryProvider
from spice.runtime.session_summary import (
    GENERAL_SESSION_SUMMARY_NAMESPACE,
    build_deterministic_session_summary,
    update_session_summary,
    update_deterministic_session_summary,
)


class _FakeSummaryClient:
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
            request_id="summary-req-1",
        )


class RuntimeSessionSummaryTests(unittest.TestCase):
    def test_build_deterministic_session_summary_compacts_decisions_and_reflections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            provider = FileMemoryProvider(tmp_dir)
            provider.write(
                [
                    {
                        "id": "memory.general.decision.run.1",
                        "session_id": "session.default",
                        "run_id": "run.1",
                        "decision_id": "decision.1",
                        "input": {"text": "Pick the next release task."},
                        "selected": {
                            "candidate_id": "candidate.1",
                            "title": "Polish Decision Card",
                            "recommendation": "Improve the Decision Card first.",
                        },
                        "approval_id": "approval.1",
                        "handoff": {"required": True},
                    }
                ],
                namespace="general.decision",
            )
            provider.write(
                [
                    {
                        "id": "memory.general.reflection.outcome.1",
                        "session_id": "session.default",
                        "run_id": "run.1",
                        "decision_id": "decision.1",
                        "approval_id": "approval.1",
                        "candidate_id": "candidate.1",
                        "selected_candidate": {"title": "Polish Decision Card"},
                        "executor": {"provider": "dry_run"},
                        "execution": {
                            "task_status": "success",
                            "protocol_status": "success",
                            "success": True,
                            "state_updated": True,
                            "outcome_id": "outcome.1",
                        },
                    }
                ],
                namespace="general.reflection",
            )

            summary = build_deterministic_session_summary(provider)

            self.assertEqual(summary["record_type"], "general.session_summary")
            self.assertEqual(summary["current_goal"]["text"], "Pick the next release task.")
            self.assertEqual(summary["active_decision"]["title"], "Polish Decision Card")
            self.assertEqual(summary["recent_decisions"][0]["decision_id"], "decision.1")
            self.assertEqual(summary["execution_outcomes"][0]["task_status"], "success")
            self.assertIn("## Recent Decisions", summary["markdown"])

    def test_update_deterministic_session_summary_writes_jsonl_and_latest_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            provider = FileMemoryProvider(tmp_dir)
            provider.write(
                [
                    {
                        "id": "memory.general.decision.run.1",
                        "run_id": "run.1",
                        "decision_id": "decision.1",
                        "input": {"text": "Decide what to do next."},
                        "selected": {
                            "candidate_id": "candidate.1",
                            "title": "Add LLM retry",
                            "recommendation": "Add retry support.",
                        },
                    }
                ],
                namespace="general.decision",
            )

            result = update_deterministic_session_summary(provider)

            self.assertEqual(result["status"], "written")
            self.assertEqual(result["namespace"], GENERAL_SESSION_SUMMARY_NAMESPACE)
            records = provider.query(namespace=GENERAL_SESSION_SUMMARY_NAMESPACE, limit=-1)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["current_goal"]["text"], "Decide what to do next.")
            generated = json.loads((provider.base_dir / "session_summary.generated.json").read_text())
            self.assertEqual(generated["active_decision"]["title"], "Add LLM retry")
            markdown = (provider.base_dir / "session_summary.md").read_text()
            self.assertIn("Add retry support.", markdown)

    def test_update_session_summary_triggers_llm_after_new_record_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            provider = FileMemoryProvider(tmp_dir)
            for index in range(2):
                provider.write(
                    [
                        {
                            "id": f"memory.general.decision.run.{index}",
                            "run_id": f"run.{index}",
                            "decision_id": f"decision.{index}",
                            "input": {"text": "Pick next release task."},
                            "selected": {
                                "candidate_id": f"candidate.{index}",
                                "title": "Polish Decision Card",
                                "recommendation": "Improve the card.",
                            },
                        }
                    ],
                    namespace="general.decision",
                )
            client = _FakeSummaryClient(
                json.dumps(
                    {
                        "current_goal": {"text": "Pick next release task."},
                        "active_decision": {"title": "Polish Decision Card"},
                        "recent_decisions": [
                            {
                                "decision_id": "decision.1",
                                "title": "Polish Decision Card",
                                "recommendation": "Improve the card.",
                                "evidence_refs": ["memory.general.decision.run.1"],
                            }
                        ],
                        "open_threads": [],
                    }
                )
            )

            result = update_session_summary(
                provider,
                config={
                    "memory_summary_provider": "llm",
                    "memory_summary_llm_min_new_records": "1",
                    "memory_summary_trigger_chars": "999999",
                    "memory_summary_target_chars": "2000",
                    "llm_provider": "openai",
                    "llm_model": "gpt-test",
                },
                llm_client=client,  # type: ignore[arg-type]
            )

            self.assertEqual(result["summary_source"], "llm")
            self.assertEqual(result["llm_summary"]["status"], "written")
            self.assertEqual(len(client.requests), 1)
            records = provider.query(namespace=GENERAL_SESSION_SUMMARY_NAMESPACE, limit=-1)
            self.assertEqual(records[-1]["summary_type"], "llm")
            self.assertIn("summary-req-1", json.dumps(records[-1]["model"]))
            generated = json.loads((provider.base_dir / "session_summary.generated.json").read_text())
            self.assertEqual(generated["summary_type"], "llm")

    def test_update_session_summary_skips_llm_until_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            provider = FileMemoryProvider(tmp_dir)
            provider.write(
                [
                    {
                        "id": "memory.general.decision.run.1",
                        "run_id": "run.1",
                        "decision_id": "decision.1",
                        "input": {"text": "Decide next."},
                    }
                ],
                namespace="general.decision",
            )

            result = update_session_summary(
                provider,
                config={
                    "memory_summary_provider": "llm",
                    "memory_summary_llm_min_new_records": "10",
                    "memory_summary_trigger_chars": "999999",
                    "llm_provider": "openai",
                    "llm_model": "gpt-test",
                },
                llm_client=_FakeSummaryClient("{}"),  # type: ignore[arg-type]
            )

            self.assertEqual(result["summary_source"], "deterministic")
            self.assertEqual(result["llm_summary"]["status"], "skipped")
            records = provider.query(namespace=GENERAL_SESSION_SUMMARY_NAMESPACE, limit=-1)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["summary_type"], "deterministic")

    def test_update_session_summary_falls_back_when_llm_returns_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            provider = FileMemoryProvider(tmp_dir)
            provider.write(
                [
                    {
                        "id": "memory.general.decision.run.1",
                        "run_id": "run.1",
                        "decision_id": "decision.1",
                        "input": {"text": "Decide next."},
                    }
                ],
                namespace="general.decision",
            )

            result = update_session_summary(
                provider,
                config={
                    "memory_summary_provider": "llm",
                    "memory_summary_llm_min_new_records": "1",
                    "memory_summary_trigger_chars": "999999",
                    "llm_provider": "openai",
                    "llm_model": "gpt-test",
                },
                llm_client=_FakeSummaryClient("not json"),  # type: ignore[arg-type]
            )

            self.assertEqual(result["summary_source"], "deterministic")
            self.assertEqual(result["llm_summary"]["status"], "fallback")
            records = provider.query(namespace=GENERAL_SESSION_SUMMARY_NAMESPACE, limit=-1)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["summary_type"], "deterministic")


if __name__ == "__main__":
    unittest.main()
