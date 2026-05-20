from __future__ import annotations

import json
import unittest
from uuid import uuid4

from spice.llm.core import LLMRequest, LLMResponse, LLMStreamChunk
from spice.runtime.execution_response_composer import (
    compose_execution_response_from_runtime_config,
    compose_execution_response_with_llm,
    execution_response_facts,
    render_execution_response_fallback,
)


class _FakeClient:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.requests: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            provider_id="fake",
            model_id="fake-model",
            output_text=self.output_text,
            raw_payload={},
            finish_reason="stop",
            usage={},
            latency_ms=0,
            request_id=f"req.{uuid4().hex}",
        )


class _FakeStreamingClient(_FakeClient):
    def __init__(self, chunks: list[str]) -> None:
        super().__init__("")
        self.chunks = chunks

    def stream(self, request: LLMRequest):
        self.requests.append(request)
        for index, text in enumerate(self.chunks):
            yield LLMStreamChunk(
                text=text,
                finish_reason="stop" if index == len(self.chunks) - 1 else "",
                raw_event={"id": "req.stream", "model": "fake-stream-model"},
            )


class RuntimeExecutionResponseComposerTests(unittest.TestCase):
    def test_execution_response_facts_are_compact_and_include_core_execution_state(self) -> None:
        facts = execution_response_facts(
            execution_artifact=_execution_artifact(),
            context_payload={
                "active_decision_frame": {
                    "decision_id": "decision.previous",
                    "selected_candidate_id": "candidate.previous",
                    "selected": {"candidate_id": "candidate.previous", "title": "Previous choice"},
                },
                "recent_conversation_turns": [
                    {"turn_id": "turn.previous", "route": "execution_request", "user_input": "Do it."}
                ],
                "session_summary": {"summary_text": "The user wants natural execution updates."},
            },
        )

        self.assertEqual(facts["execution_status"], "completed")
        self.assertEqual(facts["approval_id"], "approval.test")
        self.assertEqual(facts["decision_id"], "decision.test")
        self.assertEqual(facts["candidate_id"], "candidate.a")
        self.assertEqual(facts["executor_provider"], "hermes")
        self.assertEqual(facts["task_status"], "success")
        self.assertTrue(facts["memory_written"])
        self.assertIn("Hermes completed", facts["executor_summary"])
        self.assertEqual(facts["state_delta_summary"]["task_status"], "success")
        self.assertEqual(facts["decision_context"]["active_decision_frame"]["decision_id"], "decision.previous")
        self.assertNotIn("rendered_text", facts)
        self.assertNotIn("sdep_request", facts)

    def test_execution_response_facts_support_error_artifact(self) -> None:
        facts = execution_response_facts(
            error_artifact={
                "execution_status": "failed",
                "approval_id": "approval.failed",
                "decision_id": "decision.failed",
                "candidate_id": "candidate.failed",
                "executor_provider": "codex",
                "error": "Executor command timed out.",
            }
        )

        self.assertEqual(facts["execution_status"], "failed")
        self.assertEqual(facts["approval_id"], "approval.failed")
        self.assertEqual(facts["executor_provider"], "codex")
        self.assertIn("timed out", facts["error"])
        self.assertEqual(facts["next_actions"], ["details", "retry", "refine"])

    def test_llm_execution_response_composer_returns_natural_response(self) -> None:
        client = _FakeClient(
            json.dumps(
                {
                    "response": (
                        "Hermes finished the handoff and recorded the result back into Spice.\n\n"
                        "The useful bit is that the state-as-context plan now has an execution outcome, "
                        "so the next decision can build on it instead of starting cold.\n\n"
                        "You can inspect details, continue from this result, or refine the decision."
                    )
                }
            )
        )

        result = compose_execution_response_with_llm(
            client=client,  # type: ignore[arg-type]
            execution_artifact=_execution_artifact(),
            deterministic_text="Hermes finished the handoff.",
            model_provider="fake",
            model_id="fake-model",
            context_payload={"session_summary": {"summary_text": "Keep execution replies conversational."}},
        )

        self.assertEqual(result.status, "composed")
        self.assertEqual(result.composer_kind, "execution_response")
        self.assertIn("Hermes finished", result.response_text)
        self.assertEqual(result.facts["candidate_id"], "candidate.a")
        payload = result.to_payload()
        self.assertEqual(payload["schema_version"], "spice.composer_result.v1")
        self.assertEqual(payload["composer_kind"], "execution_response")
        self.assertEqual(payload["facts"]["execution_status"], "completed")
        self.assertEqual(client.requests[0].task_hook.value, "response_compose")
        self.assertEqual(client.requests[0].max_tokens, 3000)
        self.assertIn("should not sound like a template", client.requests[0].system_text)
        self.assertIn("Keep execution replies conversational", client.requests[0].input_text)
        self.assertIn("Hermes completed the requested state-as-context plan.", client.requests[0].input_text)
        self.assertNotIn("rendered_text", client.requests[0].input_text)
        prompt_payload = json.loads(client.requests[0].input_text)
        self.assertIn("selected_candidate", prompt_payload["facts"])
        self.assertEqual(prompt_payload["facts"]["response_depth"]["answer_mode"], "normal")

    def test_llm_execution_response_composer_streams_then_validates_response(self) -> None:
        streamed: list[str] = []
        client = _FakeStreamingClient(
            [
                "Hermes finished the handoff ",
                "and recorded the result back into Spice.",
            ]
        )

        result = compose_execution_response_with_llm(
            client=client,  # type: ignore[arg-type]
            execution_artifact=_execution_artifact(),
            deterministic_text="Hermes finished the handoff.",
            model_provider="fake",
            model_id="fake-stream-model",
            stream_callback=streamed.append,
        )

        self.assertEqual(result.status, "composed")
        self.assertIn("Hermes finished the handoff", result.response_text)
        self.assertEqual("".join(streamed), result.raw_output)
        self.assertEqual(client.requests[0].response_format_hint, "")
        streaming = result.metadata["streaming"]
        self.assertEqual(streaming["mode"], "provider_token_stream")
        self.assertTrue(streaming["valid"])

    def test_llm_execution_response_composer_falls_back_on_invalid_output(self) -> None:
        client = _FakeClient('{"not_response": "missing"}')

        result = compose_execution_response_with_llm(
            client=client,  # type: ignore[arg-type]
            execution_artifact=_execution_artifact(),
            deterministic_text="Hermes finished the handoff.",
            model_provider="fake",
            model_id="fake-model",
        )

        self.assertEqual(result.status, "fallback")
        self.assertEqual(result.response_text, "Hermes finished the handoff.")
        self.assertEqual(result.fallback_reason, "invalid_composed_response")
        self.assertEqual(result.to_payload()["facts"]["candidate_id"], "candidate.a")
        self.assertIn("missing response text", result.error)
        self.assertTrue(result.raw_output)

    def test_llm_execution_response_composer_accepts_alias_and_plain_text(self) -> None:
        alias_result = compose_execution_response_with_llm(
            client=_FakeClient(json.dumps({"answer": "Hermes finished the handoff and recorded the result."})),  # type: ignore[arg-type]
            execution_artifact=_execution_artifact(),
            deterministic_text="Hermes finished the handoff.",
            model_provider="fake",
            model_id="fake-model",
        )
        plain_result = compose_execution_response_with_llm(
            client=_FakeClient("Hermes finished the handoff and recorded the result."),  # type: ignore[arg-type]
            execution_artifact=_execution_artifact(),
            deterministic_text="Hermes finished the handoff.",
            model_provider="fake",
            model_id="fake-model",
        )

        self.assertEqual(alias_result.status, "composed")
        self.assertEqual(plain_result.status, "composed")

    def test_validator_rejects_success_claim_for_failed_execution(self) -> None:
        client = _FakeClient(
            json.dumps({"response": "Hermes finished successfully and everything is done."})
        )

        result = compose_execution_response_with_llm(
            client=client,  # type: ignore[arg-type]
            error_artifact={
                "execution_status": "failed",
                "executor_provider": "hermes",
                "task_status": "failed",
                "error": "Permission denied.",
            },
            deterministic_text="Hermes did not complete the handoff.",
            model_provider="fake",
            model_id="fake-model",
        )

        self.assertEqual(result.status, "fallback")
        self.assertIn("success claim", result.error)

    def test_validator_rejects_failure_claim_for_successful_execution(self) -> None:
        client = _FakeClient(
            json.dumps({"response": "Hermes failed and the handoff was blocked."})
        )

        result = compose_execution_response_with_llm(
            client=client,  # type: ignore[arg-type]
            execution_artifact=_execution_artifact(),
            deterministic_text="Hermes finished the handoff.",
            model_provider="fake",
            model_id="fake-model",
        )

        self.assertEqual(result.status, "fallback")
        self.assertIn("failure claim", result.error)

    def test_validator_rejects_memory_write_claim_when_memory_was_not_written(self) -> None:
        artifact = dict(_execution_artifact())
        artifact["memory_writeback"] = {"status": "skipped", "reason": "persist=false"}
        client = _FakeClient(
            json.dumps({"response": "Hermes finished, and Spice recorded the result in memory."})
        )

        result = compose_execution_response_with_llm(
            client=client,  # type: ignore[arg-type]
            execution_artifact=artifact,
            deterministic_text="Hermes finished the handoff.",
            model_provider="fake",
            model_id="fake-model",
        )

        self.assertEqual(result.status, "fallback")
        self.assertIn("memory write", result.error)

    def test_validator_rejects_state_update_claim_when_state_was_not_updated(self) -> None:
        artifact = dict(_execution_artifact())
        artifact["state_updated"] = False
        client = _FakeClient(
            json.dumps({"response": "Hermes finished the handoff, and state was updated."})
        )

        result = compose_execution_response_with_llm(
            client=client,  # type: ignore[arg-type]
            execution_artifact=artifact,
            deterministic_text="Hermes finished the handoff.",
            model_provider="fake",
            model_id="fake-model",
        )

        self.assertEqual(result.status, "fallback")
        self.assertIn("state update", result.error)

    def test_validator_rejects_fabricated_artifact_ids(self) -> None:
        client = _FakeClient(
            json.dumps({"response": "Hermes finished the handoff and produced outcome.fake.123."})
        )

        result = compose_execution_response_with_llm(
            client=client,  # type: ignore[arg-type]
            execution_artifact=_execution_artifact(),
            deterministic_text="Hermes finished the handoff.",
            model_provider="fake",
            model_id="fake-model",
        )

        self.assertEqual(result.status, "fallback")
        self.assertIn("invented artifact id", result.error)

    def test_validator_rejects_different_executor_claim(self) -> None:
        client = _FakeClient(
            json.dumps({"response": "Codex finished the handoff and Spice recorded the outcome in memory."})
        )

        result = compose_execution_response_with_llm(
            client=client,  # type: ignore[arg-type]
            execution_artifact=_execution_artifact(),
            deterministic_text="Hermes finished the handoff.",
            model_provider="fake",
            model_id="fake-model",
        )

        self.assertEqual(result.status, "fallback")
        self.assertIn("different executor", result.error)

    def test_validator_rejects_pending_approval_language_for_execution_result(self) -> None:
        client = _FakeClient(
            json.dumps({"response": "Hermes finished, and approval is ready for you to confirm."})
        )

        result = compose_execution_response_with_llm(
            client=client,  # type: ignore[arg-type]
            execution_artifact=_execution_artifact(),
            deterministic_text="Hermes finished the handoff.",
            model_provider="fake",
            model_id="fake-model",
        )

        self.assertEqual(result.status, "fallback")
        self.assertIn("pending approval", result.error)

    def test_validator_rejects_real_execution_claim_for_dry_run(self) -> None:
        artifact = dict(_execution_artifact())
        artifact["executor_provider"] = "dry_run"
        artifact["dry_run"] = True
        artifact["real_executor_called"] = False
        client = _FakeClient(
            json.dumps({"response": "dry_run finished, and it actually executed the task for real."})
        )

        result = compose_execution_response_with_llm(
            client=client,  # type: ignore[arg-type]
            execution_artifact=artifact,
            deterministic_text="dry_run finished the handoff.",
            model_provider="fake",
            model_id="fake-model",
        )

        self.assertEqual(result.status, "fallback")
        self.assertIn("real execution", result.error)

    def test_validator_rejects_dry_run_only_claim_for_real_execution(self) -> None:
        client = _FakeClient(
            json.dumps({"response": "Hermes finished the handoff, but this was dry run only."})
        )

        result = compose_execution_response_with_llm(
            client=client,  # type: ignore[arg-type]
            execution_artifact=_execution_artifact(),
            deterministic_text="Hermes finished the handoff.",
            model_provider="fake",
            model_id="fake-model",
        )

        self.assertEqual(result.status, "fallback")
        self.assertIn("dry-run-only", result.error)

    def test_validator_rejects_changed_files_claim_without_state_delta_refs(self) -> None:
        artifact = _execution_artifact()
        artifact["outcome_record"] = {
            "summary": "Hermes completed the requested state-as-context plan.",
            "state_delta": {"task_status": "success", "executor_provider": "hermes"},
            "metadata": {"output": {"summary": "Hermes completed the requested state-as-context plan."}},
        }
        client = _FakeClient(
            json.dumps({"response": "Hermes finished and modified files for the plan."})
        )

        result = compose_execution_response_with_llm(
            client=client,  # type: ignore[arg-type]
            execution_artifact=artifact,
            deterministic_text="Hermes finished the handoff.",
            model_provider="fake",
            model_id="fake-model",
        )

        self.assertEqual(result.status, "fallback")
        self.assertIn("changed files", result.error)

    def test_validator_rejects_unread_workspace_file_claim(self) -> None:
        client = _FakeClient(
            json.dumps(
                {
                    "response": (
                        "Hermes finished the handoff. I checked `spice/runtime/missing.py` "
                        "while summarizing the result, and Spice recorded the outcome in memory."
                    )
                }
            )
        )

        result = compose_execution_response_with_llm(
            client=client,  # type: ignore[arg-type]
            execution_artifact=_execution_artifact(),
            deterministic_text="Hermes finished the handoff.",
            model_provider="fake",
            model_id="fake-model",
            context_payload={
                "workspace_context": {
                    "source": "workspace_perception",
                    "perception_id": "workspace.execution",
                    "summary": "Execution response composer receives workspace context.",
                    "files_read": [{"path": "spice/runtime/execution_response_composer.py"}],
                    "facts": [
                        {
                            "text": "execution response composer validates workspace claims.",
                            "source_path": "spice/runtime/execution_response_composer.py",
                        }
                    ],
                }
            },
        )

        self.assertEqual(result.status, "fallback")
        self.assertIn("workspace file", result.error)

    def test_validator_rejects_executor_claim_when_executor_missing(self) -> None:
        artifact = _execution_artifact()
        artifact["executor_provider"] = ""
        artifact["executor_id"] = ""
        artifact["outcome_record"] = {
            "summary": "The task completed.",
            "state_delta": {
                "task_status": "success",
                "updated_refs": ["candidate.a"],
            },
            "metadata": {"output": {"summary": "The task completed."}},
        }
        client = _FakeClient(
            json.dumps({"response": "Hermes finished the handoff and Spice recorded the outcome in memory."})
        )

        result = compose_execution_response_with_llm(
            client=client,  # type: ignore[arg-type]
            execution_artifact=artifact,
            deterministic_text="The executor finished the handoff.",
            model_provider="fake",
            model_id="fake-model",
        )

        self.assertEqual(result.status, "fallback")
        self.assertIn("invented executor", result.error)

    def test_validator_rejects_raw_structured_response_text(self) -> None:
        client = _FakeClient(
            json.dumps({"response": '{"status": "success", "summary": "Hermes finished"}'})
        )

        result = compose_execution_response_with_llm(
            client=client,  # type: ignore[arg-type]
            execution_artifact=_execution_artifact(),
            deterministic_text="Hermes finished the handoff.",
            model_provider="fake",
            model_id="fake-model",
        )

        self.assertEqual(result.status, "fallback")
        self.assertIn("structured data", result.error)

    def test_runtime_config_uses_deterministic_fallback_when_provider_is_deterministic(self) -> None:
        result = compose_execution_response_from_runtime_config(
            config={"llm_provider": "deterministic", "llm_model": ""},
            execution_artifact=_execution_artifact(),
        )

        self.assertFalse(result.enabled)
        self.assertEqual(result.status, "disabled")
        self.assertIn("hermes finished the handoff", result.response_text.lower())

    def test_deterministic_fallback_renders_failed_execution(self) -> None:
        facts = execution_response_facts(
            error_artifact={
                "execution_status": "failed",
                "executor_provider": "hermes",
                "task_status": "failed",
                "error": "Permission denied.",
            }
        )

        text = render_execution_response_fallback(facts)

        self.assertIn("hermes did not complete", text.lower())
        self.assertIn("Permission denied", text)
        self.assertIn("retry", text)

    def test_deterministic_fallback_hides_technical_approval_request_mismatch(self) -> None:
        raw_error = "Approved approval does not match the SDEP request approval_id."
        facts = execution_response_facts(
            error_artifact={
                "execution_status": "failed",
                "executor_provider": "codex",
                "task_status": "failed",
                "error": raw_error,
            }
        )

        text = render_execution_response_fallback(facts)

        self.assertEqual(facts["failure_kind"], "approval_request_mismatch")
        self.assertEqual(facts["technical_error"], raw_error)
        self.assertIn("current selection is not an executable task", text)
        self.assertIn("did not call the executor", text)
        self.assertNotIn("Approved approval", text)
        self.assertNotIn("SDEP request", text)
        self.assertNotIn("mismatch", text.lower())

    def test_validator_rejects_technical_mismatch_copy(self) -> None:
        raw_error = "Approved approval does not match the SDEP request approval_id."
        client = _FakeClient(
            json.dumps({"response": "Approved approval does not match the SDEP request approval_id."})
        )

        result = compose_execution_response_with_llm(
            client=client,  # type: ignore[arg-type]
            error_artifact={
                "execution_status": "failed",
                "executor_provider": "codex",
                "task_status": "failed",
                "error": raw_error,
            },
            model_provider="fake",
            model_id="fake-model",
        )

        self.assertEqual(result.status, "fallback")
        self.assertIn("technical approval/SDEP mismatch", result.error)
        self.assertIn("current selection is not an executable task", result.response_text)
        self.assertNotIn("Approved approval", result.response_text)


def _execution_artifact() -> dict[str, object]:
    return {
        "approval_id": "approval.test",
        "decision_id": "decision.test",
        "trace_ref": "trace.test",
        "candidate_id": "candidate.a",
        "candidate_title": "Prioritize state-as-context",
        "candidate_summary": "Use Spice state as context for future decisions.",
        "executor_provider": "hermes",
        "executor_id": "spice.hermes",
        "protocol_status": "success",
        "task_status": "success",
        "outcome_id": "outcome.test",
        "execution_id": "execution.test",
        "request_id": "request.test",
        "state_updated": True,
        "dry_run": False,
        "real_executor_called": True,
        "permission": {"mode": "workspace_write", "required": "workspace_write", "granted": True},
        "outcome_record": {
            "summary": "Hermes completed the requested state-as-context plan.",
            "state_delta": {
                "task_status": "success",
                "executor_provider": "hermes",
                "updated_refs": ["candidate.a"],
            },
            "metadata": {
                "output": {
                    "summary": "Hermes completed the requested state-as-context plan.",
                    "state_delta": {"task_status": "success", "executor_provider": "hermes"},
                    "stdout": "long executor output should not be forwarded as raw UI text",
                }
            },
        },
        "memory_writeback": {"status": "written", "namespace": "general.reflection"},
        "rendered_text": "large panel text should not be passed into the composer facts",
        "sdep_request": {"large": "payload should not be passed into the composer facts"},
    }


if __name__ == "__main__":
    unittest.main()
