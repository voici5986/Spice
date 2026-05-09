from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from spice.decision.general.candidates import (
    EstimatedCost,
    ExecutionBoundary,
    ExpectedStateDelta,
    GenericCandidate,
    GenericExecutionIntent,
    RiskProfile,
)
from spice.llm.candidate_expander import LLMCandidateExpansionResult
from spice.llm.simulation_runner import LLMSimulationResult
from spice.runtime import LocalJsonStore, run_once, setup_workspace, update_workspace_config, workspace_paths
from spice.runtime import load_workspace_memory_provider


NOW = datetime(2026, 4, 29, 6, 0, tzinfo=timezone.utc)


class RuntimeRunOnceTests(unittest.TestCase):
    def test_run_once_persists_state_run_and_decision_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)

            result = run_once(
                "Review this repo and suggest the safest next action",
                project_root=tmp_dir,
                now=NOW,
                use_bars=False,
            )
            store = LocalJsonStore.from_project_root(tmp_dir)
            state = store.load_state()
            run_payload = store.load_run(result.artifact["run_id"])
            decision_payload = store.load_decision(result.artifact["decision_id"])

            self.assertTrue(result.run_path.exists())
            self.assertTrue(result.decision_path.exists())
            self.assertTrue(result.state_path.exists())
            self.assertEqual(run_payload["path_type"], "manual_intent_run_once")
            self.assertEqual(run_payload["loop_mode"], "full_loop_preview")
            self.assertEqual(run_payload["source"], "manual_intent")
            self.assertTrue(run_payload["persisted"])
            self.assertEqual(run_payload["persist_mode"], "active_state")
            self.assertTrue(run_payload["persisted_at"])
            self.assertIn("state_before_ref", run_payload)
            self.assertIn("state_after_ref", run_payload)
            self.assertIn("store_paths", run_payload)
            self.assertIn("run", run_payload["store_paths"])
            self.assertIn("decision", run_payload["store_paths"])
            self.assertIn("state", run_payload["store_paths"])
            self.assertIn("session", run_payload["store_paths"])
            self.assertIn("full_loop_preview", run_payload)
            self.assertEqual(run_payload["full_loop_preview"]["loop_status"], "completed_read_only")
            self.assertEqual(run_payload["session_id"], "session.default")
            self.assertEqual(run_payload["session"]["last_run_id"], run_payload["run_id"])
            self.assertEqual(decision_payload["checkpoint"]["decision_id"], result.artifact["decision_id"])
            memory_records = load_workspace_memory_provider(tmp_dir).query(
                namespace="general.decision",
                limit=-1,
            )
            self.assertEqual(len(memory_records), 1)
            memory = memory_records[0]
            self.assertEqual(memory["run_id"], result.artifact["run_id"])
            self.assertEqual(memory["decision_id"], result.artifact["decision_id"])
            self.assertEqual(memory["input"]["text"], "Review this repo and suggest the safest next action")
            self.assertEqual(memory["selected"]["candidate_id"], result.artifact["selected_candidate_id"])
            self.assertEqual(memory["approval_id"], result.artifact.get("approval_id") or "")
            self.assertEqual(
                memory["context_refs"]["decision_context_id"],
                result.artifact["context_refs"]["decision_context_id"],
            )
            self.assertIn("#active_decision_frame:", memory["active_decision_frame_ref"])
            self.assertEqual(result.artifact["memory_writeback"]["status"], "written")
            self.assertEqual(
                result.artifact["memory_writeback"]["namespace"],
                "general.decision",
            )
            summary_writeback = result.artifact["memory_writeback"]["session_summary"]
            self.assertEqual(summary_writeback["status"], "written")
            self.assertEqual(summary_writeback["namespace"], "general.session_summary")
            summary_records = load_workspace_memory_provider(tmp_dir).query(
                namespace="general.session_summary",
                limit=-1,
            )
            self.assertEqual(len(summary_records), 1)
            self.assertEqual(
                summary_records[0]["current_goal"]["text"],
                "Review this repo and suggest the safest next action",
            )
            self.assertTrue((workspace_paths(tmp_dir).memory_dir / "session_summary.generated.json").exists())
            self.assertTrue((workspace_paths(tmp_dir).memory_dir / "session_summary.md").exists())
            general = state["world_state"]["domain_state"]["general_decision"]
            self.assertEqual(len(general["observations"]), 3)
            self.assertEqual(len(general["intents"]), 1)
            self.assertEqual(len(general["decision_checkpoints"]), 1)
            self.assertIn(result.artifact["trace_ref"], general["trace_refs"])
            frame = general["metadata"]["active_decision_frame"]
            self.assertEqual(frame["decision_id"], result.artifact["decision_id"])
            self.assertEqual(frame["run_id"], result.artifact["run_id"])
            self.assertEqual(frame["selected_candidate_id"], result.artifact["selected_candidate_id"])
            self.assertGreaterEqual(len(frame["candidates"]), 1)
            self.assertEqual(frame["candidates"][0]["label"], "A")
            self.assertIn("execution_affordance", frame["selected"])
            self.assertIn("skill_resolution", frame["selected"])
            self.assertTrue(frame["allowed_continuations"])

    def test_run_once_artifact_is_readable_and_does_not_execute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)

            result = run_once("Summarize the current project risk.", project_root=tmp_dir, now=NOW)
            artifact = result.artifact
            rendered = result.rendered_text

            self.assertTrue(artifact["read_only_execution"])
            self.assertFalse(artifact["executor_called"])
            self.assertFalse(artifact["sdep_request_sent"])
            self.assertFalse(artifact["executed"])
            self.assertIsNone(artifact["execution"])
            self.assertTrue(artifact["state_persisted"])
            self.assertTrue(artifact["persisted"])
            self.assertEqual(artifact["persist_mode"], "active_state")
            self.assertTrue(artifact["artifacts_persisted"])
            self.assertIn("SPICE DECISION LOOP", rendered)
            self.assertIn("DECISION COMPARISON", rendered)
            self.assertIn("WHY NOT OTHERS", rendered)
            self.assertIn("no executor called | no SDEP sent", rendered)
            self.assertIn("compiled_context", artifact)
            self.assertIn("context_refs", artifact)
            self.assertEqual(
                artifact["compiled_context"]["decision_context"]["current_intent"]["text"],
                "Summarize the current project risk.",
            )
            self.assertTrue(
                artifact["context_refs"]["decision_context_id"].startswith("decision-ctx-")
            )
            self.assertTrue(
                artifact["context_refs"]["simulation_context_id"].startswith("simulation-ctx-")
            )
            self.assertEqual(
                artifact["llm_candidate_expansion"]["context_ref"],
                artifact["context_refs"]["decision_context_id"],
            )
            self.assertEqual(
                artifact["llm_simulation"]["context_ref"],
                artifact["context_refs"]["simulation_context_id"],
            )
            json.dumps(artifact)

    def test_run_once_passes_compiled_context_to_llm_generation_and_simulation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)

            def expand_side_effect(**kwargs):
                decision_context = kwargs["decision_context"]
                self.assertEqual(decision_context.context_type, "decision")
                self.assertEqual(decision_context.current_intent["text"], "Choose the next task.")
                return LLMCandidateExpansionResult(
                    enabled=False,
                    status="disabled",
                    context_ref=decision_context.id,
                    context_type=decision_context.context_type,
                )

            def simulate_side_effect(**kwargs):
                simulation_context = kwargs["simulation_context"]
                self.assertEqual(simulation_context.context_type, "simulation")
                self.assertEqual(
                    simulation_context.current_intent["text"],
                    "Choose the next task.",
                )
                self.assertGreaterEqual(len(simulation_context.candidate_decisions), 1)
                return LLMSimulationResult(
                    enabled=False,
                    status="disabled",
                    candidates=list(kwargs["candidates"]),
                    context_ref=simulation_context.id,
                    context_type=simulation_context.context_type,
                )

            with patch(
                "spice.runtime.run_once.expand_candidates_from_runtime_config",
                side_effect=expand_side_effect,
            ), patch(
                "spice.runtime.run_once.simulate_candidates_from_runtime_config",
                side_effect=simulate_side_effect,
            ):
                result = run_once("Choose the next task.", project_root=tmp_dir, now=NOW)

            self.assertEqual(
                result.artifact["llm_candidate_expansion"]["context_ref"],
                result.artifact["context_refs"]["decision_context_id"],
            )
            self.assertEqual(
                result.artifact["llm_simulation"]["context_ref"],
                result.artifact["context_refs"]["simulation_context_id"],
            )

    def test_run_once_can_stop_at_decision_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)

            result = run_once(
                "Review only the decision.",
                project_root=tmp_dir,
                now=NOW,
                full_loop_preview=False,
            )

            self.assertEqual(result.artifact["loop_mode"], "decision_only")
            self.assertNotIn("full_loop_preview", result.artifact)
            self.assertIn("SPICE RUN ONCE", result.rendered_text)
            self.assertNotIn("SPICE DECISION LOOP", result.rendered_text)

    def test_run_once_no_persist_skips_memory_writeback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)

            result = run_once(
                "Preview only.",
                project_root=tmp_dir,
                now=NOW,
                persist=False,
            )

            self.assertEqual(result.artifact["memory_writeback"]["status"], "skipped")
            self.assertEqual(result.artifact["memory_writeback"]["reason"], "persist=false")
            records = load_workspace_memory_provider(tmp_dir).query(
                namespace="general.decision",
                limit=-1,
            )
            self.assertEqual(records, [])

    def test_run_once_advise_mode_stops_at_decision_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)

            result = run_once(
                "What should I do next?",
                project_root=tmp_dir,
                now=NOW,
                run_intent_mode="advise",
            )

            self.assertEqual(result.artifact["run_intent_mode"], "advise")
            self.assertEqual(result.artifact["loop_mode"], "decision_only")
            self.assertFalse(result.artifact["handoff_required"])
            self.assertNotIn("full_loop_preview", result.artifact)
            self.assertIn("mode: advise", result.rendered_text)

    def test_run_once_act_mode_selects_handoff_candidate_and_creates_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)

            result = run_once(
                "Fix the failing test.",
                project_root=tmp_dir,
                now=NOW,
                run_intent_mode="act",
            )
            artifact = result.artifact

            self.assertEqual(artifact["run_intent_mode"], "act")
            self.assertTrue(artifact["handoff_required"])
            self.assertTrue(artifact["approval_id"])
            self.assertIsNotNone(artifact["approval"])
            self.assertEqual(artifact["approval"]["status"], "pending")
            self.assertIn("full_loop_preview", artifact)
            self.assertEqual(artifact["full_loop_preview"]["loop_status"], "completed_read_only")
            self.assertEqual(artifact["full_loop_preview"]["approval_id"], artifact["approval_id"])
            self.assertNotEqual(
                artifact["compare_payload"]["selected_recommendation"]["action"],
                "context.prepare",
            )
            selected = artifact["compare_payload"]["selected_recommendation"]
            selected_affordance = selected["execution_affordance"]
            self.assertTrue(selected_affordance["candidate_executable"])
            self.assertTrue(selected_affordance["executable"])
            self.assertEqual(selected_affordance["executor"]["executor_id"], "dry_run")
            self.assertTrue(selected_affordance["approval"]["required"])
            selected_skill = selected["skill_resolution"]
            self.assertEqual(selected_skill["status"], "resolved")
            self.assertEqual(selected_skill["resolved_skill"]["skill_id"], artifact["skill_id"])
            self.assertEqual(artifact["skill_resolution_status"], "resolved")
            frame = artifact["active_decision_frame"]
            self.assertEqual(frame["status"], "approval_pending")
            self.assertEqual(frame["approval_id"], artifact["approval_id"])
            self.assertEqual(frame["selected"]["candidate_id"], artifact["selected_candidate_id"])
            self.assertIn(
                "approve_execute",
                {item["action"] for item in frame["allowed_continuations"]},
            )
            self.assertIn("execution:", result.rendered_text)
            self.assertIn("skill:", result.rendered_text)
            self.assertIn("handoff_required: true", result.rendered_text)

    def test_run_once_keeps_card_copy_in_chinese_for_chinese_intent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)

            result = run_once(
                "在 .spice-smoke/test.txt 中写入 OK。请勿修改其他文件。",
                project_root=tmp_dir,
                now=NOW,
                run_intent_mode="act",
                full_loop_preview=False,
            )

            artifact = result.artifact
            compare = artifact["compare_payload"]
            selected = compare["selected_recommendation"]
            selected_candidate = next(
                item
                for item in compare["candidate_decisions"]
                if item["candidate_id"] == selected["candidate_id"]
            )

            self.assertEqual(artifact["display_language"], "zh")
            self.assertEqual(compare["display_language"], "zh")
            self.assertIn("批准这个决策", selected["human_summary"])
            self.assertIn("写入 OK", selected_candidate["recommended_action"])
            self.assertTrue(
                any(
                    "贡献" in str(item.get("summary", ""))
                    for item in selected["decision_basis"]
                )
            )

    def test_aha_demo_intent_generates_multi_candidate_approval_and_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)

            result = run_once(
                "I have a failing test, a pending PR review, and a meeting in 45 minutes",
                project_root=tmp_dir,
                now=NOW,
                run_intent_mode="act",
                use_bars=True,
            )
            artifact = result.artifact
            compare = artifact["compare_payload"]
            selected = compare["selected_recommendation"]
            preview = artifact["full_loop_preview"]
            candidate_decisions = compare["candidate_decisions"]

            self.assertGreaterEqual(len(artifact["candidates"]), 3)
            candidate_actions = {item["action"] for item in candidate_decisions}
            self.assertIn(selected["action"], {"intent.execute", "capability.use"})
            self.assertTrue(artifact["approval_id"])
            self.assertFalse(artifact["handoff_blocked"])
            self.assertGreaterEqual(artifact["handoff_eligible_candidate_count"], 1)
            selected_decision = next(
                item
                for item in candidate_decisions
                if item["candidate_id"] == selected["candidate_id"]
            )
            self.assertIn("failing test", selected_decision["intent"].lower())
            self.assertEqual(artifact["approval"]["status"], "pending")
            self.assertIn(preview["skill_id"], {"runtime.intent.execute", "runtime.capability.use"})
            self.assertEqual(preview["resolved_skill"]["executor_id"], artifact["executor_id"])
            self.assertTrue(preview["context_pack_id"])
            self.assertEqual(preview["context_pack"]["candidate_id"], artifact["selected_candidate_id"])
            self.assertIn("Meeting in 45 minutes", result.rendered_text)
            self.assertIn("WHY NOT OTHERS", result.rendered_text)
            self.assertIn("SKILL RESOLUTION", result.rendered_text)
            self.assertIn("planned_executor:", result.rendered_text)

    def test_act_mode_filters_llm_planning_candidates_when_executable_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            llm_planning = GenericCandidate(
                candidate_id="candidate.llm.task_split.smoke",
                action_type="task.split",
                intent="Break the smoke file creation into smaller steps.",
                target_refs=["intent.manual.smoke"],
                estimated_cost=EstimatedCost(time_minutes=5, attention="low"),
                risk_profile=RiskProfile(level="low", uncertainty="low"),
                reversibility="high",
                requires_confirmation=True,
                expected_state_delta=ExpectedStateDelta(
                    summary="A safe plan for creating the smoke note file."
                ),
                execution_boundary=ExecutionBoundary(
                    mode="llm_proposed",
                    requires_confirmation=True,
                    side_effect_class="state_change",
                ),
                why_available=["LLM proposed a conservative planning step."],
                side_effect_class="state_change",
                availability_status="needs_confirmation",
                metadata={
                    "source": "llm_candidate_expander",
                    "user_facing_title": "Break into smaller steps",
                },
            )
            expansion = LLMCandidateExpansionResult(
                enabled=True,
                status="expanded",
                candidates=[llm_planning],
                proposed_count=1,
                accepted_count=1,
            )

            with patch(
                "spice.runtime.run_once.expand_candidates_from_runtime_config",
                return_value=expansion,
            ):
                result = run_once(
                    "Add a small smoke note file at .spice-smoke/codex_executor_smoke.txt "
                    "with the exact text: SPICE_CODEX_EXECUTOR_OK. Do not modify any other files.",
                    project_root=tmp_dir,
                    now=NOW,
                    run_intent_mode="act",
                )

            artifact = result.artifact
            selected = artifact["compare_payload"]["selected_recommendation"]
            self.assertIn(selected["action"], {"intent.execute", "capability.use"})
            self.assertTrue(artifact["approval_id"])
            self.assertFalse(artifact["handoff_blocked"])
            self.assertIn(
                "candidate.llm.task_split.smoke",
                {candidate["candidate_id"] for candidate in artifact["candidates"]},
            )
            self.assertIn(
                "candidate.llm.task_split.smoke",
                {candidate["candidate_id"] for candidate in artifact["evaluated_candidates"]},
            )
            self.assertIn(
                "candidate.llm.task_split.smoke",
                {
                    candidate["candidate_id"]
                    for candidate in artifact["compare_payload"]["candidate_decisions"]
                },
            )

    def test_act_mode_does_not_select_unanchored_llm_execute_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            llm_execute = GenericCandidate(
                candidate_id="candidate.llm.intent_execute.unanchored",
                action_type="intent.execute",
                intent="Execute an unanchored LLM-proposed action.",
                target_refs=[],
                estimated_cost=EstimatedCost(time_minutes=5, attention="low"),
                risk_profile=RiskProfile(level="low", uncertainty="low"),
                reversibility="medium",
                requires_confirmation=True,
                execution_intent=GenericExecutionIntent(
                    intent_class="execution_requested",
                    requested=True,
                    handoff_task="",
                    side_effect_class="external_effect",
                ),
                expected_state_delta=ExpectedStateDelta(summary="Unanchored execution."),
                execution_boundary=ExecutionBoundary(
                    mode="llm_proposed",
                    requires_confirmation=True,
                    side_effect_class="external_effect",
                ),
                why_available=["LLM proposed an execution-like candidate without a target."],
                side_effect_class="external_effect",
                availability_status="needs_confirmation",
                metadata={
                    "source": "llm_candidate_expander",
                    "user_facing_title": "Unanchored LLM execute",
                },
            )
            expansion = LLMCandidateExpansionResult(
                enabled=True,
                status="expanded",
                candidates=[llm_execute],
                proposed_count=1,
                accepted_count=1,
            )

            with patch(
                "spice.runtime.run_once.expand_candidates_from_runtime_config",
                return_value=expansion,
            ):
                result = run_once(
                    "Add a small smoke note file at .spice-smoke/codex_executor_smoke.txt "
                    "with the exact text: SPICE_CODEX_EXECUTOR_OK. Do not modify any other files.",
                    project_root=tmp_dir,
                    now=NOW,
                    run_intent_mode="act",
                )

            artifact = result.artifact
            self.assertNotEqual(
                artifact["selected_candidate_id"],
                "candidate.llm.intent_execute.unanchored",
            )
            self.assertIn(
                artifact["compare_payload"]["selected_recommendation"]["action"],
                {"intent.execute", "capability.use"},
            )

    def test_run_once_focuses_decision_card_on_current_observations_not_stale_intents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            run_once("quit", project_root=tmp_dir, now=NOW, full_loop_preview=False)

            result = run_once(
                "我现在有一个失败的测试、一个待 review 的 PR，45 分钟后还有会，应该先做什么？",
                project_root=tmp_dir,
                now=NOW,
                full_loop_preview=False,
            )
            candidate_text = json.dumps(
                result.artifact["compare_payload"]["candidate_decisions"],
                ensure_ascii=False,
            )

            self.assertNotIn("Prepare context before acting: quit", candidate_text)
            self.assertIn("我现在有一个失败的测试", candidate_text)
            self.assertIn("llm_candidate_expansion", result.artifact)
            self.assertIn("LLM", result.rendered_text)

    def test_run_once_does_not_overwrite_repeated_runs_for_same_intent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)

            first = run_once("Review the project.", project_root=tmp_dir, now=NOW)
            second = run_once("Review the project.", project_root=tmp_dir, now=NOW)
            store = LocalJsonStore.from_project_root(tmp_dir)

            self.assertNotEqual(first.artifact["run_id"], second.artifact["run_id"])
            self.assertTrue(first.run_path.exists())
            self.assertTrue(second.run_path.exists())
            self.assertEqual(len(store.list_record_ids("runs")), 2)
            self.assertEqual(store.load_run(first.artifact["run_id"])["run_id"], first.artifact["run_id"])
            self.assertEqual(store.load_run(second.artifact["run_id"])["run_id"], second.artifact["run_id"])

    def test_run_once_no_persist_keeps_active_state_unchanged_but_saves_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            store = LocalJsonStore.from_project_root(tmp_dir)
            before = store.load_state()

            result = run_once(
                "Review this repo without persisting state.",
                project_root=tmp_dir,
                now=NOW,
                persist=False,
            )
            after = store.load_state()

            self.assertEqual(after, before)
            self.assertTrue(result.run_path.exists())
            self.assertTrue(result.decision_path.exists())
            self.assertIn("active_decision_frame", result.artifact)
            self.assertFalse(result.artifact["persisted"])
            self.assertFalse(result.artifact["state_persisted"])
            self.assertEqual(result.artifact["persist_mode"], "no_persist")
            self.assertIsNone(result.artifact["persisted_at"])
            self.assertTrue(result.artifact["state_after_ref"].startswith("preview:"))

    def test_run_once_records_disabled_llm_candidate_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)

            result = run_once(
                "Review this repo.",
                project_root=tmp_dir,
                now=NOW,
                full_loop_preview=False,
            )

            expansion = result.artifact["llm_candidate_expansion"]
            self.assertFalse(expansion["enabled"])
            self.assertEqual(expansion["status"], "disabled")
            self.assertEqual(expansion["accepted_count"], 0)
            simulation = result.artifact["llm_simulation"]
            self.assertFalse(simulation["enabled"])
            self.assertEqual(simulation["status"], "disabled")
            self.assertEqual(simulation["applied_count"], 0)

    def test_run_once_adds_explicit_choice_candidates_when_llm_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)

            result = run_once(
                "Compare these 3 next steps for Spice: add LLM retry, polish Decision Card, "
                "add JSON output. Which should we do first?",
                project_root=tmp_dir,
                now=NOW,
                full_loop_preview=False,
            )

            explicit_titles = [
                candidate["metadata"].get("user_facing_title")
                for candidate in result.artifact["candidates"]
                if candidate["metadata"].get("source") == "explicit_options"
            ]
            self.assertEqual(
                explicit_titles,
                ["add LLM retry", "polish Decision Card", "add JSON output"],
            )
            visible_titles = [
                candidate["title"]
                for candidate in result.artifact["compare_payload"]["candidate_decisions"]
            ]
            self.assertTrue(any(title in visible_titles for title in explicit_titles))
            selected_candidate = next(
                candidate
                for candidate in result.artifact["candidates"]
                if candidate["candidate_id"] == result.artifact["selected_candidate_id"]
            )
            self.assertEqual(selected_candidate["metadata"].get("source"), "explicit_options")

    def test_run_once_explicit_choice_fallback_still_selects_user_options_not_guardrails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            expansion_result = LLMCandidateExpansionResult(
                enabled=True,
                status="fallback",
                model_provider="deterministic",
                model_id="deterministic.v1",
                error="missing decisions list",
                raw_output='{"options": "invalid"}',
            )

            with patch(
                "spice.runtime.run_once.expand_candidates_from_runtime_config",
                return_value=expansion_result,
            ):
                result = run_once(
                    "Compare these 3 next steps for Spice: add LLM retry, polish Decision Card, "
                    "add JSON output. Which should we do first?",
                    project_root=tmp_dir,
                    now=NOW,
                    full_loop_preview=False,
                )

            artifact = result.artifact
            explicit_candidates = [
                candidate
                for candidate in artifact["candidates"]
                if candidate["metadata"].get("source") == "explicit_options"
            ]
            self.assertEqual(len(explicit_candidates), 3)
            self.assertEqual(
                [candidate["metadata"].get("user_facing_title") for candidate in explicit_candidates],
                ["add LLM retry", "polish Decision Card", "add JSON output"],
            )
            selected_candidate = next(
                candidate
                for candidate in artifact["candidates"]
                if candidate["candidate_id"] == artifact["selected_candidate_id"]
            )
            self.assertEqual(selected_candidate["metadata"].get("source"), "explicit_options")
            self.assertEqual(artifact["selection_pool"]["kind"], "explicit_choice")
            self.assertIsNone(artifact["approval_id"])
            context_prepare_why_not = next(
                item
                for item in artifact["compare_payload"]["why_not_the_others"]
                if item["candidate_id"].startswith("candidate.context.prepare.")
            )
            self.assertEqual(
                context_prepare_why_not["reasons"][0]["constraint_id"],
                "selection_pool_eligible",
            )
            self.assertIn("WARNINGS", result.rendered_text)
            self.assertEqual(
                artifact["raw_model_outputs"]["llm_candidate_expansion"],
                '{"options": "invalid"}',
            )

    def test_run_once_explicit_choice_selects_indexed_llm_candidate_over_rule_meta_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            expanded_candidate = GenericCandidate(
                candidate_id="candidate.llm.item_triage.retry",
                action_type="item.triage",
                intent="Implement retry logic for LLM API calls.",
                target_refs=["intent.manual"],
                requires_confirmation=False,
                side_effect_class="state_change",
                why_available=["LLM mapped this candidate to an explicit user option."],
                availability_status="available",
                metadata={
                    "source": "llm_candidate_expander",
                    "llm_generated": True,
                    "explicit_option_index": 1,
                    "user_facing_title": "Add retry handling for LLM calls",
                    "recommended_action": "Prioritize retry handling for model calls.",
                },
            )
            expansion_result = LLMCandidateExpansionResult(
                enabled=True,
                status="expanded",
                candidates=[expanded_candidate],
                proposed_count=1,
                accepted_count=1,
                model_provider="deterministic",
                model_id="deterministic.v1",
                request_id="llm-explicit-choice",
            )

            with patch(
                "spice.runtime.run_once.expand_candidates_from_runtime_config",
                return_value=expansion_result,
            ):
                result = run_once(
                    "Compare these 3 next steps for Spice: add LLM retry, polish Decision Card, "
                    "add JSON output. Which should we do first?",
                    project_root=tmp_dir,
                    now=NOW,
                    full_loop_preview=False,
                )

            selected_candidate = next(
                candidate
                for candidate in result.artifact["candidates"]
                if candidate["candidate_id"] == result.artifact["selected_candidate_id"]
            )
            self.assertIn(
                selected_candidate["metadata"].get("explicit_option_index"),
                {1, 2, 3},
            )
            self.assertIn(
                expanded_candidate.candidate_id,
                [candidate["candidate_id"] for candidate in result.artifact["candidates"]],
            )
            visible_actions = [
                candidate["action"]
                for candidate in result.artifact["compare_payload"]["candidate_decisions"]
            ]
            self.assertIn("context.prepare", visible_actions)

    def test_run_once_explicit_choice_fills_missing_options_when_llm_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            expanded_candidate = GenericCandidate(
                candidate_id="candidate.llm.item_triage.card",
                action_type="item.triage",
                intent="Polish Decision Card for clearer demos.",
                target_refs=["explicit_option.2"],
                candidate_kind="decision",
                requires_confirmation=False,
                side_effect_class="read_only",
                why_available=["LLM mapped this candidate to explicit option 2."],
                availability_status="available",
                metadata={
                    "source": "llm_candidate_expander",
                    "candidate_kind": "decision",
                    "candidate_source": "llm_generator",
                    "llm_generated": True,
                    "explicit_option_index": 2,
                    "user_facing_title": "Polish Decision Card",
                    "recommended_action": "Choose Decision Card polish first.",
                },
            )
            expansion_result = LLMCandidateExpansionResult(
                enabled=True,
                status="expanded",
                candidates=[expanded_candidate],
                proposed_count=1,
                accepted_count=1,
                model_provider="deterministic",
                model_id="deterministic.v1",
                request_id="llm-explicit-choice-partial",
            )

            with patch(
                "spice.runtime.run_once.expand_candidates_from_runtime_config",
                return_value=expansion_result,
            ):
                result = run_once(
                    "Compare these 3 next steps for Spice: add LLM retry, polish Decision Card, "
                    "add JSON output. Which should we do first?",
                    project_root=tmp_dir,
                    now=NOW,
                    full_loop_preview=False,
                )

            explicit_choice_candidates = [
                candidate
                for candidate in result.artifact["candidates"]
                if candidate["metadata"].get("explicit_option_index") in {1, 2, 3}
            ]
            option_indices = sorted(
                {
                    candidate["metadata"].get("explicit_option_index")
                    for candidate in explicit_choice_candidates
                }
            )
            self.assertEqual(option_indices, [1, 2, 3])
            self.assertEqual(
                [
                    candidate["metadata"].get("user_facing_title")
                    for candidate in explicit_choice_candidates
                    if candidate["metadata"].get("source") == "explicit_options"
                ],
                ["add LLM retry", "add JSON output"],
            )
            self.assertIn(
                expanded_candidate.candidate_id,
                [candidate["candidate_id"] for candidate in explicit_choice_candidates],
            )
            self.assertEqual(result.artifact["selection_pool"]["kind"], "explicit_choice")
            self.assertGreaterEqual(
                len(result.artifact["selection_pool"]["candidate_ids"]),
                3,
            )

    def test_run_once_prefers_decision_candidates_over_runtime_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            decision_candidate = GenericCandidate(
                candidate_id="candidate.llm.decision.onboarding",
                action_type="item.triage",
                intent="Prioritize the onboarding fix for the next release.",
                target_refs=["intent.manual"],
                requires_confirmation=False,
                side_effect_class="state_change",
                why_available=["LLM generated a concrete decision candidate."],
                availability_status="available",
                metadata={
                    "candidate_kind": "decision",
                    "candidate_source": "llm_generator",
                    "decision_mode": "open_problem",
                    "user_facing_title": "Fix onboarding first",
                    "recommended_action": "Focus the next release on onboarding.",
                },
            )
            expansion_result = LLMCandidateExpansionResult(
                enabled=True,
                status="expanded",
                candidates=[decision_candidate],
                proposed_count=1,
                accepted_count=1,
                model_provider="deterministic",
                model_id="deterministic.v1",
                request_id="llm-decision-pool",
            )

            with patch(
                "spice.runtime.run_once.expand_candidates_from_runtime_config",
                return_value=expansion_result,
            ):
                result = run_once(
                    "My CLI has stars but low active users. What should I focus on?",
                    project_root=tmp_dir,
                    now=NOW,
                    full_loop_preview=False,
                )

            artifact = result.artifact
            self.assertEqual(artifact["selected_candidate_id"], decision_candidate.candidate_id)
            self.assertEqual(artifact["selection_pool"]["kind"], "decision_candidates")
            selected = artifact["compare_payload"]["selected_recommendation"]
            self.assertFalse(selected["execution_affordance"]["candidate_executable"])
            self.assertTrue(selected["execution_affordance"]["blocked"])
            self.assertIn(
                "execution_intent.intent_class",
                selected["execution_affordance"]["blocked_reason"],
            )
            visible_actions = [
                candidate["action"]
                for candidate in artifact["compare_payload"]["candidate_decisions"]
            ]
            self.assertIn("context.prepare", visible_actions)
            context_why_not = next(
                item
                for item in artifact["compare_payload"]["why_not_the_others"]
                if item["candidate_id"].startswith("candidate.context.prepare.")
            )
            self.assertEqual(
                context_why_not["reasons"][0]["constraint_id"],
                "selection_pool_eligible",
            )
            self.assertIn(
                "runtime actions are guardrails",
                context_why_not["reasons"][0]["reason"],
            )

    def test_run_once_merges_llm_candidate_expansion_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            expanded_candidate = GenericCandidate(
                candidate_id="candidate.llm.user_clarify.test",
                action_type="user.clarify",
                intent="Ask which failing test should be fixed first.",
                target_refs=["intent.manual"],
                requires_confirmation=False,
                side_effect_class="read_only",
                why_available=["LLM proposed a clarifying option."],
                availability_status="available",
                metadata={"source": "llm_candidate_expander", "llm_generated": True},
            )
            expansion_result = LLMCandidateExpansionResult(
                enabled=True,
                status="expanded",
                candidates=[expanded_candidate],
                proposed_count=1,
                accepted_count=1,
                model_provider="deterministic",
                model_id="deterministic.v1",
                request_id="llm-test",
            )

            with patch(
                "spice.runtime.run_once.expand_candidates_from_runtime_config",
                return_value=expansion_result,
            ):
                result = run_once(
                    "Choose between add retry, polish Decision Card, add JSON output.",
                    project_root=tmp_dir,
                    now=NOW,
                    full_loop_preview=False,
                )

            candidate_ids = [candidate["candidate_id"] for candidate in result.artifact["candidates"]]
            self.assertIn("candidate.llm.user_clarify.test", candidate_ids)
            self.assertEqual(result.artifact["llm_candidate_expansion"]["status"], "expanded")
            self.assertEqual(result.artifact["llm_candidate_expansion"]["accepted_count"], 1)

    def test_run_once_loads_workspace_env_before_llm_hooks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            update_workspace_config(tmp_dir, "llm_provider", "mimo")
            update_workspace_config(tmp_dir, "llm_model", "mimo-v2.5-pro")
            update_workspace_config(tmp_dir, "llm_api_key_env", "XIAOMI_API_KEY")
            update_workspace_config(tmp_dir, "llm_candidate_expand", "true")
            env_path = workspace_paths(tmp_dir).spice_dir / ".env"
            env_path.write_text("XIAOMI_API_KEY=from-workspace-env\n", encoding="utf-8")
            expansion_result = LLMCandidateExpansionResult(
                enabled=True,
                status="fallback",
                model_provider="mimo",
                model_id="mimo-v2.5-pro",
            )

            def assert_env_loaded(**kwargs):
                self.assertEqual(os.environ.get("XIAOMI_API_KEY"), "from-workspace-env")
                return expansion_result

            with patch.dict(os.environ, {}, clear=True), patch(
                "spice.runtime.run_once.expand_candidates_from_runtime_config",
                side_effect=assert_env_loaded,
            ):
                result = run_once(
                    "Review this repo.",
                    project_root=tmp_dir,
                    now=NOW,
                    full_loop_preview=False,
                )

            self.assertEqual(result.artifact["llm_candidate_expansion"]["model_provider"], "mimo")

    def test_run_once_advisory_decision_candidate_does_not_generate_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            advisory_candidate = GenericCandidate(
                candidate_id="candidate.llm.advisory.state_context",
                action_type="item.triage",
                intent="Prioritize state-as-context as the next product direction.",
                target_refs=["intent.manual"],
                candidate_kind="decision",
                requires_confirmation=False,
                side_effect_class="read_only",
                execution_intent=GenericExecutionIntent(
                    intent_class="advisory",
                    requested=False,
                    reason="The user asked for prioritization advice.",
                    side_effect_class="read_only",
                ),
                why_available=["LLM generated an advisory decision candidate."],
                availability_status="available",
                metadata={
                    "candidate_kind": "decision",
                    "candidate_source": "llm_generator",
                    "user_facing_title": "Prioritize state-as-context",
                    "recommended_action": "Choose state-as-context as the next priority.",
                    "execution_intent": {
                        "intent_class": "advisory",
                        "requested": False,
                    },
                },
            )
            expansion_result = LLMCandidateExpansionResult(
                enabled=True,
                status="expanded",
                candidates=[advisory_candidate],
                proposed_count=1,
                accepted_count=1,
                model_provider="deterministic",
                model_id="deterministic.v1",
            )

            with patch(
                "spice.runtime.run_once.expand_candidates_from_runtime_config",
                return_value=expansion_result,
            ):
                result = run_once(
                    "Which path should Spice prioritize next?",
                    project_root=tmp_dir,
                    now=NOW,
                    full_loop_preview=False,
                )

            artifact = result.artifact
            self.assertEqual(artifact["selected_candidate_id"], advisory_candidate.candidate_id)
            self.assertIsNone(artifact["approval_id"])
            self.assertIsNone(artifact["approval"])
            selected_affordance = artifact["compare_payload"]["selected_recommendation"][
                "execution_affordance"
            ]
            self.assertFalse(selected_affordance["approval"]["required"])
            self.assertFalse(selected_affordance["approval"]["eligible_for_approval"])
            self.assertIn("advisory", selected_affordance["blocked_reason"])

    def test_run_once_execution_requested_candidate_generates_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            execution_candidate = GenericCandidate(
                candidate_id="candidate.llm.execute.smoke_file",
                action_type="intent.execute",
                intent="Create the requested smoke note file.",
                target_refs=["intent.manual"],
                candidate_kind="decision",
                requires_confirmation=True,
                side_effect_class="external_effect",
                execution_intent=GenericExecutionIntent(
                    intent_class="execution_requested",
                    requested=True,
                    handoff_task="Create .spice-smoke/next.txt with exact text OK.",
                    side_effect_class="external_effect",
                    required_permission_hint="workspace_write",
                ),
                execution_boundary=ExecutionBoundary(
                    mode="execution_intent",
                    target=".spice-smoke/next.txt",
                    protocol="sdep",
                    requires_confirmation=True,
                    side_effect_class="external_effect",
                ),
                why_available=["The user requested a bounded file write."],
                availability_status="needs_confirmation",
                metadata={
                    "candidate_kind": "decision",
                    "candidate_source": "llm_generator",
                    "user_facing_title": "Create the smoke note",
                    "recommended_action": "Create the requested smoke note file.",
                    "execution_intent": {
                        "intent_class": "execution_requested",
                        "requested": True,
                        "handoff_task": "Create .spice-smoke/next.txt with exact text OK.",
                        "required_permission_hint": "workspace_write",
                    },
                },
            )
            expansion_result = LLMCandidateExpansionResult(
                enabled=True,
                status="expanded",
                candidates=[execution_candidate],
                proposed_count=1,
                accepted_count=1,
                model_provider="deterministic",
                model_id="deterministic.v1",
            )

            with patch(
                "spice.runtime.run_once.expand_candidates_from_runtime_config",
                return_value=expansion_result,
            ):
                result = run_once(
                    "Create .spice-smoke/next.txt with exact text OK.",
                    project_root=tmp_dir,
                    now=NOW,
                    full_loop_preview=False,
                )

            artifact = result.artifact
            self.assertEqual(artifact["selected_candidate_id"], execution_candidate.candidate_id)
            self.assertTrue(artifact["approval_id"])
            self.assertIsNotNone(artifact["approval"])
            self.assertEqual(artifact["approval"]["status"], "pending")
            selected_affordance = artifact["compare_payload"]["selected_recommendation"][
                "execution_affordance"
            ]
            self.assertTrue(selected_affordance["approval"]["required"])
            self.assertTrue(selected_affordance["approval"]["eligible_for_approval"])
            self.assertTrue(selected_affordance["candidate_executable"])

    def test_run_once_surfaces_llm_fallback_warning_and_raw_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            raw_output = '{"notes": ["no decisions here"]}'
            expansion_result = LLMCandidateExpansionResult(
                enabled=True,
                status="fallback",
                model_provider="mimo",
                model_id="mimo-v2.5-pro",
                error="missing decisions list",
                raw_output=raw_output,
            )

            with patch(
                "spice.runtime.run_once.expand_candidates_from_runtime_config",
                return_value=expansion_result,
            ):
                result = run_once(
                    "What should Spice prioritize next?",
                    project_root=tmp_dir,
                    now=NOW,
                    full_loop_preview=False,
                )

            artifact = result.artifact
            self.assertEqual(len(artifact["warnings"]), 1)
            warning = artifact["warnings"][0]
            self.assertEqual(warning["source"], "llm_candidate_expansion")
            self.assertEqual(
                warning["message"],
                "LLM candidate expansion fell back to deterministic candidates.",
            )
            self.assertEqual(warning["reason"], "missing decisions list")
            self.assertEqual(
                artifact["raw_model_outputs"]["llm_candidate_expansion"],
                raw_output,
            )
            self.assertEqual(
                artifact["compare_payload"]["warnings"][0]["reason"],
                "missing decisions list",
            )
            self.assertIn("WARNINGS", result.rendered_text)
            self.assertIn(
                "LLM candidate expansion fell back to deterministic candidates.",
                result.rendered_text,
            )
            self.assertIn("Reason: missing decisions list", result.rendered_text)

    def test_run_once_attaches_llm_simulation_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            simulation_target_ids: list[str] = []
            def simulate_side_effect(**kwargs: object) -> LLMSimulationResult:
                simulated_candidates = []
                for target in kwargs["candidates"]:  # type: ignore[index]
                    simulation_target_ids.append(target.candidate_id)
                    payload = target.to_payload()
                    metadata = dict(payload.get("metadata") or {})
                    metadata["llm_simulation"] = {
                        "candidate_id": target.candidate_id,
                        "expected_outcome": "Context will be collected before execution.",
                        "downside": "Slight delay.",
                        "success_signal": "Decision has enough context to proceed.",
                        "time_fit": "fits",
                        "simulated_outcome": "Context will be collected before execution.",
                        "likely_benefits": ["Safer execution"],
                        "likely_risks": ["Slight delay"],
                        "estimated_time_minutes": 3,
                        "failure_modes": ["Context may still be incomplete"],
                        "confidence": 0.75,
                        "source": "llm_simulation_runner",
                    }
                    payload["metadata"] = metadata
                    simulated_candidates.append(GenericCandidate.from_payload(payload))
                return LLMSimulationResult(
                    enabled=True,
                    status="simulated",
                    candidates=simulated_candidates,
                    proposed_count=len(simulated_candidates),
                    applied_count=len(simulated_candidates),
                    model_provider="deterministic",
                    model_id="deterministic.v1",
                    request_id="sim-test",
                    simulation_target_ids=list(simulation_target_ids),
                    simulation_target_count=len(simulation_target_ids),
                )

            with patch(
                "spice.runtime.run_once.simulate_candidates_from_runtime_config",
                side_effect=simulate_side_effect,
            ):
                result = run_once(
                    "Review this repo.",
                    project_root=tmp_dir,
                    now=NOW,
                    full_loop_preview=False,
                )

            self.assertEqual(result.artifact["llm_simulation"]["status"], "simulated")
            self.assertGreaterEqual(result.artifact["llm_simulation"]["applied_count"], 1)
            self.assertGreater(len(result.artifact["candidates"]), len(simulation_target_ids))
            self.assertLessEqual(len(simulation_target_ids), 3)
            self.assertEqual(
                result.artifact["llm_simulation"]["simulation_target_ids"],
                simulation_target_ids,
            )
            self.assertEqual(
                result.artifact["llm_simulation"]["simulation_target_count"],
                len(simulation_target_ids),
            )
            simulated_candidates = [
                candidate
                for candidate in result.artifact["candidates"]
                if "llm_simulation" in candidate.get("metadata", {})
            ]
            self.assertGreaterEqual(len(simulated_candidates), 1)
            self.assertLessEqual(len(simulated_candidates), 3)
            simulation = simulated_candidates[0]["metadata"]["llm_simulation"]
            self.assertEqual(
                simulation["expected_outcome"],
                "Context will be collected before execution.",
            )
            self.assertEqual(simulation["success_signal"], "Decision has enough context to proceed.")
            self.assertIn("LLM simulation", result.rendered_text)
            self.assertIn("Context will be collected before execution.", result.rendered_text)
            self.assertIn("success signal", result.rendered_text)

    def test_manual_intent_observation_preserves_original_text_as_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            text = "Preserve this exact user intent."

            result = run_once(text, project_root=tmp_dir, now=NOW)
            observation = result.artifact["observations"][0]

            self.assertEqual(observation["attributes"]["original_text"], text)
            self.assertEqual(observation["metadata"]["original_text"], text)
            evidence = observation["evidence"][0]
            self.assertEqual(evidence["kind"], "user_text")
            self.assertEqual(evidence["content"], text)

    def test_run_once_requires_initialized_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(FileNotFoundError):
                run_once("Do something", project_root=tmp_dir, now=NOW)

    def test_run_once_rejects_empty_intent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)

            with self.assertRaises(ValueError):
                run_once("   ", project_root=tmp_dir, now=NOW)


if __name__ == "__main__":
    unittest.main()
