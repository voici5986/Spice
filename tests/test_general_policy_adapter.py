from __future__ import annotations

import json
import unittest

from spice.decision import (
    DecisionGuidance,
    HardConstraintGuidance,
    PrimaryObjectiveGuidance,
)
from spice.decision.compare import render_compare_text
from spice.decision.general import (
    Capability,
    EstimatedCost,
    ExecutionBoundary,
    ExpectedStateDelta,
    GeneralDecisionState,
    GenericCandidate,
    GenericPolicyAdapter,
    OutcomeRecord,
    RiskProfile,
    WorkItem,
    generate_generic_candidates,
    generic_candidate_to_policy_candidate,
    generic_decision_guidance_support,
)
from spice.decision.general.candidates import (
    GenericExecutionIntent,
    is_approval_eligible_executable_candidate,
)
from spice.decision.general.types import Intent


class GenericPolicyAdapterTests(unittest.TestCase):
    def test_generic_candidate_maps_to_policy_candidate_without_selection(self) -> None:
        candidate = _candidate(
            "candidate.item_triage.work_1",
            "item.triage",
            target_refs=["work.1"],
            risk_level="low",
            reversibility="high",
            availability_status="available",
        )

        policy_candidate = generic_candidate_to_policy_candidate(candidate)

        self.assertEqual(policy_candidate.id, candidate.candidate_id)
        self.assertEqual(policy_candidate.action, "item.triage")
        self.assertIn("outcome_value", policy_candidate.score_breakdown)
        self.assertIn("risk_reduction", policy_candidate.score_breakdown)
        self.assertIn("urgency_alignment", policy_candidate.score_breakdown)
        self.assertIn("effort_fit", policy_candidate.score_breakdown)
        self.assertIn("impact_potential", policy_candidate.score_breakdown)
        self.assertIn("historical_outcome_alignment", policy_candidate.score_breakdown)
        self.assertIn("execution_intent_fit", policy_candidate.score_breakdown)
        self.assertIn("preference_alignment", policy_candidate.score_breakdown)
        self.assertEqual(
            policy_candidate.params["constraint_checks"]["no_declared_veto_violation"],
            "pass",
        )
        self.assertNotIn("selected_candidate", policy_candidate.params)
        self.assertNotIn("recommendation", policy_candidate.params)

    def test_adapter_selects_with_guidance_and_builds_checkpoint(self) -> None:
        adapter = GenericPolicyAdapter(_guidance())
        blocked = _candidate(
            "candidate.intent_execute.intent_1",
            "intent.execute",
            availability_status="blocked",
            constraints_triggered=[
                {
                    "constraint_id": "constraint.requires_approval",
                    "severity": "veto",
                    "description": "Approval is required before side effects.",
                }
            ],
            why_blocked=["Approval is required before side effects."],
        )
        selected = _candidate(
            "candidate.item_triage.work_1",
            "item.triage",
            risk_level="low",
            reversibility="high",
            requires_confirmation=False,
        )

        result = adapter.evaluate(
            GeneralDecisionState(state_id="world.general"),
            candidates=[blocked, selected],
            decision_id="decision.general.test",
            trace_ref="trace.general.test",
        )

        self.assertEqual(result.checkpoint.decision_id, "decision.general.test")
        self.assertEqual(result.checkpoint.trace_ref, "trace.general.test")
        self.assertEqual(result.checkpoint.selected_candidate_id, selected.candidate_id)
        self.assertEqual(result.checkpoint.status, "recommended")
        self.assertIsNone(result.checkpoint.approval)
        self.assertEqual(
            {item.status for item in result.checkpoint.candidate_refs},
            {"blocked", "selected"},
        )
        self.assertEqual(
            result.compare_payload["selected_recommendation"]["candidate_id"],
            selected.candidate_id,
        )
        self.assertEqual(
            result.compare_payload["why_not_the_others"][0]["candidate_id"],
            blocked.candidate_id,
        )

    def test_compare_payload_prefers_user_facing_candidate_language(self) -> None:
        adapter = GenericPolicyAdapter(_guidance())
        candidate = _candidate(
            "candidate.intent_execute.intent_1",
            "intent.execute",
            risk_level="low",
            reversibility="high",
            requires_confirmation=True,
        )
        candidate.metadata.update(
            {
                "user_facing_title": "Fix the failing test before the meeting",
                "recommended_action": "Ask the executor to inspect the failure and make the smallest safe patch.",
                "why_now": ["The failing test blocks CI."],
                "expected_result": "The focused test passes.",
                "executor_task": "Fix the failing test and do not touch unrelated files.",
            }
        )

        result = adapter.evaluate(
            GeneralDecisionState(state_id="world.general"),
            candidates=[candidate],
            decision_id="decision.general.test",
            trace_ref="trace.general.test",
        )

        compare_candidate = result.compare_payload["candidate_decisions"][0]
        selected = result.compare_payload["selected_recommendation"]

        self.assertEqual(compare_candidate["title"], "Fix the failing test before the meeting")
        self.assertEqual(
            compare_candidate["recommended_action"],
            "Ask the executor to inspect the failure and make the smallest safe patch.",
        )
        self.assertEqual(compare_candidate["why_now"], ["The failing test blocks CI."])
        self.assertEqual(compare_candidate["expected_result"], "The focused test passes.")
        self.assertEqual(
            compare_candidate["executor_task"],
            "Fix the failing test and do not touch unrelated files.",
        )
        self.assertEqual(selected["title"], "Fix the failing test before the meeting")
        self.assertEqual(
            selected["human_summary"],
            "Ask the executor to inspect the failure and make the smallest safe patch.",
        )

    def test_compare_payload_localizes_user_facing_explanations_for_chinese_intent(self) -> None:
        adapter = GenericPolicyAdapter(_guidance())
        selected = _candidate(
            "candidate.intent_execute.intent_1",
            "intent.execute",
            intent="在 .spice-smoke/test.txt 中写入 OK。请勿修改其他文件。",
            risk_level="low",
            reversibility="high",
            requires_confirmation=True,
            execution_boundary=ExecutionBoundary(
                mode="execution_intent",
                target="intent.1",
                requires_confirmation=True,
                side_effect_class="external_effect",
            ),
        )
        alternative = _candidate(
            "candidate.context_prepare.intent_1",
            "context.prepare",
            intent="先准备上下文。",
            risk_level="low",
            reversibility="high",
            requires_confirmation=False,
        )

        result = adapter.evaluate(
            GeneralDecisionState(state_id="world.general"),
            candidates=[selected, alternative],
            decision_id="decision.general.test",
            trace_ref="trace.general.test",
            run_intent_mode="act",
            selection_candidate_ids={selected.candidate_id},
        )

        payload = result.compare_payload
        self.assertEqual(payload["display_language"], "zh")
        basis = payload["selected_recommendation"]["decision_basis"]
        self.assertIn("结果价值", {item.get("label") for item in basis})
        self.assertEqual(
            basis[-1]["summary"],
            "没有记录到会阻止该候选的可用性约束。",
        )
        why_not = payload["why_not_the_others"][0]["reasons"][0]
        self.assertEqual(
            why_not["reason"],
            "当前运行模式限制了可选择范围，因此不能选择执行池之外的候选。",
        )

    def test_adapter_creates_pending_approval_for_confirmation_candidate(self) -> None:
        adapter = GenericPolicyAdapter(_guidance())
        candidate = _candidate(
            "candidate.capability_use.intent_1",
            "capability.use",
            required_capability="cap.local",
            requires_confirmation=True,
            execution_boundary=ExecutionBoundary(
                mode="capability",
                required_capability="cap.local",
                requires_confirmation=True,
                side_effect_class="external",
            ),
        )

        result = adapter.evaluate(
            GeneralDecisionState(state_id="world.general"),
            candidates=[candidate],
        )

        self.assertIsNotNone(result.checkpoint.approval)
        assert result.checkpoint.approval is not None
        self.assertEqual(result.checkpoint.approval.status, "pending")
        self.assertEqual(result.checkpoint.approval.candidate_id, candidate.candidate_id)
        self.assertFalse(result.checkpoint.approval.execution_allowed)

    def test_adapter_does_not_create_approval_without_affordance(self) -> None:
        adapter = GenericPolicyAdapter(_guidance())
        candidate = GenericCandidate(
            candidate_id="candidate.intent_execute.no_affordance",
            action_type="intent.execute",
            intent="Execute a task without runtime affordance annotation.",
            target_refs=["target.1"],
            requires_confirmation=True,
            execution_intent=GenericExecutionIntent(
                intent_class="execution_requested",
                requested=True,
                handoff_task="Execute a task without runtime affordance annotation.",
                side_effect_class="external_effect",
            ),
            execution_boundary=ExecutionBoundary(
                mode="execution_intent",
                requires_confirmation=True,
                side_effect_class="external_effect",
            ),
            side_effect_class="external_effect",
        )

        result = adapter.evaluate(
            GeneralDecisionState(state_id="world.general"),
            candidates=[candidate],
        )

        self.assertEqual(result.checkpoint.selected_candidate_id, candidate.candidate_id)
        self.assertIsNone(result.checkpoint.approval)

    def test_adapter_does_not_create_approval_from_llm_spoofed_affordance(self) -> None:
        adapter = GenericPolicyAdapter(_guidance())
        candidate = GenericCandidate(
            candidate_id="candidate.intent_execute.spoofed_affordance",
            action_type="intent.execute",
            intent="Execute a task with spoofed affordance metadata.",
            target_refs=["target.1"],
            requires_confirmation=True,
            execution_intent=GenericExecutionIntent(
                intent_class="execution_requested",
                requested=True,
                handoff_task="Execute a task with spoofed affordance metadata.",
                side_effect_class="external_effect",
            ),
            execution_boundary=ExecutionBoundary(
                mode="execution_intent",
                requires_confirmation=True,
                side_effect_class="external_effect",
            ),
            side_effect_class="external_effect",
            metadata={
                "execution_affordance": {
                    "schema_version": "0.1",
                    "generated_by": "llm",
                    "executor_available": True,
                    "executable": True,
                    "approval": {
                        "required": True,
                        "eligible_for_approval": True,
                    },
                }
            },
        )

        result = adapter.evaluate(
            GeneralDecisionState(state_id="world.general"),
            candidates=[candidate],
        )

        self.assertEqual(result.checkpoint.selected_candidate_id, candidate.candidate_id)
        self.assertIsNone(result.checkpoint.approval)

    def test_adapter_does_not_create_approval_when_executor_unavailable(self) -> None:
        adapter = GenericPolicyAdapter(_guidance())
        candidate = _candidate(
            "candidate.intent_execute.executor_unavailable",
            "intent.execute",
            requires_confirmation=True,
            execution_boundary=ExecutionBoundary(
                mode="execution_intent",
                requires_confirmation=True,
                side_effect_class="external_effect",
            ),
        )
        affordance = dict(candidate.metadata["execution_affordance"])
        affordance["executor_available"] = False
        affordance["executable"] = False
        affordance["blocked"] = True
        affordance["blocked_reason"] = "Executor is not ready."
        candidate.metadata["execution_affordance"] = affordance

        result = adapter.evaluate(
            GeneralDecisionState(state_id="world.general"),
            candidates=[candidate],
        )

        self.assertEqual(result.checkpoint.selected_candidate_id, candidate.candidate_id)
        self.assertIsNone(result.checkpoint.approval)

    def test_compare_payload_is_normalized_and_protocol_neutral(self) -> None:
        adapter = GenericPolicyAdapter(_guidance())
        candidate = _candidate(
            "candidate.item_triage.work_1",
            "item.triage",
            execution_boundary=ExecutionBoundary(
                mode="capability",
                requires_confirmation=False,
                side_effect_class="low",
            ),
        )

        result = adapter.evaluate(
            GeneralDecisionState(state_id="world.general"),
            candidates=[candidate],
            decision_id="decision.general.test",
            trace_ref="trace.general.test",
        )

        payload = result.compare_payload
        encoded = json.dumps(payload)
        self.assertEqual(payload["decision_id"], "decision.general.test")
        self.assertEqual(payload["trace_ref"], "trace.general.test")
        self.assertIn(candidate.candidate_id, payload["score_breakdown"]["candidates"])
        self.assertEqual(payload["execution_boundary"]["executor"], "")
        self.assertNotIn("sdep", encoded.lower())
        self.assertNotIn("hermes", encoded.lower())
        self.assertNotIn("codex", encoded.lower())
        rendered = render_compare_text(payload, use_bars=False)
        self.assertIn("DECISION COMPARISON", rendered)
        self.assertIn("WHY THIS WON", rendered)
        self.assertIn("WHY NOT OTHERS", rendered)

    def test_adapter_can_generate_candidates_from_general_state(self) -> None:
        adapter = GenericPolicyAdapter(_guidance())
        state = GeneralDecisionState(
            state_id="world.general",
            intents=[
                Intent(
                    intent_id="intent.1",
                    summary="Handle an incoming request.",
                    target_refs=["target.1"],
                )
            ],
            capabilities=[
                Capability(
                    capability_id="cap.local",
                    provider="local",
                    scope="general",
                    requires_confirmation=False,
                )
            ],
            work_items=[
                WorkItem(
                    work_item_id="work.1",
                    title="Clarify an incoming item",
                )
            ],
        )
        before = state.to_payload()

        result = adapter.evaluate(state)

        self.assertTrue(result.candidates)
        self.assertEqual(
            [candidate.candidate_id for candidate in result.candidates],
            [candidate.candidate_id for candidate in generate_generic_candidates(state)],
        )
        self.assertIn(
            result.checkpoint.selected_candidate_id,
            {candidate.candidate_id for candidate in result.candidates},
        )
        self.assertEqual(state.to_payload(), before)

    def test_adapter_refuses_empty_candidate_set(self) -> None:
        adapter = GenericPolicyAdapter(_guidance())

        with self.assertRaisesRegex(ValueError, "at least one candidate"):
            adapter.evaluate(GeneralDecisionState(state_id="world.general"))

        state = GeneralDecisionState(
            state_id="world.general",
            intents=[
                Intent(
                    intent_id="intent.1",
                    summary="Handle an incoming request.",
                )
            ],
        )
        self.assertTrue(generate_generic_candidates(state))
        with self.assertRaisesRegex(ValueError, "at least one candidate"):
            adapter.evaluate(state, candidates=[])

    def test_adapter_refuses_selected_blocked_candidate(self) -> None:
        adapter = GenericPolicyAdapter(_guidance_without_hard_constraints())
        blocked = _candidate(
            "candidate.intent_execute.intent_1",
            "intent.execute",
            availability_status="blocked",
            constraints_triggered=[
                {
                    "constraint_id": "constraint.blocked",
                    "severity": "veto",
                    "description": "Blocked by availability constraint.",
                }
            ],
            why_blocked=["Blocked by availability constraint."],
        )

        with self.assertRaisesRegex(ValueError, "selected a blocked candidate"):
            adapter.evaluate(
                GeneralDecisionState(state_id="world.general"),
                candidates=[blocked],
            )

    def test_result_payload_is_json_serializable(self) -> None:
        adapter = GenericPolicyAdapter(_guidance())
        candidate = _candidate(
            "candidate.item_triage.work_1",
            "item.triage",
            risk_level="low",
            reversibility="high",
        )

        result = adapter.evaluate(
            GeneralDecisionState(state_id="world.general"),
            candidates=[candidate],
            decision_id="decision.general.test",
            trace_ref="trace.general.test",
        )
        decoded = json.loads(json.dumps(result.to_payload()))

        self.assertEqual(decoded["checkpoint"]["decision_id"], "decision.general.test")
        self.assertEqual(decoded["checkpoint"]["trace_ref"], "trace.general.test")
        self.assertEqual(
            decoded["checkpoint"]["selected_candidate_id"],
            candidate.candidate_id,
        )
        self.assertEqual(
            decoded["compare_payload"]["selected_recommendation"]["candidate_id"],
            candidate.candidate_id,
        )

    def test_support_contract_is_generic(self) -> None:
        support = generic_decision_guidance_support().to_payload()

        self.assertEqual(
            set(support["score_dimensions"]),
            {
                "outcome_value",
                "risk_reduction",
                "reversibility",
                "confidence_alignment",
                "urgency_alignment",
                "effort_fit",
                "impact_potential",
                "historical_outcome_alignment",
                "execution_intent_fit",
                "preference_alignment",
            },
        )
        self.assertEqual(
            support["constraint_ids"],
            ["no_declared_veto_violation", "selection_pool_eligible"],
        )
        self.assertNotIn("github", json.dumps(support).lower())
        self.assertNotIn("executor", json.dumps(support).lower())

    def test_score_breakdown_uses_state_effort_impact_and_preferences(self) -> None:
        state = GeneralDecisionState(
            state_id="world.general",
            work_items=[
                WorkItem(
                    work_item_id="work.urgent",
                    title="Fix blocked release",
                    urgency="high",
                    estimate_minutes=10,
                ),
                WorkItem(
                    work_item_id="work.low",
                    title="Tidy backlog",
                    urgency="low",
                    estimate_minutes=120,
                ),
            ],
            intents=[
                Intent(
                    intent_id="intent.release",
                    summary="Unblock the release.",
                    target_refs=["work.urgent"],
                    urgency="high",
                )
            ],
        )
        urgent = _candidate(
            "candidate.item_triage.work_urgent",
            "item.triage",
            target_refs=["work.urgent"],
            time_minutes=10,
            expected_summary=(
                "Resolve the active blocker, update the release state, "
                "and close the immediate follow-up loop."
            ),
            expected_updates=["work.urgent", "intent.release", "release.state"],
        )
        low = _candidate(
            "candidate.item_triage.work_low",
            "item.triage",
            target_refs=["work.low"],
            time_minutes=120,
            expected_summary="Record a small backlog update.",
            expected_updates=["work.low"],
        )

        urgent_policy = generic_candidate_to_policy_candidate(
            urgent,
            state=state,
            guidance=_guidance(),
        )
        low_policy = generic_candidate_to_policy_candidate(
            low,
            state=state,
            guidance=_guidance(),
        )

        self.assertGreater(
            urgent_policy.score_breakdown["urgency_alignment"],
            low_policy.score_breakdown["urgency_alignment"],
        )
        self.assertGreater(
            urgent_policy.score_breakdown["effort_fit"],
            low_policy.score_breakdown["effort_fit"],
        )
        self.assertGreater(
            urgent_policy.score_breakdown["impact_potential"],
            low_policy.score_breakdown["impact_potential"],
        )
        self.assertGreater(
            urgent_policy.score_breakdown["preference_alignment"],
            low_policy.score_breakdown["preference_alignment"],
        )

    def test_historical_outcome_alignment_is_neutral_without_history(self) -> None:
        candidate = _candidate(
            "candidate.item_triage.work_1",
            "item.triage",
        )

        policy_candidate = generic_candidate_to_policy_candidate(
            candidate,
            state=GeneralDecisionState(state_id="world.general"),
            guidance=_guidance(),
        )

        self.assertEqual(
            policy_candidate.score_breakdown["historical_outcome_alignment"],
            0.50,
        )

    def test_act_mode_execution_intent_fit_prefers_approval_eligible_execution(self) -> None:
        executable = _candidate(
            "candidate.intent_execute.intent_1",
            "intent.execute",
            requires_confirmation=True,
            execution_boundary=ExecutionBoundary(
                mode="execution_intent",
                requires_confirmation=True,
                side_effect_class="external",
            ),
        )
        planning = _candidate(
            "candidate.task_split.intent_1",
            "task.split",
            risk_level="low",
            reversibility="high",
        )

        executable_policy = generic_candidate_to_policy_candidate(
            executable,
            guidance=_guidance(),
            run_intent_mode="act",
        )
        planning_policy = generic_candidate_to_policy_candidate(
            planning,
            guidance=_guidance(),
            run_intent_mode="act",
        )

        self.assertEqual(
            executable_policy.score_breakdown["execution_intent_fit"],
            1.0,
        )
        self.assertLess(
            planning_policy.score_breakdown["execution_intent_fit"],
            executable_policy.score_breakdown["execution_intent_fit"],
        )

    def test_unanchored_execute_candidate_is_not_approval_eligible(self) -> None:
        candidate = GenericCandidate(
            candidate_id="candidate.intent_execute.unanchored",
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
        )

        self.assertFalse(is_approval_eligible_executable_candidate(candidate))

        policy_candidate = generic_candidate_to_policy_candidate(
            candidate,
            guidance=_guidance(),
            run_intent_mode="act",
        )
        self.assertEqual(policy_candidate.score_breakdown["execution_intent_fit"], 0.55)

    def test_act_mode_policy_can_select_executable_over_planning(self) -> None:
        executable = _candidate(
            "candidate.intent_execute.intent_1",
            "intent.execute",
            requires_confirmation=True,
            execution_boundary=ExecutionBoundary(
                mode="execution_intent",
                requires_confirmation=True,
                side_effect_class="external",
            ),
        )
        planning = _candidate(
            "candidate.task_split.intent_1",
            "task.split",
            risk_level="low",
            reversibility="high",
        )

        result = GenericPolicyAdapter(_guidance()).evaluate(
            GeneralDecisionState(state_id="world.general"),
            candidates=[planning, executable],
            decision_id="decision.act",
            trace_ref="trace.act",
            run_intent_mode="act",
            selection_candidate_ids={executable.candidate_id},
        )

        self.assertEqual(result.checkpoint.selected_candidate_id, executable.candidate_id)
        self.assertIsNotNone(result.checkpoint.approval)

    def test_selection_pool_keeps_non_executable_candidates_visible(self) -> None:
        executable = _candidate(
            "candidate.intent_execute.intent_1",
            "intent.execute",
            requires_confirmation=True,
            execution_boundary=ExecutionBoundary(
                mode="execution_intent",
                requires_confirmation=True,
                side_effect_class="external",
            ),
        )
        planning = _candidate(
            "candidate.task_split.intent_1",
            "task.split",
            risk_level="low",
            reversibility="high",
        )

        result = GenericPolicyAdapter(_guidance()).evaluate(
            GeneralDecisionState(state_id="world.general"),
            candidates=[planning, executable],
            decision_id="decision.act",
            trace_ref="trace.act",
            run_intent_mode="act",
            selection_candidate_ids={executable.candidate_id},
        )

        visible_ids = {
            item["candidate_id"] for item in result.compare_payload["candidate_decisions"]
        }
        self.assertEqual(visible_ids, {planning.candidate_id, executable.candidate_id})
        self.assertEqual(result.checkpoint.selected_candidate_id, executable.candidate_id)
        why_not = result.compare_payload["why_not_the_others"][0]
        self.assertEqual(why_not["candidate_id"], planning.candidate_id)
        self.assertEqual(why_not["reasons"][0]["constraint_id"], "selection_pool_eligible")

    def test_compare_payload_orders_decision_candidates_before_runtime_guardrails(self) -> None:
        runtime_guardrail = _candidate(
            "candidate.context_prepare.intent_1",
            "context.prepare",
            risk_level="low",
            reversibility="high",
        )
        selected_decision = _candidate(
            "candidate.llm.decision.state_context",
            "item.triage",
            intent="Prioritize state-as-context.",
            risk_level="low",
            reversibility="high",
        )
        selected_decision.candidate_kind = "decision"
        selected_decision.metadata.update(
            {
                "candidate_kind": "decision",
                "candidate_source": "llm_generator",
                "user_facing_title": "Improve state-as-context",
            }
        )
        other_decision = _candidate(
            "candidate.llm.decision.perception",
            "item.triage",
            intent="Prioritize proactive perception.",
            risk_level="medium",
        )
        other_decision.candidate_kind = "decision"
        other_decision.metadata.update(
            {
                "candidate_kind": "decision",
                "candidate_source": "llm_generator",
                "user_facing_title": "Improve proactive perception",
            }
        )

        result = GenericPolicyAdapter(_guidance()).evaluate(
            GeneralDecisionState(state_id="world.general"),
            candidates=[runtime_guardrail, selected_decision, other_decision],
            decision_id="decision.layered",
            trace_ref="trace.layered",
            selection_candidate_ids={
                selected_decision.candidate_id,
                other_decision.candidate_id,
            },
        )

        ordered_ids = [
            item["candidate_id"] for item in result.compare_payload["candidate_decisions"]
        ]
        self.assertEqual(
            ordered_ids,
            [
                selected_decision.candidate_id,
                other_decision.candidate_id,
                runtime_guardrail.candidate_id,
            ],
        )
        self.assertEqual(
            result.compare_payload["why_not_the_others"][0]["candidate_id"],
            other_decision.candidate_id,
        )

    def test_historical_outcome_alignment_uses_same_action_type_outcomes(self) -> None:
        state = GeneralDecisionState(
            state_id="world.general",
            outcomes=[
                _outcome("outcome.success.1", "item.triage", "success"),
                _outcome("outcome.success.2", "item.triage", "success"),
                _outcome("outcome.fail.1", "time.defer", "failed"),
            ],
        )
        triage = _candidate("candidate.item_triage.work_1", "item.triage")
        defer = _candidate("candidate.time_defer.work_1", "time.defer")

        triage_policy = generic_candidate_to_policy_candidate(
            triage,
            state=state,
            guidance=_guidance(),
        )
        defer_policy = generic_candidate_to_policy_candidate(
            defer,
            state=state,
            guidance=_guidance(),
        )

        self.assertGreater(
            triage_policy.score_breakdown["historical_outcome_alignment"],
            0.90,
        )
        self.assertLess(
            defer_policy.score_breakdown["historical_outcome_alignment"],
            0.10,
        )

    def test_historical_outcome_alignment_uses_latest_ten_outcomes(self) -> None:
        state = GeneralDecisionState(
            state_id="world.general",
            outcomes=[
                _outcome(f"outcome.old.{index}", "item.triage", "failed")
                for index in range(2)
            ]
            + [
                _outcome(f"outcome.recent.{index}", "item.triage", "success")
                for index in range(10)
            ],
        )
        candidate = _candidate("candidate.item_triage.work_1", "item.triage")

        result = GenericPolicyAdapter(_guidance()).evaluate(
            state,
            candidates=[candidate],
            decision_id="decision.history",
            trace_ref="trace.history",
        )

        history = result.compare_payload["candidate_decisions"][0]["history"]
        self.assertEqual(history["similar_outcome_count"], 10)
        self.assertEqual(history["success_count"], 10)
        self.assertEqual(history["failure_count"], 0)
        self.assertEqual(
            result.policy_candidates[0].score_breakdown["historical_outcome_alignment"],
            1.0,
        )

    def test_compare_payload_includes_historical_outcome_evidence(self) -> None:
        state = GeneralDecisionState(
            state_id="world.general",
            outcomes=[
                OutcomeRecord(
                    outcome_id="outcome.from.candidate.id",
                    candidate_id="candidate.item_triage.previous",
                    task_status="success",
                    status="observed",
                    summary="Previous triage succeeded.",
                    metadata={},
                )
            ],
        )
        candidate = _candidate("candidate.item_triage.work_1", "item.triage")

        result = GenericPolicyAdapter(_guidance()).evaluate(
            state,
            candidates=[candidate],
            decision_id="decision.history",
            trace_ref="trace.history",
        )

        history = result.compare_payload["candidate_decisions"][0]["history"]
        self.assertEqual(history["similar_outcome_count"], 1)
        self.assertEqual(history["success_count"], 1)
        rendered = render_compare_text(result.compare_payload, use_bars=False)
        self.assertIn("history:", rendered)
        self.assertIn("1/1 success", rendered)


def _guidance() -> DecisionGuidance:
    return DecisionGuidance(
        source_path="test://general-decision.md",
        source_hash="guidance.test.general",
        artifact_id="decision.test.general",
        schema_version="0.1",
        artifact_version="0.1.0",
        status="test",
        primary_objective=PrimaryObjectiveGuidance(
            text="Select the best available generic action.",
            direction="maximize",
        ),
        weights={
            "outcome_value": 0.20,
            "risk_reduction": 0.15,
            "reversibility": 0.10,
            "confidence_alignment": 0.10,
            "urgency_alignment": 0.15,
            "effort_fit": 0.10,
            "impact_potential": 0.10,
            "historical_outcome_alignment": 0.03,
            "execution_intent_fit": 0.10,
            "preference_alignment": 0.02,
        },
        hard_constraints=[
            HardConstraintGuidance(
                id="no_declared_veto_violation",
                rule="do not select candidates blocked by declared availability constraints",
                severity="veto",
            ),
            HardConstraintGuidance(
                id="selection_pool_eligible",
                rule="do not select candidates outside the active runtime selection pool",
                severity="veto",
            ),
        ],
    )


def _guidance_without_hard_constraints() -> DecisionGuidance:
    return DecisionGuidance(
        source_path="test://general-decision.md",
        source_hash="guidance.test.general.no_constraints",
        artifact_id="decision.test.general",
        schema_version="0.1",
        artifact_version="0.1.0",
        status="test",
        primary_objective=PrimaryObjectiveGuidance(
            text="Select the best available generic action.",
            direction="maximize",
        ),
        weights={
            "outcome_value": 0.20,
            "risk_reduction": 0.15,
            "reversibility": 0.10,
            "confidence_alignment": 0.10,
            "urgency_alignment": 0.15,
            "effort_fit": 0.10,
            "impact_potential": 0.10,
            "historical_outcome_alignment": 0.03,
            "execution_intent_fit": 0.10,
            "preference_alignment": 0.02,
        },
    )


def _outcome(outcome_id: str, action_type: str, task_status: str) -> OutcomeRecord:
    return OutcomeRecord(
        outcome_id=outcome_id,
        candidate_id=f"candidate.{action_type.replace('.', '_')}.{outcome_id}",
        task_status=task_status,
        status="observed",
        summary=f"{action_type} ended with {task_status}.",
        metadata={"action_type": action_type},
    )


def _candidate(
    candidate_id: str,
    action_type: str,
    *,
    intent: str | None = None,
    target_refs: list[str] | None = None,
    required_capability: str = "",
    risk_level: str = "medium",
    reversibility: str = "medium",
    availability_status: str = "available",
    requires_confirmation: bool = False,
    constraints_triggered: list[dict[str, str]] | None = None,
    why_blocked: list[str] | None = None,
    execution_boundary: ExecutionBoundary | None = None,
    time_minutes: int | None = 5,
    expected_summary: str = "Expected generic state update.",
    expected_updates: list[str] | None = None,
) -> GenericCandidate:
    return GenericCandidate(
        candidate_id=candidate_id,
        action_type=action_type,
        intent=intent or f"Test {action_type}.",
        execution_intent=GenericExecutionIntent(
            intent_class="execution_requested",
            requested=True,
            handoff_task=intent or f"Test {action_type}.",
            side_effect_class="external_effect",
        )
        if action_type in {"intent.execute", "capability.use"}
        else GenericExecutionIntent(),
        target_refs=list(target_refs or ["target.1"]),
        required_capability=required_capability,
        estimated_cost=EstimatedCost(time_minutes=time_minutes, attention="low"),
        risk_profile=RiskProfile(level=risk_level, uncertainty="low"),
        reversibility=reversibility,
        requires_confirmation=requires_confirmation,
        expected_state_delta=ExpectedStateDelta(
            updates_refs=list(
                expected_updates
                if expected_updates is not None
                else (target_refs or ["target.1"])
            ),
            summary=expected_summary,
        ),
        execution_boundary=execution_boundary or ExecutionBoundary(
            mode="none",
            requires_confirmation=requires_confirmation,
        ),
        constraints_triggered=list(constraints_triggered or []),
        why_available=["Candidate is available for test."],
        why_blocked=list(why_blocked or []),
        side_effect_class="none",
        availability_status=availability_status,
        metadata=_execution_affordance_metadata(
            action_type=action_type,
            requires_confirmation=requires_confirmation,
            required_capability=required_capability,
        ),
    )


def _execution_affordance_metadata(
    *,
    action_type: str,
    requires_confirmation: bool,
    required_capability: str,
) -> dict[str, object]:
    if action_type == "capability.use" and not required_capability:
        return {}
    if action_type not in {"intent.execute", "capability.use"} or not requires_confirmation:
        return {}
    return {
        "execution_affordance": {
            "schema_version": "0.1",
            "generated_by": "spice.runtime.execution_affordance",
            "candidate_executable": True,
            "executor_available": True,
            "executable": True,
            "blocked": False,
            "blocked_reason": "",
            "executor": {
                "executor_id": "test",
                "status": "ready",
            },
            "approval": {
                "required": True,
                "eligible_for_approval": True,
            },
            "permission": {
                "required": "workspace_write",
                "reason": "Test execution affordance.",
                "source": "test_fixture",
                "side_effect_class": "external_effect",
            },
        }
    }


if __name__ == "__main__":
    unittest.main()
