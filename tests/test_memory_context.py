from __future__ import annotations

import unittest

from spice.decision.general.approval import Approval
from spice.decision.general.state import GeneralDecisionState
from spice.decision.general.trace import DecisionCheckpoint
from spice.decision.general.types import Capability, Constraint, Intent, OutcomeRecord, Risk, Signal, WorkItem
from spice.memory import DecisionContext, ReflectionContext, SimulationContext
from spice.memory.base import ContextCompiler, MemoryProvider
from spice.memory.deterministic import DeterministicContextCompiler
from spice.protocols import WorldState


class StubMemoryProvider(MemoryProvider):
    def __init__(self) -> None:
        self.records_by_namespace = {
            "general.decision": [{"id": "memory.decision.1", "summary": "Prior choice"}],
            "general.preference": [{"id": "memory.preference.1", "summary": "Prefer low risk"}],
            "general.simulation": [{"id": "memory.simulation.1", "summary": "Similar sim"}],
            "general.reflection": [{"id": "memory.reflection.1", "summary": "Similar lesson"}],
            "general.session_summary": [
                {
                    "id": "memory.summary.1",
                    "summary_type": "deterministic",
                    "current_goal": {"text": "Pick next release task"},
                    "active_decision": {"decision_id": "decision.001"},
                    "recent_decisions": [{"decision_id": "decision.001"}],
                    "execution_outcomes": [],
                    "open_threads": [],
                    "markdown": "# Session Summary\n",
                }
            ],
        }
        self.queries: list[str] = []
        self.writes: list[tuple[str, list[dict], list[str] | None]] = []

    def write(
        self,
        records: list[dict],
        *,
        namespace: str,
        refs: list[str] | None = None,
    ) -> list[str]:
        self.writes.append((namespace, records, refs))
        return [str(record.get("id", f"{namespace}.written")) for record in records]

    def query(
        self,
        *,
        namespace: str,
        filters: dict | None = None,
        limit: int = 20,
        order_by: str | None = None,
    ) -> list[dict]:
        self.queries.append(namespace)
        records = list(self.records_by_namespace.get(namespace, []))
        return records if limit < 0 else records[:limit]


class MemoryContextSchemaTests(unittest.TestCase):
    def test_decision_context_has_general_runtime_fields(self) -> None:
        context = DecisionContext.create(
            world_state_id="worldstate.local",
            domain="general",
            current_intent={"text": "Pick the next release task."},
            active_decision_frame={"decision_id": "decision.001"},
            recent_decisions=[{"decision_id": "decision.000"}],
            recent_approvals=[{"approval_id": "approval.001", "status": "pending"}],
            executor_affordance={"executor": "codex", "available": True},
            session_summary={"session_id": "session.default", "runs": 2},
            workspace_context={"root": ".", "memory_provider": "file"},
        )

        self.assertEqual(context.context_type, "decision")
        self.assertEqual(context.current_intent["text"], "Pick the next release task.")
        self.assertEqual(context.active_decision_frame["decision_id"], "decision.001")
        self.assertEqual(context.recent_decisions[0]["decision_id"], "decision.000")
        self.assertEqual(context.recent_approvals[0]["status"], "pending")
        self.assertTrue(context.executor_affordance["available"])
        self.assertEqual(context.session_summary["runs"], 2)
        self.assertEqual(context.workspace_context["memory_provider"], "file")

    def test_simulation_context_has_general_runtime_fields(self) -> None:
        context = SimulationContext.create(
            world_state_id="worldstate.local",
            domain="general",
            current_intent={"text": "Simulate option B."},
            active_decision_frame={"selected_candidate_id": "candidate.b"},
            recent_decisions=[{"decision_id": "decision.001"}],
            recent_approvals=[{"approval_id": "approval.001"}],
            executor_affordance={"permission": "workspace_write"},
            session_summary={"session_id": "session.default"},
            workspace_context={"executor": "codex"},
        )

        self.assertEqual(context.context_type, "simulation")
        self.assertEqual(context.current_intent["text"], "Simulate option B.")
        self.assertEqual(context.active_decision_frame["selected_candidate_id"], "candidate.b")
        self.assertEqual(context.recent_decisions[0]["decision_id"], "decision.001")
        self.assertEqual(context.recent_approvals[0]["approval_id"], "approval.001")
        self.assertEqual(context.executor_affordance["permission"], "workspace_write")
        self.assertEqual(context.session_summary["session_id"], "session.default")
        self.assertEqual(context.workspace_context["executor"], "codex")

    def test_reflection_context_has_general_runtime_fields_with_empty_defaults(self) -> None:
        context = ReflectionContext.create(
            world_state_id="worldstate.local",
            domain="general",
        )

        self.assertEqual(context.context_type, "reflection")
        self.assertEqual(context.current_intent, {})
        self.assertEqual(context.active_decision_frame, {})
        self.assertEqual(context.recent_decisions, [])
        self.assertEqual(context.recent_approvals, [])
        self.assertEqual(context.executor_affordance, {})
        self.assertEqual(context.session_summary, {})
        self.assertEqual(context.workspace_context, {})


class DeterministicGeneralContextCompilerTests(unittest.TestCase):
    def test_general_methods_do_not_break_existing_context_compiler_subclasses(self) -> None:
        class LegacyCompiler(ContextCompiler):
            def compile_decision_context(self, state, *, domain="generic", recent_history=None):
                return DecisionContext.create(world_state_id=state.id, domain=domain)

            def compile_simulation_context(
                self,
                state,
                *,
                domain="generic",
                candidate_decisions=None,
                candidate_intents=None,
                recent_history=None,
            ):
                return SimulationContext.create(world_state_id=state.id, domain=domain)

            def compile_reflection_context(
                self,
                state,
                outcome,
                *,
                domain="generic",
                decision=None,
                intent=None,
                execution_result=None,
                recent_history=None,
            ):
                return ReflectionContext.create(world_state_id=state.id, domain=domain)

            def write_reflection(self, reflection_record, *, domain="generic", provider=None):
                return []

        compiler = LegacyCompiler()

        with self.assertRaises(NotImplementedError):
            compiler.compile_general_decision_context(_world_state(), _general_state())

    def test_compile_general_decision_context_adds_runtime_fields(self) -> None:
        provider = StubMemoryProvider()
        compiler = DeterministicContextCompiler(memory_provider=provider)
        state = _world_state()
        general_state = _general_state()

        context = compiler.compile_general_decision_context(
            state,
            general_state,
            current_intent="Pick the next release task.",
            active_decision_frame=_active_frame(),
            session=_session(),
            config=_config(),
        )

        self.assertEqual(context.context_type, "decision")
        self.assertEqual(context.current_intent["text"], "Pick the next release task.")
        self.assertEqual(context.active_decision_frame["decision_id"], "decision.active")
        self.assertEqual(context.recent_decisions[0]["decision_id"], "decision.001")
        self.assertEqual(context.recent_approvals[0]["approval_id"], "approval.001")
        self.assertEqual(context.executor_affordance["executor"], "codex")
        self.assertEqual(context.session_summary["session_id"], "session.default")
        self.assertEqual(
            context.session_summary["rolling_summary"]["current_goal"]["text"],
            "Pick next release task",
        )
        self.assertEqual(context.session_summary["summary_text"], "# Session Summary\n")
        self.assertEqual(context.workspace_context["memory_provider"], "file")
        self.assertEqual(
            context.workspace_context["skill_catalog"]["executors"][0]["executor_id"],
            "spice.codex",
        )
        self.assertIn(
            "runtime.intent.execute",
            context.workspace_context["skill_catalog"]["executors"][0]["skill_ids"],
        )
        self.assertEqual(
            [record["id"] for record in context.retrieved_memory],
            ["memory.decision.1", "memory.preference.1", "memory.reflection.1"],
        )
        self.assertEqual(
            provider.queries,
            [
                "general.decision",
                "general.preference",
                "general.reflection",
                "general.session_summary",
            ],
        )
        self.assertEqual(
            context.metadata["general_state"]["work_items"][0]["work_item_id"],
            "work.general",
        )
        self.assertEqual(context.metadata["general_state"]["counts"]["capabilities"], 1)

    def test_compile_general_simulation_context_serializes_candidates(self) -> None:
        provider = StubMemoryProvider()
        compiler = DeterministicContextCompiler(memory_provider=provider)

        context = compiler.compile_general_simulation_context(
            _world_state(),
            _general_state(),
            current_intent={"text": "Simulate option B."},
            candidates=[
                {
                    "candidate_id": "candidate.b",
                    "title": "Polish Decision Card",
                    "expected_result": "Clearer card",
                }
            ],
            active_decision_frame=_active_frame(),
            session=_session(),
            config=_config(),
        )

        self.assertEqual(context.context_type, "simulation")
        self.assertEqual(context.current_intent["text"], "Simulate option B.")
        self.assertEqual(context.candidate_decisions[0]["candidate_id"], "candidate.b")
        self.assertEqual(context.executor_affordance["permission"], "workspace_write")
        self.assertEqual(
            [record["id"] for record in context.retrieved_memory],
            [
                "memory.simulation.1",
                "memory.decision.1",
                "memory.preference.1",
                "memory.reflection.1",
            ],
        )
        self.assertEqual(
            provider.queries[-4:],
            [
                "general.decision",
                "general.preference",
                "general.reflection",
                "general.session_summary",
            ],
        )
        self.assertEqual(
            provider.queries[:4],
            [
                "general.decision",
                "general.preference",
                "general.reflection",
                "general.session_summary",
            ],
        )
        self.assertEqual(
            provider.queries[4:],
            [
                "general.simulation",
                "general.decision",
                "general.preference",
                "general.reflection",
                "general.session_summary",
            ],
        )

    def test_compile_general_reflection_context_carries_execution_artifacts(self) -> None:
        provider = StubMemoryProvider()
        compiler = DeterministicContextCompiler(memory_provider=provider)

        context = compiler.compile_general_reflection_context(
            _world_state(),
            _general_state(),
            {
                "outcome_id": "outcome.001",
                "protocol_status": "success",
                "task_status": "success",
                "state_updated": True,
                "summary": "Created file",
            },
            current_intent="Create the smoke file.",
            decision_artifact={"decision_id": "decision.001"},
            execution_artifact={"execution_id": "exec.001", "executor": "codex"},
            active_decision_frame=_active_frame(),
            session=_session(),
            config=_config(),
        )

        self.assertEqual(context.context_type, "reflection")
        self.assertEqual(context.executed_path["execution"]["execution_id"], "exec.001")
        self.assertEqual(context.expected_vs_actual["actual"]["task_status"], "success")
        self.assertEqual(context.impact_summary["outcome_id"], "outcome.001")
        self.assertEqual(context.retrieved_lessons[0]["id"], "memory.reflection.1")
        self.assertEqual(
            provider.queries,
            [
                "general.reflection",
                "general.decision",
                "general.preference",
                "general.session_summary",
            ],
        )


def _world_state() -> WorldState:
    return WorldState(
        id="worldstate.local",
        refs=["state.ref"],
        goals=[{"id": "goal.release", "summary": "Ship next release"}],
        constraints=[{"id": "constraint.time", "summary": "Two weeks"}],
        resources={"executor": "codex"},
        risks=[{"id": "risk.scope", "summary": "Scope creep"}],
        signals=[{"id": "signal.stars", "summary": "500 stars, 10 DAU"}],
        active_intents=[{"id": "intent.current", "summary": "Pick next task"}],
        recent_outcomes=[{"id": "outcome.prev", "status": "success"}],
        confidence={"state": 0.8},
    )


def _general_state() -> GeneralDecisionState:
    approval = Approval(
        approval_id="approval.001",
        decision_id="decision.001",
        candidate_id="candidate.001",
        status="pending",
    )
    return GeneralDecisionState(
        state_id="worldstate.local",
        signals=[
            Signal(
                signal_id="signal.general",
                source="manual",
                kind="intent",
                summary="User asked what to build next.",
            )
        ],
        intents=[
            Intent(
                intent_id="intent.general",
                summary="Choose the highest impact release task.",
            )
        ],
        work_items=[
            WorkItem(
                work_item_id="work.general",
                title="Polish Decision Card",
                status="open",
                urgency="high",
            )
        ],
        capabilities=[
            Capability(
                capability_id="capability.executor.codex",
                provider="codex",
                scope="workspace",
                status="available",
                side_effects=["file_write"],
            )
        ],
        constraints=[
            Constraint(
                constraint_id="constraint.general",
                kind="time",
                description="Keep the release scoped.",
            )
        ],
        risks=[
            Risk(
                risk_id="risk.general",
                kind="execution",
                description="Candidate may be too broad.",
            )
        ],
        approvals=[approval],
        decision_checkpoints=[
            DecisionCheckpoint(
                decision_id="decision.001",
                trace_ref="trace.001",
                state_ref="state.ref",
                profile_ref="profile.ref",
                selected_candidate_id="candidate.001",
                recommendation="Polish the Decision Card",
                approval=approval,
            )
        ],
        outcomes=[
            OutcomeRecord(
                outcome_id="outcome.general",
                decision_id="decision.001",
                task_status="success",
            )
        ],
        trace_refs=["trace.001"],
    )


def _active_frame() -> dict:
    return {
        "decision_id": "decision.active",
        "selected_candidate_id": "candidate.active",
        "selected": {
            "candidate_id": "candidate.active",
            "title": "Add LLM retry",
            "expected_result": "Retry support lands.",
            "executor_affordance": {
                "executor": "codex",
                "permission": "workspace_write",
                "requires_approval": True,
            },
        },
    }


def _session() -> dict:
    return {
        "session_id": "session.default",
        "status": "active",
        "runs": 2,
        "decisions": 2,
        "pending_approvals": 1,
        "active_state_ref": ".spice/state/state.json#after:abc",
        "last_decision": {
            "run_id": "run.001",
            "decision_id": "decision.001",
            "trace_ref": "trace.001",
        },
    }


def _config() -> dict:
    return {
        "llm_provider": "mimo",
        "llm_model": "mimo-v2.5-pro",
        "llm_candidate_expand": True,
        "llm_simulation": True,
        "executor": "codex",
        "executor_permission": "workspace_write",
        "executor_transport": "sdep_subprocess_wrapper",
        "perception_provider": "manual",
        "memory_provider": "file",
        "memory_path": ".spice/memory",
        "context_compiler": "deterministic",
    }


if __name__ == "__main__":
    unittest.main()
