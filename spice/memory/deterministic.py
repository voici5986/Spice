from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from spice.memory.base import ContextCompiler, MemoryProvider
from spice.memory.context import DecisionContext, ReflectionContext, SimulationContext
from spice.protocols import Decision, ExecutionIntent, ExecutionResult, Outcome, ProtocolRecord, WorldState


class DeterministicContextCompiler(ContextCompiler):
    """Reference deterministic compiler for bounded stage-specific contexts."""

    def __init__(
        self,
        memory_provider: MemoryProvider | None = None,
        *,
        top_k_entities: int = 10,
        top_k_signals: int = 10,
        top_k_risks: int = 10,
        top_k_goals: int = 10,
        top_k_constraints: int = 10,
        top_k_active_intents: int = 10,
        recent_outcomes_limit: int = 10,
        memory_query_limit: int = 10,
        history_ref_limit: int = 20,
        candidate_limit: int = 10,
    ) -> None:
        self.memory_provider = memory_provider
        self.top_k_entities = top_k_entities
        self.top_k_signals = top_k_signals
        self.top_k_risks = top_k_risks
        self.top_k_goals = top_k_goals
        self.top_k_constraints = top_k_constraints
        self.top_k_active_intents = top_k_active_intents
        self.recent_outcomes_limit = recent_outcomes_limit
        self.memory_query_limit = memory_query_limit
        self.history_ref_limit = history_ref_limit
        self.candidate_limit = candidate_limit

    def compile_decision_context(
        self,
        state: WorldState,
        *,
        domain: str = "generic",
        recent_history: list[ProtocolRecord] | None = None,
    ) -> DecisionContext:
        recent_history = recent_history or []
        retrieved = self._query_memory(f"{domain}.decision")
        refs = self._build_refs(state, recent_history, [record.get("id") for record in retrieved])

        return DecisionContext.create(
            world_state_id=state.id,
            domain=domain,
            budget=self._base_budget(),
            confidence={"state": dict(state.confidence), "method": "deterministic"},
            provenance=self._base_provenance(domain, refs, retrieved),
            refs=refs,
            objectives=self._tail(state.goals, self.top_k_goals),
            constraints=self._tail(state.constraints, self.top_k_constraints),
            entities=self._slice_entities(state.entities, self.top_k_entities),
            signals=self._tail(state.signals, self.top_k_signals),
            risks=self._tail(state.risks, self.top_k_risks),
            resources=dict(state.resources),
            active_intents=self._tail(state.active_intents, self.top_k_active_intents),
            recent_outcomes=self._tail(state.recent_outcomes, self.recent_outcomes_limit),
            retrieved_memory=retrieved,
            warnings=self._decision_warnings(state),
        )

    def compile_simulation_context(
        self,
        state: WorldState,
        *,
        domain: str = "generic",
        candidate_decisions: list[Decision] | None = None,
        candidate_intents: list[ExecutionIntent] | None = None,
        recent_history: list[ProtocolRecord] | None = None,
    ) -> SimulationContext:
        recent_history = recent_history or []
        decision_context = self.compile_decision_context(
            state,
            domain=domain,
            recent_history=recent_history,
        )
        retrieved = self._query_memory(f"{domain}.simulation")
        decision_candidates = self._serialize_list(candidate_decisions or [], self.candidate_limit)
        intent_candidates = self._serialize_list(candidate_intents or [], self.candidate_limit)
        refs = self._build_refs(
            state,
            recent_history,
            [
                decision_context.id,
                *[candidate.get("id") for candidate in decision_candidates],
                *[candidate.get("id") for candidate in intent_candidates],
                *[record.get("id") for record in retrieved],
            ],
        )

        return SimulationContext.create(
            world_state_id=state.id,
            domain=domain,
            decision_context_ref=decision_context.id,
            budget={**self._base_budget(), "candidate_limit": self.candidate_limit},
            confidence={"state": dict(state.confidence), "method": "deterministic"},
            provenance=self._base_provenance(domain, refs, retrieved),
            refs=refs,
            candidate_decisions=decision_candidates,
            candidate_intents=intent_candidates,
            assumptions=[
                {
                    "id": "bounded_horizon",
                    "description": "Simulation assumes short-horizon effects only.",
                }
            ],
            evaluation_axes=[
                {"id": "success", "description": "Likelihood of success criteria satisfaction."},
                {"id": "risk", "description": "Risk exposure under candidate execution."},
            ],
            historical_analogs=retrieved[: self.memory_query_limit],
            retrieved_memory=retrieved,
        )

    def compile_reflection_context(
        self,
        state: WorldState,
        outcome: Outcome,
        *,
        domain: str = "generic",
        decision: Decision | None = None,
        intent: ExecutionIntent | None = None,
        execution_result: ExecutionResult | None = None,
        recent_history: list[ProtocolRecord] | None = None,
    ) -> ReflectionContext:
        recent_history = recent_history or []
        retrieved = self._query_memory(f"{domain}.reflection")
        refs = self._build_refs(
            state,
            recent_history,
            [
                outcome.id,
                decision.id if decision else None,
                intent.id if intent else None,
                execution_result.id if execution_result else None,
                *[record.get("id") for record in retrieved],
            ],
        )

        expected = intent.success_criteria if intent else []
        actual = {
            "outcome_status": outcome.status,
            "change_count": len(outcome.changes),
            "execution_status": execution_result.status if execution_result else None,
        }

        return ReflectionContext.create(
            world_state_id=state.id,
            domain=domain,
            budget=self._base_budget(),
            confidence={"state": dict(state.confidence), "method": "deterministic"},
            provenance=self._base_provenance(domain, refs, retrieved),
            refs=refs,
            executed_path={
                "decision": self._serialize_record(decision),
                "execution_intent": self._serialize_record(intent),
                "execution_result": self._serialize_record(execution_result),
                "outcome": self._serialize_record(outcome),
            },
            expected_vs_actual={
                "expected": expected,
                "actual": actual,
            },
            impact_summary={
                "outcome_id": outcome.id,
                "outcome_status": outcome.status,
                "world_state_id": state.id,
                "recent_outcomes_count": len(state.recent_outcomes),
            },
            retrieved_lessons=retrieved[: self.memory_query_limit],
            retrieved_memory=retrieved,
            open_questions=[],
        )

    def compile_general_decision_context(
        self,
        state: WorldState,
        general_state: Any,
        *,
        current_intent: str | dict[str, Any] = "",
        active_decision_frame: dict[str, Any] | None = None,
        session: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        recent_history: list[ProtocolRecord] | None = None,
        domain: str = "general",
    ) -> DecisionContext:
        recent_history = recent_history or []
        frame = self._active_decision_frame(general_state, active_decision_frame)
        retrieved = self._query_general_memory(domain, "decision")
        refs = self._build_refs(
            state,
            recent_history,
            [
                *self._general_refs(general_state),
                frame.get("decision_id"),
                frame.get("run_id"),
                frame.get("approval_id"),
                *[record.get("id") for record in retrieved],
            ],
        )

        return DecisionContext.create(
            world_state_id=state.id,
            domain=domain,
            budget=self._base_budget(),
            confidence={"state": dict(state.confidence), "method": "deterministic"},
            provenance=self._base_provenance(domain, refs, retrieved),
            refs=refs,
            current_intent=self._current_intent_payload(current_intent),
            active_decision_frame=frame,
            objectives=self._tail(state.goals, self.top_k_goals),
            constraints=[
                *self._tail(state.constraints, self.top_k_constraints),
                *self._state_payload_list(general_state, "constraints", self.top_k_constraints),
            ][: self.top_k_constraints],
            entities=self._slice_entities(state.entities, self.top_k_entities),
            signals=[
                *self._tail(state.signals, self.top_k_signals),
                *self._state_payload_list(general_state, "signals", self.top_k_signals),
            ][: self.top_k_signals],
            risks=[
                *self._tail(state.risks, self.top_k_risks),
                *self._state_payload_list(general_state, "risks", self.top_k_risks),
            ][: self.top_k_risks],
            resources={
                **dict(state.resources),
                "general_resources": self._state_payload_list(
                    general_state,
                    "resources",
                    self.top_k_entities,
                ),
            },
            active_intents=[
                *self._tail(state.active_intents, self.top_k_active_intents),
                *self._state_payload_list(
                    general_state,
                    "intents",
                    self.top_k_active_intents,
                ),
            ][: self.top_k_active_intents],
            recent_decisions=self._recent_decisions(general_state),
            recent_approvals=self._recent_approvals(general_state),
            recent_outcomes=[
                *self._tail(state.recent_outcomes, self.recent_outcomes_limit),
                *self._state_payload_list(
                    general_state,
                    "outcomes",
                    self.recent_outcomes_limit,
                ),
            ][: self.recent_outcomes_limit],
            executor_affordance=self._executor_affordance(frame, config),
            session_summary=self._session_summary(session, domain=domain),
            workspace_context=self._workspace_context(config),
            retrieved_memory=retrieved,
            warnings=self._general_warnings(state, general_state),
            metadata={"general_state": self._general_state_summary(general_state)},
        )

    def compile_general_simulation_context(
        self,
        state: WorldState,
        general_state: Any,
        *,
        current_intent: str | dict[str, Any] = "",
        candidates: list[Any] | None = None,
        active_decision_frame: dict[str, Any] | None = None,
        session: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        recent_history: list[ProtocolRecord] | None = None,
        domain: str = "general",
    ) -> SimulationContext:
        recent_history = recent_history or []
        frame = self._active_decision_frame(general_state, active_decision_frame)
        decision_context = self.compile_general_decision_context(
            state,
            general_state,
            current_intent=current_intent,
            active_decision_frame=frame,
            session=session,
            config=config,
            recent_history=recent_history,
            domain=domain,
        )
        retrieved = self._query_general_memory(domain, "simulation")
        decision_candidates = self._serialize_list(candidates or [], self.candidate_limit)
        refs = self._build_refs(
            state,
            recent_history,
            [
                decision_context.id,
                frame.get("decision_id"),
                *[candidate.get("candidate_id") or candidate.get("id") for candidate in decision_candidates],
                *[record.get("id") for record in retrieved],
            ],
        )

        return SimulationContext.create(
            world_state_id=state.id,
            domain=domain,
            decision_context_ref=decision_context.id,
            budget={**self._base_budget(), "candidate_limit": self.candidate_limit},
            confidence={"state": dict(state.confidence), "method": "deterministic"},
            provenance=self._base_provenance(domain, refs, retrieved),
            refs=refs,
            current_intent=self._current_intent_payload(current_intent),
            active_decision_frame=frame,
            candidate_decisions=decision_candidates,
            candidate_intents=[],
            recent_decisions=self._recent_decisions(general_state),
            recent_approvals=self._recent_approvals(general_state),
            executor_affordance=self._executor_affordance(frame, config),
            session_summary=self._session_summary(session, domain=domain),
            workspace_context=self._workspace_context(config),
            assumptions=[
                {
                    "id": "decision_relevant_state",
                    "description": "Simulation uses compact state and recent decision memory, not full workspace context.",
                }
            ],
            evaluation_axes=[
                {"id": "expected_outcome", "description": "Likely result if this candidate is chosen."},
                {"id": "downside", "description": "Main downside or failure mode."},
                {"id": "success_signal", "description": "Observable signal that the choice worked."},
                {"id": "time_fit", "description": "Fit with the current time and executor constraints."},
            ],
            historical_analogs=retrieved[: self.memory_query_limit],
            retrieved_memory=retrieved,
            metadata={"general_state": self._general_state_summary(general_state)},
        )

    def compile_general_reflection_context(
        self,
        state: WorldState,
        general_state: Any,
        outcome: Outcome | dict[str, Any],
        *,
        current_intent: str | dict[str, Any] = "",
        decision_artifact: dict[str, Any] | None = None,
        execution_artifact: dict[str, Any] | None = None,
        active_decision_frame: dict[str, Any] | None = None,
        session: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        recent_history: list[ProtocolRecord] | None = None,
        domain: str = "general",
    ) -> ReflectionContext:
        recent_history = recent_history or []
        frame = self._active_decision_frame(general_state, active_decision_frame)
        outcome_payload = self._serialize_record(outcome)
        retrieved = self._query_general_memory(domain, "reflection")
        refs = self._build_refs(
            state,
            recent_history,
            [
                outcome_payload.get("outcome_id") or outcome_payload.get("id"),
                self._dict_id(decision_artifact, "decision_id"),
                self._dict_id(execution_artifact, "execution_id"),
                frame.get("decision_id"),
                *[record.get("id") for record in retrieved],
            ],
        )
        actual = {
            "protocol_status": outcome_payload.get("protocol_status"),
            "task_status": outcome_payload.get("task_status")
            or outcome_payload.get("status"),
            "state_updated": outcome_payload.get("state_updated"),
            "summary": outcome_payload.get("summary", ""),
        }
        expected = self._frame_expected_result(frame)

        return ReflectionContext.create(
            world_state_id=state.id,
            domain=domain,
            budget=self._base_budget(),
            confidence={"state": dict(state.confidence), "method": "deterministic"},
            provenance=self._base_provenance(domain, refs, retrieved),
            refs=refs,
            current_intent=self._current_intent_payload(current_intent),
            active_decision_frame=frame,
            recent_decisions=self._recent_decisions(general_state),
            recent_approvals=self._recent_approvals(general_state),
            executor_affordance=self._executor_affordance(frame, config),
            session_summary=self._session_summary(session, domain=domain),
            workspace_context=self._workspace_context(config),
            executed_path={
                "decision": dict(decision_artifact or {}),
                "execution": dict(execution_artifact or {}),
                "outcome": outcome_payload,
                "active_decision_frame": frame,
            },
            expected_vs_actual={
                "expected": expected,
                "actual": actual,
            },
            impact_summary={
                "decision_id": self._dict_id(decision_artifact, "decision_id")
                or frame.get("decision_id", ""),
                "outcome_id": outcome_payload.get("outcome_id") or outcome_payload.get("id", ""),
                "task_status": actual["task_status"],
                "state_updated": actual["state_updated"],
                "world_state_id": state.id,
            },
            retrieved_lessons=retrieved[: self.memory_query_limit],
            retrieved_memory=retrieved,
            open_questions=[],
            metadata={"general_state": self._general_state_summary(general_state)},
        )

    def write_reflection(
        self,
        reflection_record: dict[str, Any],
        *,
        domain: str = "generic",
        provider: MemoryProvider | None = None,
    ) -> list[str]:
        active_provider = provider or self.memory_provider
        if active_provider is None:
            return []

        payload = dict(reflection_record)
        payload.setdefault("domain", domain)
        refs = payload.get("refs")
        ref_list = refs if isinstance(refs, list) else None
        return active_provider.write(
            [payload],
            namespace=f"{domain}.reflection",
            refs=ref_list,
        )

    def _base_budget(self) -> dict[str, Any]:
        return {
            "top_k_entities": self.top_k_entities,
            "top_k_signals": self.top_k_signals,
            "top_k_risks": self.top_k_risks,
            "top_k_goals": self.top_k_goals,
            "top_k_constraints": self.top_k_constraints,
            "top_k_active_intents": self.top_k_active_intents,
            "recent_outcomes_limit": self.recent_outcomes_limit,
            "memory_query_limit": self.memory_query_limit,
        }

    def _base_provenance(
        self,
        domain: str,
        refs: list[str],
        retrieved_memory: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "domain": domain,
            "source_refs": refs,
            "memory_refs": [record.get("id") for record in retrieved_memory if "id" in record],
            "compiler": "DeterministicContextCompiler@0.1",
        }

    def _query_memory(self, namespace: str) -> list[dict[str, Any]]:
        if self.memory_provider is None:
            return []
        return self.memory_provider.query(
            namespace=namespace,
            limit=self.memory_query_limit,
        )

    def _query_memory_namespaces(self, namespaces: list[str]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for namespace in namespaces:
            for record in self._query_memory(namespace):
                key = str(record.get("id") or (namespace, len(records)))
                if key in seen:
                    continue
                seen.add(key)
                records.append(record)
                if len(records) >= self.memory_query_limit:
                    return records
        return records

    def _query_general_memory(self, domain: str, stage: str) -> list[dict[str, Any]]:
        namespaces_by_stage = {
            "decision": [
                f"{domain}.decision",
                f"{domain}.preference",
                f"{domain}.reflection",
            ],
            "simulation": [
                f"{domain}.simulation",
                f"{domain}.decision",
                f"{domain}.preference",
                f"{domain}.reflection",
            ],
            "reflection": [
                f"{domain}.reflection",
                f"{domain}.decision",
                f"{domain}.preference",
            ],
        }
        return self._query_memory_namespaces(
            namespaces_by_stage.get(stage, [f"{domain}.{stage}"])
        )

    def _build_refs(
        self,
        state: WorldState,
        recent_history: list[ProtocolRecord],
        extra_refs: list[Any] | None = None,
    ) -> list[str]:
        refs: list[str] = [state.id, *state.refs[-self.history_ref_limit :]]
        refs.extend(record.id for record in recent_history[-self.history_ref_limit :])
        if extra_refs:
            refs.extend(str(ref) for ref in extra_refs if ref)
        return list(dict.fromkeys(refs))

    @staticmethod
    def _tail(values: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        if limit < 0:
            return list(values)
        return list(values[-limit:])

    @staticmethod
    def _slice_entities(entities: dict[str, Any], limit: int) -> dict[str, Any]:
        if limit < 0:
            return dict(entities)
        keys = sorted(entities.keys())[:limit]
        return {key: entities[key] for key in keys}

    def _decision_warnings(self, state: WorldState) -> list[str]:
        warnings: list[str] = []
        if len(state.signals) > self.top_k_signals:
            warnings.append("signals_truncated")
        if len(state.risks) > self.top_k_risks:
            warnings.append("risks_truncated")
        if len(state.recent_outcomes) > self.recent_outcomes_limit:
            warnings.append("recent_outcomes_truncated")
        return warnings

    def _general_warnings(self, state: WorldState, general_state: Any) -> list[str]:
        warnings = self._decision_warnings(state)
        if len(self._state_sequence(general_state, "intents")) > self.top_k_active_intents:
            warnings.append("general_intents_truncated")
        if len(self._state_sequence(general_state, "approvals")) > self.recent_outcomes_limit:
            warnings.append("recent_approvals_truncated")
        if len(self._state_sequence(general_state, "decision_checkpoints")) > self.recent_outcomes_limit:
            warnings.append("recent_decisions_truncated")
        return warnings

    def _serialize_list(self, values: list[Any], limit: int) -> list[dict[str, Any]]:
        if limit < 0:
            return [self._serialize_record(value) for value in values]
        return [self._serialize_record(value) for value in values[:limit]]

    @staticmethod
    def _serialize_record(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if hasattr(value, "to_payload"):
            return value.to_payload()
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, dict):
            return dict(value)
        return {"value": str(value)}

    def _current_intent_payload(self, current_intent: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(current_intent, dict):
            return dict(current_intent)
        text = str(current_intent).strip()
        return {"text": text} if text else {}

    def _active_decision_frame(
        self,
        general_state: Any,
        active_decision_frame: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if isinstance(active_decision_frame, dict):
            return dict(active_decision_frame)
        metadata = self._state_metadata(general_state)
        frame = metadata.get("active_decision_frame")
        return dict(frame) if isinstance(frame, dict) else {}

    def _executor_affordance(
        self,
        active_decision_frame: dict[str, Any],
        config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        for key in ("executor_affordance", "affordance"):
            value = active_decision_frame.get(key)
            if isinstance(value, dict):
                return dict(value)
        selected = active_decision_frame.get("selected")
        if isinstance(selected, dict):
            for key in ("executor_affordance", "affordance"):
                value = selected.get(key)
                if isinstance(value, dict):
                    return dict(value)
        config = config or {}
        return {
            key: config[key]
            for key in (
                "executor",
                "executor_provider",
                "executor_permission",
                "executor_transport",
                "executor_status",
            )
            if key in config
        }

    def _session_summary(self, session: dict[str, Any] | None, *, domain: str = "general") -> dict[str, Any]:
        if not isinstance(session, dict):
            summary: dict[str, Any] = {}
        else:
            summary = {
                key: session[key]
                for key in (
                    "session_id",
                    "status",
                    "runs",
                    "decisions",
                    "pending_approvals",
                    "active_state_ref",
                )
                if key in session
            }
            last_decision = session.get("last_decision")
            if isinstance(last_decision, dict):
                for key in ("run_id", "decision_id", "trace_ref"):
                    value = last_decision.get(key)
                    if value:
                        summary[f"last_{key}"] = value
        latest = self._latest_session_summary(domain)
        if latest:
            summary["rolling_summary"] = self._compact_session_summary(latest)
            markdown = str(latest.get("markdown") or "")
            if markdown:
                summary["summary_text"] = markdown
        return summary

    def _latest_session_summary(self, domain: str) -> dict[str, Any]:
        if self.memory_provider is None:
            return {}
        records = self.memory_provider.query(
            namespace=f"{domain}.session_summary",
            limit=-1,
        )
        return dict(records[-1]) if records else {}

    @staticmethod
    def _compact_session_summary(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(record.get("id") or ""),
            "schema_version": str(record.get("schema_version") or ""),
            "summary_type": str(record.get("summary_type") or ""),
            "domain": str(record.get("domain") or ""),
            "updated_at": str(record.get("updated_at") or ""),
            "model": dict(record.get("model")) if isinstance(record.get("model"), dict) else {},
            "counts": dict(record.get("counts")) if isinstance(record.get("counts"), dict) else {},
            "current_goal": dict(record.get("current_goal")) if isinstance(record.get("current_goal"), dict) else {},
            "active_decision": dict(record.get("active_decision")) if isinstance(record.get("active_decision"), dict) else {},
            "user_preferences": list(record.get("user_preferences")) if isinstance(record.get("user_preferences"), list) else [],
            "recent_decisions": list(record.get("recent_decisions")) if isinstance(record.get("recent_decisions"), list) else [],
            "execution_outcomes": list(record.get("execution_outcomes")) if isinstance(record.get("execution_outcomes"), list) else [],
            "open_threads": list(record.get("open_threads")) if isinstance(record.get("open_threads"), list) else [],
        }

    def _workspace_context(self, config: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(config, dict):
            return {}
        keys = (
            "llm_provider",
            "llm_model",
            "llm_candidate_expand",
            "llm_simulation",
            "executor",
            "executor_provider",
            "executor_permission",
            "executor_transport",
            "perception_provider",
            "memory_provider",
            "memory_path",
            "context_compiler",
            "memory_summary_provider",
            "memory_summary_llm_min_new_records",
            "memory_summary_trigger_chars",
            "memory_summary_target_chars",
        )
        context = {key: config[key] for key in keys if key in config}
        context["skill_catalog"] = _runtime_skill_catalog_summary(config)
        return context

    def _recent_decisions(self, general_state: Any) -> list[dict[str, Any]]:
        decisions = self._state_payload_list(
            general_state,
            "decision_checkpoints",
            self.recent_outcomes_limit,
        )
        compact: list[dict[str, Any]] = []
        for decision in decisions:
            approval = decision.get("approval")
            compact.append(
                {
                    "decision_id": decision.get("decision_id", ""),
                    "trace_ref": decision.get("trace_ref", ""),
                    "selected_candidate_id": decision.get("selected_candidate_id", ""),
                    "status": decision.get("status", ""),
                    "recommendation": decision.get("recommendation", ""),
                    "approval_id": approval.get("approval_id", "")
                    if isinstance(approval, dict)
                    else "",
                    "compare_ref": decision.get("compare_ref", ""),
                }
            )
        return compact

    def _recent_approvals(self, general_state: Any) -> list[dict[str, Any]]:
        approvals = self._state_payload_list(
            general_state,
            "approvals",
            self.recent_outcomes_limit,
        )
        return [
            {
                "approval_id": item.get("approval_id", ""),
                "decision_id": item.get("decision_id", ""),
                "candidate_id": item.get("candidate_id", ""),
                "status": item.get("status", ""),
                "execution_allowed": item.get("execution_allowed", False),
            }
            for item in approvals
        ]

    def _general_state_summary(self, general_state: Any) -> dict[str, Any]:
        fields = (
            "observations",
            "commitments",
            "work_items",
            "capabilities",
            "open_loops",
        )
        summary = {
            field: self._state_payload_list(general_state, field, self.recent_outcomes_limit)
            for field in fields
        }
        summary["counts"] = {
            field: len(self._state_sequence(general_state, field))
            for field in (
                "signals",
                "observations",
                "intents",
                "commitments",
                "work_items",
                "resources",
                "capabilities",
                "constraints",
                "risks",
                "open_loops",
                "approvals",
                "decision_checkpoints",
                "outcomes",
            )
        }
        return summary

    def _state_payload_list(self, state: Any, attr: str, limit: int) -> list[dict[str, Any]]:
        return [
            self._serialize_record(item)
            for item in self._limited_tail(self._state_sequence(state, attr), limit)
        ]

    @staticmethod
    def _limited_tail(values: list[Any], limit: int) -> list[Any]:
        if limit < 0:
            return list(values)
        if limit == 0:
            return []
        return list(values[-limit:])

    @staticmethod
    def _state_sequence(state: Any, attr: str) -> list[Any]:
        if isinstance(state, dict):
            value = state.get(attr, [])
        else:
            value = getattr(state, attr, [])
        return list(value) if isinstance(value, list) else []

    @staticmethod
    def _state_metadata(state: Any) -> dict[str, Any]:
        if isinstance(state, dict):
            metadata = state.get("metadata", {})
        else:
            metadata = getattr(state, "metadata", {})
        return dict(metadata) if isinstance(metadata, dict) else {}

    def _general_refs(self, general_state: Any) -> list[str]:
        refs: list[str] = []
        if hasattr(general_state, "state_id"):
            refs.append(str(general_state.state_id))
        refs.extend(str(ref) for ref in self._state_sequence(general_state, "trace_refs") if ref)
        metadata = self._state_metadata(general_state)
        for key in ("state_ref", "session_ref", "decision_ref"):
            value = metadata.get(key)
            if value:
                refs.append(str(value))
        return refs

    @staticmethod
    def _dict_id(payload: dict[str, Any] | None, key: str) -> str:
        if not isinstance(payload, dict):
            return ""
        value = payload.get(key) or payload.get("id")
        return str(value) if value else ""

    @staticmethod
    def _frame_expected_result(active_decision_frame: dict[str, Any]) -> Any:
        selected = active_decision_frame.get("selected")
        if isinstance(selected, dict):
            for key in ("expected_result", "expected", "simulation"):
                value = selected.get(key)
                if value:
                    return value
        for key in ("expected_result", "expected", "simulation"):
            value = active_decision_frame.get(key)
            if value:
                return value
        return {}


def _runtime_skill_catalog_summary(config: dict[str, Any]) -> dict[str, Any]:
    executor = str(config.get("executor") or "dry_run")
    executor_id = f"spice.{_safe_segment(executor)}"
    return {
        "source": "runtime_static_catalog",
        "executors": [
            {
                "executor_id": executor_id,
                "display_name": f"{executor} executor hint",
                "skill_ids": [
                    "runtime.context.prepare",
                    "runtime.work_item.triage",
                    "runtime.state.record",
                    "runtime.intent.execute",
                ],
            }
        ],
        "builtin_skill_ids": [
            "state.record.builtin",
            "user.clarify.builtin",
            "work_item.triage.read_only",
            "intent.execute.generic",
        ],
    }


def _safe_segment(value: str) -> str:
    text = "".join(char.lower() if char.isalnum() or char in "._-" else "_" for char in value)
    return text.strip("._-") or "executor"
