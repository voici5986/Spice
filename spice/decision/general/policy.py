from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

from spice.decision import (
    CandidateDecision,
    DecisionGuidance,
    DecisionGuidanceSupport,
    DecisionObjective,
    GuidedDecisionPolicy,
    PolicyIdentity,
    SafetyConstraint,
    load_decision_guidance,
)
from spice.decision.compare_payload import normalize_compare_payload
from spice.decision.general.approval import Approval
from spice.decision.general.candidates import (
    ExecutionBoundary,
    GenericCandidate,
    PLANNING_ACTION_TYPES,
    generate_generic_candidates,
    is_approval_eligible_executable_candidate,
)
from spice.decision.general.permissions import infer_executor_permission_requirement
from spice.decision.general.state import GeneralDecisionState
from spice.decision.general.trace import CandidateTraceRef, DecisionCheckpoint
from spice.decision.general.types import payload_value
from spice.protocols import Decision
from spice.language import detect_display_language

GENERIC_SCORE_DIMENSIONS = (
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
)
GENERIC_CONSTRAINT_IDS = ("no_declared_veto_violation",)
SELECTION_POOL_CONSTRAINT_ID = "selection_pool_eligible"
GENERIC_TRADEOFF_RULE_IDS = (
    "prefer_lower_risk_when_candidates_differ",
    "prefer_higher_confidence_when_candidates_differ",
)


@dataclass(slots=True)
class GenericPolicyResult:
    decision: Decision
    checkpoint: DecisionCheckpoint
    compare_payload: dict[str, Any]
    candidates: list[GenericCandidate] = field(default_factory=list)
    policy_candidates: list[CandidateDecision] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "decision": payload_value(self.decision),
            "checkpoint": self.checkpoint.to_payload(),
            "compare_payload": payload_value(self.compare_payload),
            "candidates": [candidate.to_payload() for candidate in self.candidates],
            "policy_candidates": [payload_value(candidate) for candidate in self.policy_candidates],
        }


class GenericPolicyAdapter:
    def __init__(
        self,
        guidance: DecisionGuidance,
        support: DecisionGuidanceSupport | None = None,
    ) -> None:
        self.guidance = guidance
        self.support = support or generic_decision_guidance_support()

    @classmethod
    def from_decision_profile(
        cls,
        path: str | Path,
        support: DecisionGuidanceSupport | None = None,
    ) -> "GenericPolicyAdapter":
        return cls(load_decision_guidance(path), support=support)

    def evaluate(
        self,
        state: GeneralDecisionState,
        candidates: list[GenericCandidate] | None = None,
        *,
        decision_id: str | None = None,
        trace_ref: str | None = None,
        run_intent_mode: str = "auto",
        selection_candidate_ids: set[str] | None = None,
        selection_pool_reason: str | None = None,
    ) -> GenericPolicyResult:
        resolved_candidates = (
            generate_generic_candidates(state) if candidates is None else list(candidates)
        )
        if not resolved_candidates:
            raise ValueError("GenericPolicyAdapter requires at least one candidate.")

        base_policy = _GenericCandidateDecisionPolicy(
            resolved_candidates,
            guidance=self.guidance,
            run_intent_mode=run_intent_mode,
            selection_candidate_ids=selection_candidate_ids,
            selection_pool_reason=selection_pool_reason,
        )
        guided_policy = GuidedDecisionPolicy(
            base_policy,
            self.guidance,
            support=self.support,
        )
        policy_candidates = base_policy.propose(state, None)
        decision = guided_policy.select(policy_candidates, DecisionObjective(), [])
        selected_candidate_id = str(decision.attributes.get("selected_candidate_id", ""))
        if not selected_candidate_id:
            raise ValueError("Guided policy did not return selected_candidate_id.")

        resolved_decision_id = decision_id or decision.id
        resolved_trace_ref = trace_ref or _stable_ref(
            "trace",
            {
                "state_id": state.state_id,
                "guidance": self.guidance.source_hash,
                "candidates": [candidate.to_payload() for candidate in resolved_candidates],
                "selected_candidate_id": selected_candidate_id,
            },
        )
        profile_ref = self.guidance.artifact_id or self.guidance.source_path or self.guidance.source_hash
        selected_candidate = _generic_candidate_by_id(
            resolved_candidates,
            selected_candidate_id,
        )
        if selected_candidate.availability_status == "blocked":
            raise ValueError(
                "GenericPolicyAdapter selected a blocked candidate; "
                "check guidance hard constraints or candidate availability."
            )
        checkpoint = _build_checkpoint(
            decision_id=resolved_decision_id,
            trace_ref=resolved_trace_ref,
            state=state,
            profile_ref=profile_ref,
            decision=decision,
            candidates=resolved_candidates,
            selected_candidate=selected_candidate,
        )
        compare_payload = _build_compare_payload(
            decision_id=resolved_decision_id,
            trace_ref=resolved_trace_ref,
            state=state,
            decision=decision,
            candidates=resolved_candidates,
            policy_candidates=policy_candidates,
            selected_candidate=selected_candidate,
        )
        return GenericPolicyResult(
            decision=decision,
            checkpoint=checkpoint,
            compare_payload=compare_payload,
            candidates=resolved_candidates,
            policy_candidates=policy_candidates,
        )


def generic_decision_guidance_support() -> DecisionGuidanceSupport:
    return DecisionGuidanceSupport(
        score_dimensions=set(GENERIC_SCORE_DIMENSIONS),
        constraint_ids={*GENERIC_CONSTRAINT_IDS, SELECTION_POOL_CONSTRAINT_ID},
        tradeoff_rule_ids=set(GENERIC_TRADEOFF_RULE_IDS),
    )


def generic_candidate_to_policy_candidate(
    candidate: GenericCandidate,
    *,
    state: GeneralDecisionState | None = None,
    guidance: DecisionGuidance | None = None,
    run_intent_mode: str = "auto",
) -> CandidateDecision:
    score_breakdown = _score_breakdown(
        candidate,
        state=state,
        guidance=guidance,
        run_intent_mode=run_intent_mode,
    )
    return CandidateDecision(
        id=candidate.candidate_id,
        action=candidate.action_type,
        params={
            "generic_candidate": candidate.to_payload(),
            "constraint_checks": {
                "no_declared_veto_violation": (
                    "fail" if candidate.availability_status == "blocked" else "pass"
                ),
            },
        },
        score_total=sum(score_breakdown.values()) / len(score_breakdown),
        score_breakdown=score_breakdown,
        risk=_risk_value(candidate),
        confidence=_confidence_value(candidate),
    )


class _GenericCandidateDecisionPolicy:
    identity = PolicyIdentity.create(
        policy_name="spice.general.generic_candidate_policy",
        policy_version="0.1",
        implementation_fingerprint="generic-candidate-policy-v2",
    )
    decision_guidance_support = generic_decision_guidance_support()

    def __init__(
        self,
        candidates: list[GenericCandidate],
        *,
        guidance: DecisionGuidance | None = None,
        run_intent_mode: str = "auto",
        selection_candidate_ids: set[str] | None = None,
        selection_pool_reason: str | None = None,
    ) -> None:
        self.candidates = list(candidates)
        self.guidance = guidance
        self.run_intent_mode = run_intent_mode
        self.selection_candidate_ids = (
            {str(item) for item in selection_candidate_ids}
            if selection_candidate_ids is not None
            else None
        )
        self.selection_pool_reason = (
            str(selection_pool_reason).strip()
            if str(selection_pool_reason or "").strip()
            else "Candidate is visible for comparison but excluded from the active selection pool."
        )

    def propose(self, state: Any, context: Any) -> list[CandidateDecision]:
        del context
        resolved_state = state if isinstance(state, GeneralDecisionState) else None
        return [
            self._to_policy_candidate(candidate, state=resolved_state)
            for candidate in self.candidates
        ]

    def _to_policy_candidate(
        self,
        candidate: GenericCandidate,
        *,
        state: GeneralDecisionState | None,
    ) -> CandidateDecision:
        policy_candidate = generic_candidate_to_policy_candidate(
            candidate,
            state=state,
            guidance=self.guidance,
            run_intent_mode=self.run_intent_mode,
        )
        if (
            self.selection_candidate_ids is not None
            and candidate.candidate_id not in self.selection_candidate_ids
        ):
            params = dict(policy_candidate.params)
            checks = dict(params.get("constraint_checks", {}))
            checks["selection_pool_eligible"] = "fail"
            params["constraint_checks"] = checks
            params["selection_pool"] = {
                "eligible": False,
                "reason": self.selection_pool_reason,
            }
            policy_candidate.params = params
        return policy_candidate

    def select(
        self,
        candidates: list[CandidateDecision],
        objective: DecisionObjective,
        constraints: list[SafetyConstraint],
    ) -> Decision:
        del candidates, objective, constraints
        raise AssertionError("GuidedDecisionPolicy owns generic policy selection.")


def _build_checkpoint(
    *,
    decision_id: str,
    trace_ref: str,
    state: GeneralDecisionState,
    profile_ref: str,
    decision: Decision,
    candidates: list[GenericCandidate],
    selected_candidate: GenericCandidate,
) -> DecisionCheckpoint:
    selected_candidate_id = str(decision.attributes.get("selected_candidate_id", ""))
    approval = _build_approval_from_affordance(
        decision_id=decision_id,
        selected_candidate_id=selected_candidate_id,
        selected_candidate=selected_candidate,
    )
    return DecisionCheckpoint(
        decision_id=decision_id,
        trace_ref=trace_ref,
        state_ref=state.state_id,
        profile_ref=profile_ref,
        selected_candidate_id=selected_candidate_id,
        status="recommended",
        recommendation=f"Select {selected_candidate.action_type}: {selected_candidate.intent}",
        candidate_refs=[
            CandidateTraceRef(
                candidate_id=candidate.candidate_id,
                action_type=candidate.action_type,
                status=_candidate_trace_status(candidate, selected_candidate_id),
                metadata={"availability_status": candidate.availability_status},
            )
            for candidate in candidates
        ],
        approval=approval,
        execution_boundary=selected_candidate.execution_boundary.to_payload(),
        metadata={
            "policy_name": str(decision.attributes.get("policy_name", "")),
            "policy_version": str(decision.attributes.get("policy_version", "")),
            "policy_hash": str(decision.attributes.get("policy_hash", "")),
            "decision_guidance": dict(decision.attributes.get("decision_guidance", {})),
            "decision_guidance_validation": dict(
                decision.attributes.get("decision_guidance_validation", {})
            ),
        },
    )


def _build_approval_from_affordance(
    *,
    decision_id: str,
    selected_candidate_id: str,
    selected_candidate: GenericCandidate,
) -> Approval | None:
    execution_affordance = dict(
        selected_candidate.metadata.get("execution_affordance", {}) or {}
    )
    if not _is_runtime_execution_affordance(execution_affordance):
        return None
    if not is_approval_eligible_executable_candidate(selected_candidate):
        return None
    if not execution_affordance.get("executor_available"):
        return None
    if not execution_affordance.get("executable"):
        return None
    approval_affordance = dict(execution_affordance.get("approval", {}) or {})
    if not approval_affordance.get("required"):
        return None

    permission = infer_executor_permission_requirement(selected_candidate)
    executor = dict(execution_affordance.get("executor", {}) or {})
    return Approval(
        approval_id=f"approval.{_ref_slug(decision_id)}",
        decision_id=decision_id,
        candidate_id=selected_candidate_id,
        status="pending",
        mode="confirm_before_execution",
        execution_allowed=False,
        metadata={
            "required_executor_permission": permission.required_permission,
            "required_executor_permission_reason": permission.reason,
            "executor_available": True,
            "executor_id": str(executor.get("executor_id") or ""),
            "permission_requirement": {
                "required_permission": permission.required_permission,
                "reason": permission.reason,
                "source": permission.source,
                "side_effect_class": permission.side_effect_class,
                "target_refs": list(permission.target_refs),
            },
            "execution_affordance": execution_affordance,
        },
    )


def _is_runtime_execution_affordance(value: dict[str, Any]) -> bool:
    return (
        str(value.get("schema_version") or "") == "0.1"
        and str(value.get("generated_by") or "") == "spice.runtime.execution_affordance"
    )


def _build_compare_payload(
    *,
    decision_id: str,
    trace_ref: str,
    state: GeneralDecisionState,
    decision: Decision,
    candidates: list[GenericCandidate],
    policy_candidates: list[CandidateDecision],
    selected_candidate: GenericCandidate,
) -> dict[str, Any]:
    selected_candidate_id = str(decision.attributes.get("selected_candidate_id", ""))
    policy_by_id = {candidate.id: candidate for candidate in policy_candidates}
    display_candidates = _display_ordered_candidates(candidates, selected_candidate_id)
    explanation = dict(decision.attributes.get("decision_guidance_explanation", {}))
    candidate_scores = dict(explanation.get("candidate_scores", {}))
    veto_events = list(decision.attributes.get("veto_events", []))
    selected_product = _candidate_product_metadata(selected_candidate)
    display_language = detect_display_language(
        " ".join(
            item
            for item in [
                selected_candidate.intent,
                *(candidate.intent for candidate in candidates[:3]),
            ]
            if item
        )
    )

    raw_payload = {
        "decision_id": decision_id,
        "trace_ref": trace_ref,
        "display_language": display_language,
        "decision_relevant_state_summary": _state_summary(state),
        "candidate_decisions": [
            _compare_candidate(candidate, selected_candidate_id, state)
            for candidate in display_candidates
        ],
        "score_breakdown": {
            "selection_direction": str(explanation.get("selection_direction", "max")),
            "candidates": {
                candidate.candidate_id: _compare_score_candidate(
                    candidate=candidate,
                    policy_candidate=policy_by_id[candidate.candidate_id],
                    score_entry=dict(candidate_scores.get(candidate.candidate_id, {})),
                    veto_events=veto_events,
                )
                for candidate in display_candidates
            },
        },
        "selected_recommendation": {
            "candidate_id": selected_candidate_id,
            "action": selected_candidate.action_type,
            "title": selected_product.get("user_facing_title") or _title(selected_candidate.action_type),
            "selection_reason": str(explanation.get("final_selection_reason", "")),
            "decision_basis": _selected_basis(
                selected_candidate_id=selected_candidate_id,
                score_entry=dict(candidate_scores.get(selected_candidate_id, {})),
                veto_events=veto_events,
                display_language=display_language,
            ),
            "human_summary": selected_product.get("recommended_action")
            or f"Select {selected_candidate.action_type}.",
            "reason_summary": selected_product.get("why_now")
            or list(selected_candidate.why_available),
            "requires_confirmation": selected_candidate.requires_confirmation,
            "execution_affordance": _candidate_execution_affordance(selected_candidate),
            "skill_resolution": _candidate_skill_resolution(selected_candidate),
        },
        "why_not_the_others": [
            _why_not(
                candidate=candidate,
                selected_candidate=selected_candidate,
                selected_policy_candidate=policy_by_id[selected_candidate_id],
                policy_candidate=policy_by_id[candidate.candidate_id],
                selected_score_entry=dict(candidate_scores.get(selected_candidate_id, {})),
                candidate_score_entry=dict(candidate_scores.get(candidate.candidate_id, {})),
                veto_events=veto_events,
                display_language=display_language,
            )
            for candidate in display_candidates
            if candidate.candidate_id != selected_candidate_id
        ],
        "expected_outcome_or_risk": _expected_effect(selected_candidate),
        "execution_boundary": _compare_execution_boundary(selected_candidate.execution_boundary),
    }
    return normalize_compare_payload(raw_payload)


def _display_ordered_candidates(
    candidates: list[GenericCandidate],
    selected_candidate_id: str,
) -> list[GenericCandidate]:
    indexed = list(enumerate(candidates))
    indexed.sort(
        key=lambda item: _candidate_display_sort_key(
            item[1],
            selected_candidate_id=selected_candidate_id,
            original_index=item[0],
        )
    )
    return [candidate for _, candidate in indexed]


def _candidate_display_sort_key(
    candidate: GenericCandidate,
    *,
    selected_candidate_id: str,
    original_index: int,
) -> tuple[int, int, int]:
    if candidate.candidate_id == selected_candidate_id:
        return (0, 0, original_index)
    explicit_index = _candidate_explicit_option_index(candidate)
    if explicit_index is not None:
        return (1, explicit_index, original_index)
    if _candidate_is_decision_layer(candidate):
        return (2, 0, original_index)
    if _candidate_has_execution_handoff(candidate):
        return (3, 0, original_index)
    return (4, 0, original_index)


def _candidate_explicit_option_index(candidate: GenericCandidate) -> int | None:
    metadata = dict(candidate.metadata or {})
    raw_index = metadata.get("explicit_option_index")
    try:
        index = int(raw_index)
    except (TypeError, ValueError):
        return None
    return index if index > 0 else None


def _candidate_is_decision_layer(candidate: GenericCandidate) -> bool:
    metadata = dict(candidate.metadata or {})
    return (
        candidate.candidate_kind == "decision"
        or str(metadata.get("candidate_kind") or "") == "decision"
        or str(metadata.get("candidate_source") or "") in {"llm_generator", "explicit_options"}
        or str(metadata.get("source") or "") == "explicit_options"
    )


def _candidate_has_execution_handoff(candidate: GenericCandidate) -> bool:
    return (
        getattr(candidate.execution_intent, "requested", False)
        or str(getattr(candidate.execution_intent, "intent_class", "")) == "execution_requested"
        or candidate.requires_confirmation
        or candidate.action_type in {"intent.execute", "capability.use"}
    )


def _score_breakdown(
    candidate: GenericCandidate,
    *,
    state: GeneralDecisionState | None = None,
    guidance: DecisionGuidance | None = None,
    run_intent_mode: str = "auto",
) -> dict[str, float]:
    risk = _risk_value(candidate)
    scores = {
        "outcome_value": _outcome_value(candidate),
        "risk_reduction": max(0.0, 1.0 - risk),
        "reversibility": _reversibility_value(candidate),
        "confidence_alignment": _confidence_value(candidate),
        "urgency_alignment": _urgency_alignment(candidate, state),
        "effort_fit": _effort_fit(candidate),
        "impact_potential": _impact_potential(candidate),
        "historical_outcome_alignment": _historical_outcome_alignment(candidate, state),
        "execution_intent_fit": _execution_intent_fit(candidate, run_intent_mode),
    }
    scores["preference_alignment"] = _preference_alignment(scores, guidance)
    return scores


def _execution_intent_fit(candidate: GenericCandidate, run_intent_mode: str) -> float:
    mode = (run_intent_mode or "auto").strip().lower()
    if candidate.availability_status == "blocked":
        return 0.05
    if mode != "act":
        return 0.50
    if _candidate_approval_eligible_from_affordance(candidate):
        return 1.0
    if candidate.action_type in PLANNING_ACTION_TYPES:
        return 0.35
    if _candidate_requests_execution(candidate):
        return 0.55
    return 0.45


def _candidate_requests_execution(candidate: GenericCandidate) -> bool:
    execution_intent = getattr(candidate, "execution_intent", None)
    if execution_intent is None:
        return False
    return (
        getattr(execution_intent, "intent_class", "") == "execution_requested"
        and bool(getattr(execution_intent, "requested", False))
    )


def _candidate_approval_eligible_from_affordance(candidate: GenericCandidate) -> bool:
    execution_affordance = dict(candidate.metadata.get("execution_affordance", {}) or {})
    approval = dict(execution_affordance.get("approval", {}) or {})
    return bool(approval.get("eligible_for_approval") and approval.get("required"))


def _outcome_value(candidate: GenericCandidate) -> float:
    values = {
        "intent.execute": 0.75,
        "capability.use": 0.72,
        "item.triage": 0.68,
        "context.prepare": 0.56,
        "state.observe_more": 0.52,
        "artifact.draft": 0.62,
        "approval.request": 0.48,
        "user.clarify": 0.54,
        "time.defer": 0.36,
        "state.record": 0.28,
        "item.ignore": 0.18,
        "task.split": 0.66,
    }
    value = values.get(candidate.action_type, 0.40)
    if candidate.availability_status == "blocked":
        return min(value, 0.10)
    return value


def _risk_value(candidate: GenericCandidate) -> float:
    levels = {
        "none": 0.0,
        "low": 0.20,
        "medium": 0.50,
        "high": 0.80,
        "unknown": 0.50,
    }
    if candidate.availability_status == "blocked":
        return 1.0
    return levels.get(candidate.risk_profile.level, 0.50)


def _reversibility_value(candidate: GenericCandidate) -> float:
    values = {
        "high": 0.90,
        "medium": 0.55,
        "low": 0.20,
        "unknown": 0.45,
    }
    return values.get(candidate.reversibility, 0.45)


def _confidence_value(candidate: GenericCandidate) -> float:
    values = {
        "available": 0.82,
        "needs_confirmation": 0.68,
        "insufficient_context": 0.30,
        "blocked": 0.05,
    }
    value = values.get(candidate.availability_status, 0.45)
    if candidate.risk_profile.uncertainty == "high":
        value = min(value, 0.50)
    return value


def _urgency_alignment(
    candidate: GenericCandidate,
    state: GeneralDecisionState | None,
) -> float:
    if candidate.availability_status == "blocked":
        return 0.05

    urgency_scores: list[float] = []
    if state is not None:
        target_refs = set(candidate.target_refs)
        for work_item in state.work_items:
            if work_item.work_item_id in target_refs:
                urgency_scores.append(_urgency_value(work_item.urgency))
        for intent in state.intents:
            if intent.intent_id in target_refs:
                urgency_scores.append(_urgency_value(intent.urgency))

    if urgency_scores:
        return max(urgency_scores)

    fallback = {
        "intent.execute": 0.62,
        "capability.use": 0.58,
        "item.triage": 0.56,
        "context.prepare": 0.46,
        "state.observe_more": 0.40,
        "artifact.draft": 0.50,
        "approval.request": 0.52,
        "user.clarify": 0.48,
        "time.defer": 0.22,
        "state.record": 0.18,
        "item.ignore": 0.12,
        "task.split": 0.44,
    }
    return fallback.get(candidate.action_type, 0.40)


def _urgency_value(value: str) -> float:
    scores = {
        "critical": 1.0,
        "urgent": 0.95,
        "high": 0.85,
        "medium": 0.55,
        "normal": 0.45,
        "low": 0.25,
        "none": 0.10,
        "unknown": 0.40,
    }
    return scores.get(str(value).lower(), 0.40)


def _effort_fit(candidate: GenericCandidate) -> float:
    minutes = candidate.estimated_cost.time_minutes
    if minutes is None:
        return 0.50
    if minutes <= 5:
        return 0.95
    if minutes <= 15:
        return 0.82
    if minutes <= 30:
        return 0.66
    if minutes <= 60:
        return 0.48
    if minutes <= 120:
        return 0.28
    return 0.12


def _impact_potential(candidate: GenericCandidate) -> float:
    if candidate.availability_status == "blocked":
        return 0.10

    delta = candidate.expected_state_delta
    refs = len(delta.creates_refs) + len(delta.updates_refs) + len(delta.closes_refs)
    summary_len = len(delta.summary.strip())
    value = 0.25
    if summary_len >= 120:
        value += 0.28
    elif summary_len >= 60:
        value += 0.20
    elif summary_len >= 20:
        value += 0.12
    if refs >= 3:
        value += 0.28
    elif refs == 2:
        value += 0.22
    elif refs == 1:
        value += 0.14
    if candidate.action_type in {"intent.execute", "capability.use", "task.split"}:
        value += 0.10
    if candidate.action_type in {"state.record", "item.ignore", "time.defer"}:
        value -= 0.08
    return _clamp01(value)


def _preference_alignment(
    scores: dict[str, float],
    guidance: DecisionGuidance | None,
) -> float:
    if guidance is None or not guidance.weights:
        return 0.50
    weighted_total = 0.0
    weight_total = 0.0
    for dimension, raw_weight in guidance.weights.items():
        if dimension == "preference_alignment" or dimension not in scores:
            continue
        weight = max(0.0, float(raw_weight))
        if weight == 0:
            continue
        weighted_total += scores[dimension] * weight
        weight_total += weight
    if weight_total <= 0:
        return 0.50
    return _clamp01(weighted_total / weight_total)


def _historical_outcome_alignment(
    candidate: GenericCandidate,
    state: GeneralDecisionState | None,
) -> float:
    summary = _candidate_history(candidate, state)
    if not summary["similar_outcome_count"]:
        return 0.50
    return float(summary["historical_score"])


def _candidate_history(
    candidate: GenericCandidate,
    state: GeneralDecisionState | None,
) -> dict[str, Any]:
    if state is None:
        return _empty_history(candidate.action_type)

    similar = [
        outcome
        for outcome in state.outcomes
        if _outcome_action_type(outcome) == candidate.action_type
    ][-10:]
    if not similar:
        return _empty_history(candidate.action_type)

    success = 0
    failed = 0
    partial = 0
    other = 0
    weighted_total = 0.0
    weight_total = 0.0
    for index, outcome in enumerate(similar):
        status = _outcome_status(outcome)
        value = _outcome_status_value(status)
        weight = 1.0 + (index / max(1, len(similar) - 1)) * 0.5
        weighted_total += value * weight
        weight_total += weight
        if status == "success":
            success += 1
        elif status in {"failed", "error", "timeout", "rejected"}:
            failed += 1
        elif status == "partial":
            partial += 1
        else:
            other += 1

    historical_score = weighted_total / weight_total if weight_total else 0.50
    return {
        "action_type": candidate.action_type,
        "similar_outcome_count": len(similar),
        "success_count": success,
        "failure_count": failed,
        "partial_count": partial,
        "other_count": other,
        "historical_score": _clamp01(historical_score),
        "recent_outcome_ids": [outcome.outcome_id for outcome in similar[-3:]],
    }


def _empty_history(action_type: str) -> dict[str, Any]:
    return {
        "action_type": action_type,
        "similar_outcome_count": 0,
        "success_count": 0,
        "failure_count": 0,
        "partial_count": 0,
        "other_count": 0,
        "historical_score": 0.50,
        "recent_outcome_ids": [],
    }


def _outcome_action_type(outcome: Any) -> str:
    metadata = dict(getattr(outcome, "metadata", {}) or {})
    candidates = [
        getattr(outcome, "action_type", ""),
        metadata.get("action_type"),
        metadata.get("candidate_action_type"),
        metadata.get("general_action_type"),
    ]
    traceability = metadata.get("traceability")
    if isinstance(traceability, dict):
        candidates.extend(
            [
                traceability.get("action_type"),
                traceability.get("candidate_action_type"),
                traceability.get("general_action_type"),
            ]
        )
    output = metadata.get("output")
    if isinstance(output, dict):
        candidates.extend(
            [
                output.get("action_type"),
                output.get("candidate_action_type"),
                output.get("general_action_type"),
            ]
        )
    for item in candidates:
        if item:
            return str(item)
    return _action_type_from_candidate_id(str(getattr(outcome, "candidate_id", "") or ""))


def _action_type_from_candidate_id(candidate_id: str) -> str:
    if not candidate_id.startswith("candidate."):
        return ""
    encoded = candidate_id.removeprefix("candidate.")
    for action_type in _generic_action_types():
        if encoded.startswith(action_type.replace(".", "_")):
            return action_type
    return ""


def _generic_action_types() -> tuple[str, ...]:
    return (
        "intent.execute",
        "capability.use",
        "item.triage",
        "context.prepare",
        "state.observe_more",
        "artifact.draft",
        "approval.request",
        "user.clarify",
        "time.defer",
        "state.record",
        "item.ignore",
        "task.split",
    )


def _outcome_status(outcome: Any) -> str:
    for value in (
        getattr(outcome, "task_status", None),
        getattr(outcome, "status", None),
        getattr(outcome, "protocol_status", None),
    ):
        if value:
            return str(value).lower()
    metadata = dict(getattr(outcome, "metadata", {}) or {})
    for key in ("task_status", "status", "protocol_status"):
        if metadata.get(key):
            return str(metadata[key]).lower()
    return "observed"


def _outcome_status_value(status: str) -> float:
    if status in {"success", "succeeded", "completed", "ok"}:
        return 1.0
    if status in {"partial", "partially_successful"}:
        return 0.55
    if status in {"failed", "failure", "error", "timeout", "rejected"}:
        return 0.0
    return 0.50


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _state_summary(state: GeneralDecisionState) -> dict[str, Any]:
    return {
        "active_commitments": [
            {
                "id": item.commitment_id,
                "summary": item.title,
                "priority_hint": item.priority,
                "prep_start_time": item.prep_start_at,
                "start_time": item.start_at,
                "end_time": item.end_at,
            }
            for item in state.commitments
            if item.status == "active"
        ],
        "open_work_items": [
            {
                "id": item.work_item_id,
                "title": item.title,
                "urgency_hint": item.urgency,
                "estimated_minutes_hint": item.estimate_minutes,
                "requires_attention": item.status in {"open", "active"},
            }
            for item in state.work_items
            if item.status in {"open", "active"}
        ],
        "active_conflicts": [
            {
                "type": item.kind,
                "severity": item.severity,
            }
            for item in state.constraints
            if item.status == "active"
        ],
        "executor_available": any(
            item.status == "available" for item in state.capabilities
        ),
    }


def _compare_candidate(
    candidate: GenericCandidate,
    selected_candidate_id: str,
    state: GeneralDecisionState,
) -> dict[str, Any]:
    product = _candidate_product_metadata(candidate)
    return {
        "candidate_id": candidate.candidate_id,
        "title": product.get("user_facing_title") or _title(candidate.action_type),
        "action": candidate.action_type,
        "intent": candidate.intent,
        "recommended_action": product.get("recommended_action", ""),
        "why_now": list(product.get("why_now", [])),
        "expected_result": product.get("expected_result", ""),
        "executor_task": product.get("executor_task", ""),
        "enabled_reason": "; ".join(candidate.why_available),
        "disabled_reason": "; ".join(candidate.why_blocked),
        "requires_confirmation": candidate.requires_confirmation,
        "execution_affordance": _candidate_execution_affordance(candidate),
        "skill_resolution": _candidate_skill_resolution(candidate),
        "key_constraints": [
            str(item.get("description", ""))
            for item in candidate.constraints_triggered
            if item.get("severity") == "veto"
        ],
        "expected_effect": _expected_effect(candidate),
        "simulation": _candidate_simulation(candidate),
        "history": _candidate_history(candidate, state),
        "vetoes": [
            {
                "constraint_id": str(item.get("constraint_id", "")),
                "reason": str(item.get("description", "")),
                "status": "fail",
            }
            for item in candidate.constraints_triggered
            if item.get("severity") == "veto"
        ],
        "tradeoff_rules": [],
        "is_selected": candidate.candidate_id == selected_candidate_id,
    }


def _compare_score_candidate(
    *,
    candidate: GenericCandidate,
    policy_candidate: CandidateDecision,
    score_entry: dict[str, Any],
    veto_events: list[Any],
) -> dict[str, Any]:
    weighted = dict(score_entry.get("weighted_contributions", {}))
    dimensions = []
    for dimension in GENERIC_SCORE_DIMENSIONS:
        raw = dict(weighted.get(dimension, {}))
        dimensions.append(
            {
                "dimension": dimension,
                "label": dimension.replace("_", " ").title(),
                "value": float(raw.get("value", policy_candidate.score_breakdown.get(dimension, 0.0))),
                "weight": float(raw.get("weight", 0.0)),
                "contribution": float(raw.get("contribution", 0.0)),
            }
        )
    return {
        "action": candidate.action_type,
        "score_total": float(score_entry.get("score_total", policy_candidate.score_total)),
        "dimensions": dimensions,
        "constraints": [
            {
                "constraint_id": item["constraint_id"],
                "status": "fail" if candidate.availability_status == "blocked" else "pass",
                "severity": item.get("severity", "unknown"),
                "rule": item.get("description", ""),
                "supported": True,
            }
            for item in candidate.constraints_triggered
        ],
        "vetoes": [
            {
                "constraint_id": str(item.get("constraint_id", "")),
                "reason": str(item.get("reason", "")),
                "status": str(item.get("status", "fail")),
            }
            for item in veto_events
            if isinstance(item, dict) and item.get("candidate_id") == candidate.candidate_id
        ],
        "tradeoff_rules": [],
    }


def _candidate_product_metadata(candidate: GenericCandidate) -> dict[str, Any]:
    metadata = dict(candidate.metadata or {})
    why_now = metadata.get("why_now")
    if isinstance(why_now, str):
        metadata["why_now"] = [why_now]
    elif isinstance(why_now, list):
        metadata["why_now"] = [str(item) for item in why_now if str(item).strip()]
    else:
        metadata["why_now"] = []
    for key in (
        "user_facing_title",
        "recommended_action",
        "expected_result",
        "executor_task",
    ):
        metadata[key] = str(metadata.get(key) or "").strip()
    return metadata


def _selected_basis(
    *,
    selected_candidate_id: str,
    score_entry: dict[str, Any],
    veto_events: list[Any],
    display_language: str = "en",
) -> list[dict[str, Any]]:
    del veto_events
    weighted = dict(score_entry.get("weighted_contributions", {}))
    ranked = []
    for dimension, raw in weighted.items():
        payload = dict(raw)
        ranked.append(
            (
                float(payload.get("contribution", 0.0)),
                {
                    "kind": "weighted_dimension",
                    "evidence_source": "score_breakdown",
                    "candidate_id": selected_candidate_id,
                    "dimension": str(dimension),
                    "label": _dimension_label(str(dimension), display_language),
                    "weight": float(payload.get("weight", 0.0)),
                    "contribution": float(payload.get("contribution", 0.0)),
                    "summary": _weighted_dimension_summary(
                        label=_dimension_label(str(dimension), display_language),
                        weight=float(payload.get("weight", 0.0)),
                        contribution=float(payload.get("contribution", 0.0)),
                        display_language=display_language,
                    ),
                },
            )
        )
    ranked.sort(key=lambda item: item[0], reverse=True)
    basis = [item for _, item in ranked[:2] if item["contribution"] > 0]
    basis.append(
        {
            "kind": "constraint_clear",
            "evidence_source": "constraints",
            "candidate_id": selected_candidate_id,
            "summary": _constraint_clear_summary(display_language),
        }
    )
    return basis


def _why_not(
    *,
    candidate: GenericCandidate,
    selected_candidate: GenericCandidate,
    selected_policy_candidate: CandidateDecision,
    policy_candidate: CandidateDecision,
    selected_score_entry: dict[str, Any],
    candidate_score_entry: dict[str, Any],
    veto_events: list[Any],
    display_language: str = "en",
) -> dict[str, Any]:
    reasons = []
    for item in veto_events:
        if not isinstance(item, dict) or item.get("candidate_id") != candidate.candidate_id:
            continue
        reason = str(item.get("reason", ""))
        if str(item.get("constraint_id", "")) == SELECTION_POOL_CONSTRAINT_ID:
            selection_pool = dict(policy_candidate.params.get("selection_pool", {}) or {})
            reason = str(selection_pool.get("reason") or reason)
        reasons.append(
            {
                "kind": "veto",
                "evidence_source": "veto_events",
                "constraint_id": str(item.get("constraint_id", "")),
                "reason": _localized_veto_reason(reason, display_language),
            }
        )
    if not reasons:
        selected_weighted = dict(selected_score_entry.get("weighted_contributions", {}))
        candidate_weighted = dict(candidate_score_entry.get("weighted_contributions", {}))
        ranked = []
        for dimension in GENERIC_SCORE_DIMENSIONS:
            selected_raw = dict(selected_weighted.get(dimension, {}))
            candidate_raw = dict(candidate_weighted.get(dimension, {}))
            selected_contribution = float(
                selected_raw.get(
                    "contribution",
                    selected_policy_candidate.score_breakdown.get(dimension, 0.0),
                )
            )
            candidate_contribution = float(
                candidate_raw.get(
                    "contribution",
                    policy_candidate.score_breakdown.get(dimension, 0.0),
                )
            )
            delta = selected_contribution - candidate_contribution
            if delta > 0:
                ranked.append(
                    (
                        delta,
                        {
                            "kind": "weighted_dimension_gap",
                            "evidence_source": "score_breakdown",
                            "dimension": dimension,
                            "label": _dimension_label(dimension, display_language),
                            "selected_contribution": selected_contribution,
                            "candidate_contribution": candidate_contribution,
                            "weight": float(selected_raw.get("weight", 0.0)),
                            "summary": _weighted_dimension_gap_summary(
                                label=_dimension_label(dimension, display_language),
                                selected_contribution=selected_contribution,
                                candidate_contribution=candidate_contribution,
                                display_language=display_language,
                            ),
                        },
                    )
                )
        ranked.sort(key=lambda item: item[0], reverse=True)
        reasons.extend(item for _, item in ranked[:2])
    return {
        "candidate_id": candidate.candidate_id,
        "title": _candidate_product_metadata(candidate).get("user_facing_title")
        or _title(candidate.action_type),
        "reasons": reasons,
    }


def _candidate_execution_affordance(candidate: GenericCandidate) -> dict[str, Any]:
    value = candidate.metadata.get("execution_affordance")
    return dict(value) if isinstance(value, dict) else {}


def _candidate_skill_resolution(candidate: GenericCandidate) -> dict[str, Any]:
    value = candidate.metadata.get("skill_resolution")
    return dict(value) if isinstance(value, dict) else {}


def _dimension_label(dimension: str, display_language: str = "en") -> str:
    if display_language != "zh":
        return dimension.replace("_", " ").title()
    labels = {
        "outcome_value": "结果价值",
        "risk_reduction": "风险降低",
        "reversibility": "可逆性",
        "confidence_alignment": "置信度匹配",
        "urgency_alignment": "紧急度匹配",
        "effort_fit": "工作量匹配",
        "impact_potential": "影响潜力",
        "historical_outcome_alignment": "历史结果匹配",
        "execution_intent_fit": "执行意图匹配",
        "preference_alignment": "偏好匹配",
    }
    return labels.get(dimension, dimension.replace("_", " ").title())


def _constraint_clear_summary(display_language: str = "en") -> str:
    if display_language == "zh":
        return "没有记录到会阻止该候选的可用性约束。"
    return "No declared candidate availability block was recorded."


def _weighted_dimension_summary(
    *,
    label: str,
    weight: float,
    contribution: float,
    display_language: str = "en",
) -> str:
    if display_language == "zh":
        return f"{label}权重为 {weight:.2f}，对当前选择的贡献为 {contribution:.2f}。"
    return f"{label} carried weight {weight:.2f} with contribution {contribution:.2f}."


def _weighted_dimension_gap_summary(
    *,
    label: str,
    selected_contribution: float,
    candidate_contribution: float,
    display_language: str = "en",
) -> str:
    if display_language == "zh":
        return (
            f"{label}贡献更低"
            f"（{candidate_contribution:.2f} vs {selected_contribution:.2f}）。"
        )
    return (
        f"lower {label} contribution "
        f"({candidate_contribution:.2f} vs {selected_contribution:.2f})."
    )


def _localized_veto_reason(reason: str, display_language: str = "en") -> str:
    if display_language != "zh":
        return reason
    known = {
        "when runtime mode restricts the selection pool, do not select candidates outside that pool": (
            "当前运行模式限制了可选择范围，因此不能选择执行池之外的候选。"
        ),
        "do not select candidates outside the active runtime selection pool": (
            "当前运行模式限制了可选择范围，因此不能选择执行池之外的候选。"
        ),
        "Candidate is visible for comparison but excluded from the active selection pool.": (
            "当前运行模式限制了可选择范围，因此不能选择执行池之外的候选。"
        ),
    }
    return known.get(reason, reason)


def _expected_effect(candidate: GenericCandidate) -> dict[str, Any]:
    return {
        "expected_time_cost_minutes": candidate.estimated_cost.time_minutes,
        "commitment_risk": candidate.risk_profile.level,
        "work_item_risk_change": "",
        "followup_needed": bool(candidate.expected_state_delta.updates_refs),
        "followup_summary": candidate.expected_state_delta.summary,
        "reversibility": candidate.reversibility,
        "attention_cost": candidate.estimated_cost.attention,
    }


def _candidate_simulation(candidate: GenericCandidate) -> dict[str, Any]:
    simulation = candidate.metadata.get("llm_simulation")
    if isinstance(simulation, dict):
        return payload_value(simulation)
    return {}


def _compare_execution_boundary(boundary: ExecutionBoundary) -> dict[str, Any]:
    return {
        "selected_action": boundary.mode,
        "requires_confirmation": boundary.requires_confirmation,
        "execution_path": boundary.mode,
        "executor": "",
        "note": "Execution is downstream of generic policy selection and is not performed by this adapter.",
    }


def _candidate_trace_status(candidate: GenericCandidate, selected_candidate_id: str) -> str:
    if candidate.candidate_id == selected_candidate_id:
        return "selected"
    if candidate.availability_status == "blocked":
        return "blocked"
    return "considered"


def _generic_candidate_by_id(
    candidates: list[GenericCandidate],
    candidate_id: str,
) -> GenericCandidate:
    for candidate in candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    raise ValueError(f"selected candidate not found: {candidate_id}")


def _stable_ref(prefix: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"{prefix}.{sha256(encoded.encode('utf-8')).hexdigest()[:12]}"


def _ref_slug(value: str) -> str:
    return value.replace(".", "_").replace("/", "_")


def _title(action_type: str) -> str:
    return action_type.replace(".", " ").replace("_", " ").title()
