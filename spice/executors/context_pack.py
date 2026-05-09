from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from spice.decision.general.candidates import GenericCandidate
from spice.decision.general.state import GeneralDecisionState
from spice.executors.skills import ResolvedSkill, payload_value, safe_dataclass_from_payload


CONTEXT_PACK_SCHEMA_VERSION = "0.1"
CONTEXT_PACK_BUILDER = "spice.executors.context_pack"


@dataclass(slots=True)
class ExecutionContextPack:
    context_pack_id: str
    schema_version: str = CONTEXT_PACK_SCHEMA_VERSION
    generated_by: str = CONTEXT_PACK_BUILDER
    state_ref: str = ""
    decision_id: str = ""
    trace_ref: str = ""
    candidate_id: str = ""
    approval_id: str = ""
    execution_id: str = ""
    request_id: str = ""
    skill_id: str = ""
    executor_id: str = ""
    action_type: str = ""
    target_refs: list[str] = field(default_factory=list)
    objective: str = ""
    task: str = ""
    why_now: str = ""
    do_not: list[str] = field(default_factory=list)
    expected_output: str = ""
    return_schema: dict[str, Any] = field(default_factory=dict)
    selected_candidate: dict[str, Any] = field(default_factory=dict)
    resolved_skill: dict[str, Any] = field(default_factory=dict)
    state_summary: dict[str, Any] = field(default_factory=dict)
    relevant_state: dict[str, Any] = field(default_factory=dict)
    constraints: list[dict[str, Any]] = field(default_factory=list)
    risks: list[dict[str, Any]] = field(default_factory=list)
    recent_outcomes: list[dict[str, Any]] = field(default_factory=list)
    instructions: list[str] = field(default_factory=list)
    traceability: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ExecutionContextPack":
        item = safe_dataclass_from_payload(cls, payload)
        item.target_refs = _string_list(payload.get("target_refs"))
        item.do_not = _string_list(payload.get("do_not"))
        item.return_schema = _dict(payload.get("return_schema"))
        item.selected_candidate = _dict(payload.get("selected_candidate"))
        item.resolved_skill = _dict(payload.get("resolved_skill"))
        item.state_summary = _dict(payload.get("state_summary"))
        item.relevant_state = _dict(payload.get("relevant_state"))
        item.constraints = _dict_list(payload.get("constraints"))
        item.risks = _dict_list(payload.get("risks"))
        item.recent_outcomes = _dict_list(payload.get("recent_outcomes"))
        item.instructions = _string_list(payload.get("instructions"))
        item.traceability = _dict(payload.get("traceability"))
        item.metadata = _dict(payload.get("metadata"))
        item.validate()
        return item

    def validate(self) -> None:
        _require_non_empty(self.context_pack_id, "context_pack_id")
        _require_non_empty(self.candidate_id, "candidate_id")
        _require_non_empty(self.skill_id, "skill_id")
        _require_non_empty(self.action_type, "action_type")
        if self.resolved_skill and self.resolved_skill.get("skill_id") != self.skill_id:
            raise ValueError("context_pack.resolved_skill.skill_id must match skill_id")
        if self.selected_candidate and self.selected_candidate.get("candidate_id") != self.candidate_id:
            raise ValueError("context_pack.selected_candidate.candidate_id must match candidate_id")


def build_execution_context_pack(
    *,
    state: GeneralDecisionState,
    candidate: GenericCandidate,
    resolved_skill: ResolvedSkill,
    decision_id: str = "",
    trace_ref: str = "",
    approval_id: str = "",
    execution_id: str = "",
    request_id: str = "",
    max_items_per_section: int = 5,
    metadata: dict[str, Any] | None = None,
) -> ExecutionContextPack:
    """Build a compact executor context pack without planning or executing.

    The pack contains only decision-relevant state around candidate target refs.
    It does not read external memory, generate SDEP payloads, call executors, or
    mutate GeneralDecisionState.
    """

    if not isinstance(state, GeneralDecisionState):
        raise ValueError("state must be a GeneralDecisionState")
    if not isinstance(candidate, GenericCandidate):
        raise ValueError("candidate must be a GenericCandidate")
    if not isinstance(resolved_skill, ResolvedSkill):
        raise ValueError("resolved_skill must be a ResolvedSkill")
    resolved_skill.validate()
    _validate_resolution(candidate, resolved_skill)

    target_refs = _stable_refs(
        candidate.target_refs
        + candidate.expected_state_delta.creates_refs
        + candidate.expected_state_delta.updates_refs
        + candidate.expected_state_delta.closes_refs
        + _string_list(resolved_skill.metadata.get("target_refs"))
    )
    context_pack_id = _context_pack_id(
        state_ref=state.state_id,
        decision_id=decision_id,
        trace_ref=trace_ref,
        candidate_id=candidate.candidate_id,
        approval_id=approval_id,
        execution_id=execution_id,
        request_id=request_id,
        skill_id=resolved_skill.skill_id,
        executor_id=resolved_skill.executor_id,
        target_refs=target_refs,
    )
    relevant_state = _relevant_state_payload(
        state=state,
        refs=target_refs,
        candidate=candidate,
        max_items_per_section=max_items_per_section,
    )
    traceability = {
        "state_ref": state.state_id,
        "decision_id": decision_id,
        "trace_ref": trace_ref,
        "candidate_id": candidate.candidate_id,
        "approval_id": approval_id,
        "execution_id": execution_id,
        "request_id": request_id,
        "skill_id": resolved_skill.skill_id,
        "executor_id": resolved_skill.executor_id,
        "target_refs": list(target_refs),
    }
    pack = ExecutionContextPack(
        context_pack_id=context_pack_id,
        state_ref=state.state_id,
        decision_id=decision_id,
        trace_ref=trace_ref,
        candidate_id=candidate.candidate_id,
        approval_id=approval_id,
        execution_id=execution_id,
        request_id=request_id,
        skill_id=resolved_skill.skill_id,
        executor_id=resolved_skill.executor_id,
        action_type=candidate.action_type,
        target_refs=target_refs,
        objective=candidate.intent,
        task=_task(candidate, resolved_skill),
        why_now=_why_now(candidate),
        do_not=_do_not(candidate, relevant_state),
        expected_output=_expected_output(resolved_skill),
        return_schema=dict(resolved_skill.output_schema),
        selected_candidate=_compact_candidate_payload(candidate),
        resolved_skill=resolved_skill.to_payload(),
        state_summary=_state_summary(state),
        relevant_state=relevant_state,
        constraints=list(relevant_state["constraints"]),
        risks=list(relevant_state["risks"]),
        recent_outcomes=list(relevant_state["outcomes"]),
        instructions=list(resolved_skill.instructions),
        traceability=traceability,
        metadata={
            "max_items_per_section": max_items_per_section,
            "context_is_compact": True,
            "external_memory_loaded": False,
            **dict(metadata or {}),
        },
    )
    pack.validate()
    return pack


def _validate_resolution(candidate: GenericCandidate, resolved_skill: ResolvedSkill) -> None:
    if resolved_skill.action_type != candidate.action_type:
        raise ValueError("resolved_skill.action_type must match candidate.action_type")
    metadata_candidate_id = resolved_skill.metadata.get("candidate_id")
    if metadata_candidate_id and metadata_candidate_id != candidate.candidate_id:
        raise ValueError("resolved_skill.metadata.candidate_id must match candidate.candidate_id")


def _relevant_state_payload(
    *,
    state: GeneralDecisionState,
    refs: list[str],
    candidate: GenericCandidate,
    max_items_per_section: int,
) -> dict[str, list[dict[str, Any]]]:
    ref_set = set(refs)
    return {
        "signals": _limited(
            [item.to_payload() for item in state.signals if _signal_matches(item, ref_set)],
            max_items_per_section,
        ),
        "observations": _limited(
            [item.to_payload() for item in state.observations if _observation_matches(item, ref_set)],
            max_items_per_section,
        ),
        "intents": _limited(
            [item.to_payload() for item in state.intents if _record_matches(item.intent_id, item.target_refs, ref_set)],
            max_items_per_section,
        ),
        "commitments": _limited(
            [item.to_payload() for item in state.commitments if item.status == "active"],
            max_items_per_section,
        ),
        "work_items": _limited(
            [
                item.to_payload()
                for item in state.work_items
                if _record_matches(
                    item.work_item_id,
                    item.source_refs + item.blocker_refs,
                    ref_set,
                )
            ],
            max_items_per_section,
        ),
        "resources": _limited(
            [item.to_payload() for item in state.resources if item.resource_id in ref_set],
            max_items_per_section,
        ),
        "capabilities": _limited(
            [
                item.to_payload()
                for item in state.capabilities
                if item.capability_id in {candidate.required_capability, candidate.execution_boundary.required_capability}
            ],
            max_items_per_section,
        ),
        "constraints": _limited(
            _constraint_payloads(state, candidate, ref_set),
            max_items_per_section,
        ),
        "risks": _limited(
            [
                item.to_payload()
                for item in state.risks
                if _record_matches(item.risk_id, item.applies_to_refs + item.mitigation_refs, ref_set)
            ],
            max_items_per_section,
        ),
        "open_loops": _limited(
            [
                item.to_payload()
                for item in state.open_loops
                if _record_matches(item.open_loop_id, item.source_refs + item.target_refs, ref_set)
            ],
            max_items_per_section,
        ),
        "outcomes": _limited(
            [
                item.to_payload()
                for item in state.outcomes
                if item.decision_id == candidate.metadata.get("decision_id")
                or item.candidate_id == candidate.candidate_id
                or item.execution_ref in ref_set
            ],
            max_items_per_section,
        ),
    }


def _constraint_payloads(
    state: GeneralDecisionState,
    candidate: GenericCandidate,
    ref_set: set[str],
) -> list[dict[str, Any]]:
    triggered_ids = {
        str(item.get("constraint_id"))
        for item in candidate.constraints_triggered
        if item.get("constraint_id")
    }
    payloads = [
        item.to_payload()
        for item in state.constraints
        if item.constraint_id in triggered_ids
        or _record_matches(item.constraint_id, item.applies_to_refs, ref_set)
    ]
    known = {item.get("constraint_id") for item in payloads}
    for item in candidate.constraints_triggered:
        if item.get("constraint_id") not in known:
            payloads.append(dict(item))
    return payloads


def _compact_candidate_payload(candidate: GenericCandidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "action_type": candidate.action_type,
        "intent": candidate.intent,
        "target_refs": list(candidate.target_refs),
        "required_capability": candidate.required_capability,
        "requires_confirmation": candidate.requires_confirmation,
        "availability_status": candidate.availability_status,
        "side_effect_class": candidate.side_effect_class,
        "expected_state_delta": candidate.expected_state_delta.to_payload(),
        "execution_boundary": candidate.execution_boundary.to_payload(),
    }


def _task(candidate: GenericCandidate, resolved_skill: ResolvedSkill) -> str:
    if candidate.intent:
        return candidate.intent
    if resolved_skill.resolution_reason:
        return resolved_skill.resolution_reason
    return f"Handle {candidate.action_type}."


def _why_now(candidate: GenericCandidate) -> str:
    if candidate.why_available:
        return candidate.why_available[0]
    if candidate.expected_state_delta.summary:
        return candidate.expected_state_delta.summary
    return "Selected by the current decision checkpoint."


def _do_not(candidate: GenericCandidate, relevant_state: dict[str, list[dict[str, Any]]]) -> list[str]:
    items: list[str] = []
    for constraint in relevant_state.get("constraints", []):
        description = constraint.get("description") or constraint.get("constraint_id")
        if description:
            items.append(str(description))
    if candidate.requires_confirmation:
        items.append("Do not execute without the recorded approval checkpoint.")
    if candidate.execution_boundary.side_effect_class:
        items.append(f"Do not exceed side_effect_class={candidate.execution_boundary.side_effect_class}.")
    return list(dict.fromkeys(items)) or ["Do not perform actions outside the selected candidate."]


def _expected_output(resolved_skill: ResolvedSkill) -> str:
    schema_type = resolved_skill.output_schema.get("type")
    if schema_type:
        return f"Return output matching {schema_type}."
    return "Return a concise structured execution outcome."


def _state_summary(state: GeneralDecisionState) -> dict[str, Any]:
    return {
        "state_id": state.state_id,
        "observation_count": len(state.observations),
        "intent_count": len(state.intents),
        "commitment_count": len(state.commitments),
        "work_item_count": len(state.work_items),
        "resource_count": len(state.resources),
        "capability_count": len(state.capabilities),
        "constraint_count": len(state.constraints),
        "risk_count": len(state.risks),
        "open_loop_count": len(state.open_loops),
        "outcome_count": len(state.outcomes),
    }


def _signal_matches(signal: Any, ref_set: set[str]) -> bool:
    return bool(
        signal.signal_id in ref_set
        or signal.subject_ref in ref_set
        or ref_set.intersection(signal.refs)
    )


def _observation_matches(observation: Any, ref_set: set[str]) -> bool:
    subject_id = observation.subject.subject_id if observation.subject else ""
    return bool(
        observation.observation_id in ref_set
        or subject_id in ref_set
        or ref_set.intersection(observation.refs)
    )


def _record_matches(record_id: str, refs: list[str], ref_set: set[str]) -> bool:
    return bool(record_id in ref_set or ref_set.intersection(refs))


def _limited(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    return items[:limit]


def _context_pack_id(
    *,
    state_ref: str,
    decision_id: str,
    trace_ref: str,
    candidate_id: str,
    approval_id: str,
    execution_id: str,
    request_id: str,
    skill_id: str,
    executor_id: str,
    target_refs: list[str],
) -> str:
    seed = "|".join(
        [
            state_ref,
            decision_id,
            trace_ref,
            candidate_id,
            approval_id,
            execution_id,
            request_id,
            skill_id,
            executor_id,
            ",".join(target_refs),
        ]
    )
    return f"context_pack.{sha256(seed.encode('utf-8')).hexdigest()[:16]}"


def _stable_refs(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if isinstance(value, str) and value))


def _require_non_empty(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"context_pack.{field_name} is required")


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item.strip()]
