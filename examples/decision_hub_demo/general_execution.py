from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

from spice.decision.general import GenericCandidate
from spice.decision.general.approval import Approval
from spice.decision.general.types import payload_value, safe_dataclass_from_payload
from spice.executors import (
    CapabilityDescriptor,
    ExecutorDescriptor,
    SkillCatalog,
    SkillDescriptor,
    build_execution_context_pack,
    builtin_fallback_skill_catalog,
    resolve_skill_for_candidate,
)
from spice.executors.sdep_mapping import build_sdep_execute_request
from spice.protocols.execution import ExecutionIntent

from examples.decision_hub_demo.general_adapter import GeneralDecisionHubResult
from examples.decision_hub_demo.general_approval import (
    GeneralApprovalBridgeResult,
    build_general_approval_bridge,
)
from examples.decision_hub_demo.ids import timestamp_segment


GENERAL_EXECUTION_ADAPTER = "decision_hub_demo.general_execution_planner"
GENERAL_SDEP_ACTION_PREFIX = "spice.general"


@dataclass(slots=True)
class GeneralExecutionPlanResult:
    """Read-only execution planning result for a General Core decision."""

    status: str
    decision_id: str
    trace_ref: str
    selected_candidate_id: str
    approval_id: str | None
    selected_candidate: GenericCandidate
    execution_intent: ExecutionIntent | None = None
    sdep_request: dict[str, Any] | None = None
    resolved_skill: dict[str, Any] | None = None
    context_pack: dict[str, Any] | None = None
    reason: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "decision_id": self.decision_id,
            "trace_ref": self.trace_ref,
            "selected_candidate_id": self.selected_candidate_id,
            "approval_id": self.approval_id,
            "selected_candidate": self.selected_candidate.to_payload(),
            "execution_allowed": self.execution_intent is not None,
            "executed": False,
            "execution": None,
            "state_updated": False,
            "execution_intent": _execution_intent_payload(self.execution_intent)
            if self.execution_intent
            else None,
            "sdep_request": payload_value(self.sdep_request),
            "resolved_skill": payload_value(self.resolved_skill),
            "context_pack": payload_value(self.context_pack),
            "reason": self.reason,
        }


def approve_general_approval(
    approval: Approval | dict[str, Any],
    *,
    now: datetime | None = None,
    actor: str = "user",
    response: str = "confirm",
) -> Approval:
    """Return an approved copy of a General approval checkpoint.

    This helper is deterministic and does not write a confirmation store or
    trigger execution. It is only a typed input for the planning adapter.
    """

    parsed = _approval_from_payload(approval)
    resolved = now or datetime.now(timezone.utc).replace(microsecond=0)
    return Approval(
        approval_id=parsed.approval_id,
        decision_id=parsed.decision_id,
        candidate_id=parsed.candidate_id,
        status="approved",
        mode=parsed.mode,
        requested_at=parsed.requested_at,
        resolved_at=timestamp_segment(resolved),
        actor=actor,
        prompt=parsed.prompt,
        response=response,
        reason=parsed.reason,
        execution_allowed=True,
        metadata=dict(parsed.metadata),
    )


def build_general_execution_plan(
    result: GeneralDecisionHubResult,
    *,
    approval: Approval | dict[str, Any] | None = None,
    skill_catalog: SkillCatalog | None = None,
    now: datetime | None = None,
) -> GeneralExecutionPlanResult:
    """Convert an approved General decision into a planned SDEP handoff.

    This function creates request payloads only. It does not call an executor,
    send SDEP over transport, produce an outcome, or update state.
    """

    created = now or datetime.now(timezone.utc).replace(microsecond=0)
    bridge = build_general_approval_bridge(result, now=created)
    selected_candidate = bridge.selected_candidate
    resolved_approval = _resolve_approval(bridge, approval)

    if selected_candidate.requires_confirmation and not _is_approved(resolved_approval):
        return GeneralExecutionPlanResult(
            status="approval_required",
            decision_id=bridge.decision_id,
            trace_ref=bridge.trace_ref,
            selected_candidate_id=bridge.selected_candidate_id,
            approval_id=resolved_approval.approval_id if resolved_approval else None,
            selected_candidate=selected_candidate,
            reason="approval is required before creating an execution plan",
        )

    resolution = resolve_skill_for_candidate(
        selected_candidate,
        skill_catalog or _default_skill_catalog(result),
    )
    if resolution.status != "resolved" or resolution.resolved_skill is None:
        return GeneralExecutionPlanResult(
            status="skill_unresolved",
            decision_id=bridge.decision_id,
            trace_ref=bridge.trace_ref,
            selected_candidate_id=bridge.selected_candidate_id,
            approval_id=resolved_approval.approval_id if resolved_approval else None,
            selected_candidate=selected_candidate,
            reason="; ".join(resolution.unresolved_reasons),
            resolved_skill=None,
            context_pack=None,
        )

    execution_intent = _execution_intent_from_candidate(
        result=result,
        bridge=bridge,
        approval=resolved_approval,
        selected_candidate=selected_candidate,
        resolved_skill=resolution.resolved_skill,
        now=created,
    )
    sdep_request = _sdep_request_payload(execution_intent, now=created)
    return GeneralExecutionPlanResult(
        status="planned",
        decision_id=bridge.decision_id,
        trace_ref=bridge.trace_ref,
        selected_candidate_id=bridge.selected_candidate_id,
        approval_id=resolved_approval.approval_id if resolved_approval else None,
        selected_candidate=selected_candidate,
        execution_intent=execution_intent,
        sdep_request=sdep_request,
        resolved_skill=resolution.resolved_skill.to_payload(),
        context_pack=payload_value(execution_intent.input_payload.get("context_pack")),
        reason="execution plan created but not executed",
    )


def build_general_execution_artifact(
    result: GeneralDecisionHubResult,
    *,
    approval: Approval | dict[str, Any] | None = None,
    skill_catalog: SkillCatalog | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    created = now or datetime.now(timezone.utc).replace(microsecond=0)
    plan = build_general_execution_plan(
        result,
        approval=approval,
        skill_catalog=skill_catalog,
        now=created,
    )
    plan_payload = plan.to_payload()
    execution_intent = plan_payload["execution_intent"]
    execution_allowed = bool(plan_payload["execution_allowed"])
    context_pack = plan_payload["context_pack"]
    resolved_skill = plan_payload["resolved_skill"]
    return {
        "path_type": "read_only_general_execution_plan",
        "generated_by": "general_execution_planner",
        "decision_id": plan.decision_id,
        "trace_ref": plan.trace_ref,
        "candidate_id": plan.selected_candidate_id,
        "selected_candidate_id": plan.selected_candidate_id,
        "approval_id": plan.approval_id,
        "skill_resolution_status": "resolved" if resolved_skill else "unresolved",
        "skill_id": resolved_skill.get("skill_id") if isinstance(resolved_skill, dict) else None,
        "executor_id": resolved_skill.get("executor_id") if isinstance(resolved_skill, dict) else None,
        "context_pack_id": context_pack.get("context_pack_id") if isinstance(context_pack, dict) else None,
        "status": plan.status,
        "approved": execution_allowed,
        "execution_status": _execution_status(plan.status, execution_allowed),
        "execution_id": execution_intent["id"] if execution_intent else None,
        "execution_allowed": execution_allowed,
        "executed": False,
        "execution": None,
        "outcome": None,
        "state_updated": False,
        "created_at": timestamp_segment(created),
        "resolved_skill": resolved_skill,
        "context_pack": context_pack,
        "execution_intent": execution_intent,
        "sdep_request": plan_payload["sdep_request"],
        "execution_plan": plan_payload,
    }


def _execution_intent_from_candidate(
    *,
    result: GeneralDecisionHubResult,
    bridge: GeneralApprovalBridgeResult,
    approval: Approval | None,
    selected_candidate: GenericCandidate,
    resolved_skill: Any,
    now: datetime,
) -> ExecutionIntent:
    acted_on = selected_candidate.target_refs[0] if selected_candidate.target_refs else None
    execution_id = _make_general_execution_id(
        now=now,
        decision_id=bridge.decision_id,
        trace_ref=bridge.trace_ref,
        candidate_id=selected_candidate.candidate_id,
        approval_id=approval.approval_id if approval else None,
        action_type=selected_candidate.action_type,
        target_refs=selected_candidate.target_refs,
    )
    operation_name = f"{GENERAL_SDEP_ACTION_PREFIX}.{selected_candidate.action_type}"
    request_id = f"sdep-req.general.{_hash(execution_id)}"
    context_pack = build_execution_context_pack(
        state=result.state,
        candidate=selected_candidate,
        resolved_skill=resolved_skill,
        decision_id=bridge.decision_id,
        trace_ref=bridge.trace_ref,
        approval_id=approval.approval_id if approval else "",
        execution_id=execution_id,
        request_id=request_id,
    )
    return ExecutionIntent(
        id=execution_id,
        timestamp=now,
        refs=_refs(
            execution_id=execution_id,
            decision_id=bridge.decision_id,
            trace_ref=bridge.trace_ref,
            candidate_id=selected_candidate.candidate_id,
            target_refs=selected_candidate.target_refs,
            approval_id=approval.approval_id if approval else None,
        ),
        intent_type="general.execution_request",
        status="planned",
        objective={
            "id": bridge.decision_id,
            "description": selected_candidate.intent,
        },
        executor_type=_executor_type(selected_candidate),
        target=_target_payload(selected_candidate),
        operation={
            "name": operation_name,
            "mode": _execution_mode(selected_candidate),
            "dry_run": False,
        },
        input_payload={
            "decision_id": bridge.decision_id,
            "trace_ref": bridge.trace_ref,
            "candidate_id": selected_candidate.candidate_id,
            "approval_id": approval.approval_id if approval else None,
            "selected_action": selected_candidate.action_type,
            "general_action_type": selected_candidate.action_type,
            "skill_id": resolved_skill.skill_id,
            "executor_id": resolved_skill.executor_id,
            "skill_hint": {
                "skill_id": resolved_skill.skill_id,
                "executor_id": resolved_skill.executor_id,
                "skill_source": resolved_skill.metadata.get("skill_source"),
                "side_effect_class": resolved_skill.side_effect_class,
                "context_pack_id": context_pack.context_pack_id,
            },
            "context_pack_id": context_pack.context_pack_id,
            "context_pack": context_pack.to_payload(),
        },
        parameters={
            "resolved_skill": resolved_skill.to_payload(),
            "required_capability": selected_candidate.required_capability,
            "side_effect_class": selected_candidate.side_effect_class,
            "estimated_cost": selected_candidate.estimated_cost.to_payload(),
            "risk_profile": selected_candidate.risk_profile.to_payload(),
            "reversibility": selected_candidate.reversibility,
            "expected_state_delta": selected_candidate.expected_state_delta.to_payload(),
            "execution_boundary": selected_candidate.execution_boundary.to_payload(),
            "why_available": list(selected_candidate.why_available),
        },
        constraints=list(selected_candidate.constraints_triggered),
        success_criteria=[
            {
                "description": selected_candidate.expected_state_delta.summary
                or f"Complete {selected_candidate.action_type}.",
                "source": "generic_candidate.expected_state_delta",
            }
        ],
        failure_policy={"strategy": "fail_fast", "max_retries": 0},
        provenance={
            "adapter": GENERAL_EXECUTION_ADAPTER,
            "spice_decision_id": bridge.decision_id,
            "trace_ref": bridge.trace_ref,
            "candidate_id": selected_candidate.candidate_id,
            "approval_id": approval.approval_id if approval else None,
            "skill_id": resolved_skill.skill_id,
            "executor_id": resolved_skill.executor_id,
            "context_pack_id": context_pack.context_pack_id,
            "created_at": timestamp_segment(now),
        },
        metadata={
            "selected_candidate": selected_candidate.to_payload(),
            "approval": approval.to_payload() if approval else None,
            "resolved_skill": resolved_skill.to_payload(),
            "context_pack": context_pack.to_payload(),
            "planning_only": True,
        },
    )


def _sdep_request_payload(intent: ExecutionIntent, *, now: datetime) -> dict[str, Any]:
    request_id = f"sdep-req.general.{_hash(intent.id)}"
    request = build_sdep_execute_request(
        intent,
        request_id=request_id,
        metadata={
            "runtime": "spice",
            "adapter": GENERAL_EXECUTION_ADAPTER,
            "demo_domain": "decision_hub_demo",
            "planning_only": True,
        },
    )
    request.message_id = f"sdep-msg.general.{_hash(intent.id)}"
    request.timestamp = now
    request.traceability.update(
        {
            "execution_id": intent.id,
            "spice_decision_id": str(intent.provenance.get("spice_decision_id", "")),
            "trace_ref": str(intent.provenance.get("trace_ref", "")),
            "candidate_id": str(intent.provenance.get("candidate_id", "")),
            "approval_id": intent.provenance.get("approval_id"),
        }
    )
    request.execution.metadata.update(
        {
            "adapter": GENERAL_EXECUTION_ADAPTER,
            "general_action_type": str(intent.input_payload.get("general_action_type", "")),
            "skill_id": str(intent.input_payload.get("skill_id", "")),
            "executor_id": str(intent.input_payload.get("executor_id", "")),
            "context_pack_id": str(intent.input_payload.get("context_pack_id", "")),
            "planning_only": True,
        }
    )
    request.validate()
    return request.to_dict()


def _resolve_approval(
    bridge: GeneralApprovalBridgeResult,
    approval: Approval | dict[str, Any] | None,
) -> Approval | None:
    if approval is not None:
        parsed = _approval_from_payload(approval)
        _validate_approval_attribution(bridge, parsed)
        return parsed
    return bridge.approval


def _validate_approval_attribution(
    bridge: GeneralApprovalBridgeResult,
    approval: Approval,
) -> None:
    if not _is_approved(approval):
        raise ValueError(
            "approval must be approved with execution_allowed=true before planning execution"
        )
    if approval.decision_id != bridge.decision_id:
        raise ValueError(
            f"approval decision_id mismatch: expected {bridge.decision_id!r}, "
            f"got {approval.decision_id!r}"
        )
    trace_ref = str(approval.metadata.get("trace_ref", "")).strip()
    if trace_ref and trace_ref != bridge.trace_ref:
        raise ValueError(
            f"approval trace_ref mismatch: expected {bridge.trace_ref!r}, got {trace_ref!r}"
        )
    if approval.candidate_id != bridge.selected_candidate_id:
        raise ValueError(
            f"approval candidate_id mismatch: expected {bridge.selected_candidate_id!r}, "
            f"got {approval.candidate_id!r}"
        )

    expected_approval_id = ""
    if bridge.approval is not None:
        expected_approval_id = bridge.approval.approval_id
    if not expected_approval_id and isinstance(bridge.confirmation_request, dict):
        expected_approval_id = str(bridge.confirmation_request.get("approval_id", ""))
    if expected_approval_id and approval.approval_id != expected_approval_id:
        raise ValueError(
            f"approval_id mismatch: expected {expected_approval_id!r}, "
            f"got {approval.approval_id!r}"
        )


def _approval_from_payload(value: Approval | dict[str, Any]) -> Approval:
    if isinstance(value, Approval):
        return value
    if isinstance(value, dict):
        return safe_dataclass_from_payload(Approval, value)
    raise ValueError("approval must be an Approval or payload dict")


def _is_approved(approval: Approval | None) -> bool:
    return (
        approval is not None
        and approval.status in {"approved", "confirmed"}
        and approval.execution_allowed
    )


def _execution_intent_payload(intent: ExecutionIntent | None) -> dict[str, Any] | None:
    if intent is None:
        return None
    return {
        "id": intent.id,
        "timestamp": intent.timestamp.isoformat(),
        "refs": list(intent.refs),
        "metadata": payload_value(intent.metadata),
        "intent_type": intent.intent_type,
        "status": intent.status,
        "objective": payload_value(intent.objective),
        "executor_type": intent.executor_type,
        "target": payload_value(intent.target),
        "operation": payload_value(intent.operation),
        "input_payload": payload_value(intent.input_payload),
        "parameters": payload_value(intent.parameters),
        "constraints": payload_value(intent.constraints),
        "success_criteria": payload_value(intent.success_criteria),
        "failure_policy": payload_value(intent.failure_policy),
        "provenance": payload_value(intent.provenance),
    }


def _target_payload(candidate: GenericCandidate) -> dict[str, Any]:
    target_id = candidate.target_refs[0] if candidate.target_refs else candidate.candidate_id
    return {
        "kind": "general_reference",
        "id": target_id,
        "refs": list(candidate.target_refs),
    }


def _executor_type(candidate: GenericCandidate) -> str:
    return (
        candidate.execution_boundary.required_capability
        or candidate.required_capability
        or "external-agent"
    )


def _execution_mode(candidate: GenericCandidate) -> str:
    mode = candidate.execution_boundary.mode.strip() if candidate.execution_boundary.mode else ""
    if not mode or mode == "none":
        return "sync"
    return mode


def _execution_status(status: str, execution_allowed: bool) -> str:
    if execution_allowed:
        return "planned_not_executed"
    if status == "skill_unresolved":
        return "skill_unresolved"
    return "approval_required"


def _default_skill_catalog(result: GeneralDecisionHubResult) -> SkillCatalog:
    catalog = builtin_fallback_skill_catalog()
    executors = [
        _executor_from_capability(capability)
        for capability in result.state.capabilities
        if capability.status == "available"
    ]
    generic_executor = ExecutorDescriptor(
        executor_id="spice.general_executor",
        display_name="Spice General Executor Descriptor",
        description="Read-only descriptor used for planned General execution handoffs.",
        capabilities=[
            CapabilityDescriptor(capability_id="work_item_triage"),
            CapabilityDescriptor(capability_id="general_state_change"),
        ],
        skills=[
            SkillDescriptor(
                skill_id="general.item.triage",
                source="executor",
                supported_action_types=["item.triage", "context.prepare"],
                required_capabilities=["work_item_triage"],
                side_effect_class="state_change",
                requires_confirmation=False,
                input_schema={"type": "context_pack.v1"},
                output_schema={"type": "triage_report.v1"},
                instructions=[
                    "Use the provided context pack only.",
                    "Return status, risk change, follow-up need, and a concise summary.",
                ],
            ),
            SkillDescriptor(
                skill_id="general.state.record",
                source="executor",
                supported_action_types=["state.record", "artifact.draft", "task.split"],
                required_capabilities=["general_state_change"],
                side_effect_class="state_change",
                requires_confirmation=True,
                input_schema={"type": "context_pack.v1"},
                output_schema={"type": "state_change_report.v1"},
                instructions=[
                    "Prepare a state-change artifact without executing external side effects.",
                    "Return the proposed update and any required follow-up.",
                ],
            ),
        ],
        priority=500,
    )
    return SkillCatalog(
        executors=executors + [generic_executor],
        builtin_skills=catalog.builtin_skills,
        metadata={"source": GENERAL_EXECUTION_ADAPTER},
    )


def _executor_from_capability(capability: Any) -> ExecutorDescriptor:
    executor_id = capability.provider or capability.capability_id
    return ExecutorDescriptor(
        executor_id=executor_id,
        display_name=executor_id,
        description=f"Executor descriptor derived from capability {capability.capability_id}.",
        priority=100,
        capabilities=[
            CapabilityDescriptor(
                capability_id=capability.capability_id,
                display_name=capability.scope,
                side_effect_classes=["external_effect" if capability.side_effects else "read_only"],
                max_duration_seconds=capability.max_duration_seconds,
            )
        ],
        skills=[
            SkillDescriptor(
                skill_id=f"{_id_segment(capability.capability_id)}.general_execution",
                source="executor",
                supported_action_types=["capability.use", "intent.execute"],
                required_capabilities=[capability.capability_id],
                side_effect_class="external_effect" if capability.side_effects else "read_only",
                requires_confirmation=capability.requires_confirmation,
                input_schema={"type": "context_pack.v1"},
                output_schema={"type": "execution_report.v1"},
                instructions=[
                    "Use the selected candidate and context pack as the execution boundary.",
                    "Return a structured task outcome for Spice state feedback.",
                ],
            )
        ],
    )


def _refs(
    *,
    execution_id: str,
    decision_id: str,
    trace_ref: str,
    candidate_id: str,
    target_refs: list[str],
    approval_id: str | None,
) -> list[str]:
    refs = [execution_id, decision_id, trace_ref, candidate_id, *target_refs]
    if approval_id:
        refs.append(approval_id)
    return [ref for ref in refs if ref]


def _make_general_execution_id(
    *,
    now: datetime,
    decision_id: str,
    trace_ref: str,
    candidate_id: str,
    approval_id: str | None,
    action_type: str,
    target_refs: list[str],
) -> str:
    stamp = timestamp_segment(now)
    seed = {
        "stamp": stamp,
        "decision_id": decision_id,
        "trace_ref": trace_ref,
        "candidate_id": candidate_id,
        "approval_id": approval_id,
        "action_type": action_type,
        "target_refs": list(target_refs),
    }
    return f"exec.{stamp}.{_id_segment(action_type)}.{_hash(repr(seed))}"


def _id_segment(value: str) -> str:
    normalized = "".join(
        char if char.isalnum() else "_"
        for char in str(value).strip().lower()
    ).strip("_")
    return normalized[:48] or "unknown"


def _hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()[:12]
