from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

from spice.decision.general import (
    GeneralDecisionState,
    GenericCandidate,
    GenericObservation,
    GenericPolicyResult,
    ObservationConfidence,
    ObservationEvidence,
    ObservationKind,
    ObservationSource,
    ObservationSubject,
    OutcomeRecord,
    reduce_generic_observations,
)
from spice.decision.general.approval import Approval
from spice.decision.general.types import payload_value
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
from spice.protocols.sdep import SDEPExecuteResponse


RUNTIME_FULL_LOOP_PREVIEW_BUILDER = "spice.runtime.full_loop_preview"


@dataclass(slots=True)
class RuntimeFullLoopPreviewResult:
    artifact: dict[str, Any]
    state_snapshot: GeneralDecisionState


def build_runtime_full_loop_preview(
    *,
    state: GeneralDecisionState,
    candidates: list[GenericCandidate],
    policy_result: GenericPolicyResult,
    config: dict[str, Any],
    now: datetime | None = None,
) -> RuntimeFullLoopPreviewResult:
    """Build a read-only full-loop preview for one runtime decision.

    This creates a planned handoff artifact only. It does not send SDEP, call an
    executor, persist the feedback state, or mutate the input state.
    """

    created = now or datetime.now(timezone.utc)
    selected_candidate = _selected_candidate(
        candidates,
        policy_result.checkpoint.selected_candidate_id,
    )
    approval = _preview_approval(policy_result, selected_candidate, now=created)
    catalog = _runtime_skill_catalog(config)
    resolution = resolve_skill_for_candidate(selected_candidate, catalog)
    if resolution.status != "resolved" or resolution.resolved_skill is None:
        return RuntimeFullLoopPreviewResult(
            artifact=_unresolved_artifact(
                policy_result=policy_result,
                selected_candidate=selected_candidate,
                approval=approval,
                resolution=resolution.to_payload(),
                now=created,
            ),
            state_snapshot=state,
        )

    execution_id = _stable_id(
        "exec.runtime",
        {
            "decision_id": policy_result.checkpoint.decision_id,
            "trace_ref": policy_result.checkpoint.trace_ref,
            "candidate_id": selected_candidate.candidate_id,
            "approval_id": approval.approval_id if approval else None,
            "skill_id": resolution.resolved_skill.skill_id,
            "target_refs": selected_candidate.target_refs,
        },
        created,
    )
    request_id = f"sdep-req.runtime.{_hash(execution_id)[:16]}"
    context_pack = build_execution_context_pack(
        state=state,
        candidate=selected_candidate,
        resolved_skill=resolution.resolved_skill,
        decision_id=policy_result.checkpoint.decision_id,
        trace_ref=policy_result.checkpoint.trace_ref,
        approval_id=approval.approval_id if approval else "",
        execution_id=execution_id,
        request_id=request_id,
        metadata={"runtime_preview": True},
    )
    execution_intent = _execution_intent(
        selected_candidate=selected_candidate,
        approval=approval,
        policy_result=policy_result,
        resolved_skill=resolution.resolved_skill,
        context_pack=context_pack.to_payload(),
        execution_id=execution_id,
        now=created,
    )
    sdep_request = _sdep_request(execution_intent, request_id=request_id, now=created)
    sdep_response = _sdep_response_fixture(
        sdep_request=sdep_request,
        execution_id=execution_id,
        decision_id=policy_result.checkpoint.decision_id,
        trace_ref=policy_result.checkpoint.trace_ref,
        candidate_id=selected_candidate.candidate_id,
        approval_id=approval.approval_id if approval else None,
        now=created,
    )
    outcome_record = _outcome_record(
        response_payload=sdep_response,
        decision_id=policy_result.checkpoint.decision_id,
        trace_ref=policy_result.checkpoint.trace_ref,
        candidate_id=selected_candidate.candidate_id,
        execution_id=execution_id,
        request_id=request_id,
    )
    outcome_observation = _outcome_observation(
        outcome=outcome_record,
        response_payload=sdep_response,
        approval_id=approval.approval_id if approval else None,
        now=created,
    )
    state_snapshot = reduce_generic_observations(state, [outcome_observation])
    artifact = {
        "path_type": "runtime_full_loop_preview",
        "generated_by": RUNTIME_FULL_LOOP_PREVIEW_BUILDER,
        "loop_mode": "full_loop_preview",
        "loop_status": "completed_read_only",
        "created_at": _timestamp(created),
        "decision_id": policy_result.checkpoint.decision_id,
        "trace_ref": policy_result.checkpoint.trace_ref,
        "selected_candidate_id": selected_candidate.candidate_id,
        "approval_id": approval.approval_id if approval else None,
        "skill_id": resolution.resolved_skill.skill_id,
        "executor_id": resolution.resolved_skill.executor_id,
        "context_pack_id": context_pack.context_pack_id,
        "execution_id": execution_id,
        "request_id": request_id,
        "outcome_id": outcome_record.outcome_id,
        "protocol_status": sdep_response["status"],
        "task_status": sdep_response["outcome"]["status"],
        "read_only": True,
        "executor_called": False,
        "sdep_request_sent": False,
        "executed": False,
        "execution": None,
        "persisted": False,
        "state_snapshot_updated": True,
        "update_mode": "read_only_snapshot",
        "approval": approval.to_payload() if approval else None,
        "approval_fixture": {
            "local_approval_fixture_used_for_preview": approval is not None,
            "approval_required": selected_candidate.requires_confirmation,
        },
        "resolved_skill": resolution.resolved_skill.to_payload(),
        "context_pack": context_pack.to_payload(),
        "execution_intent": _execution_intent_payload(execution_intent),
        "sdep_request": sdep_request,
        "sdep_response_fixture": sdep_response,
        "outcome_record": outcome_record.to_payload(),
        "outcome_observation": outcome_observation.to_payload(),
        "state_feedback": {
            "state_snapshot_updated": True,
            "update_mode": "read_only_snapshot",
            "persisted": False,
            "state_before_summary": _state_summary(state),
            "state_after_summary": _state_summary(state_snapshot),
        },
    }
    artifact["rendered_text"] = render_runtime_full_loop_preview_text(artifact)
    return RuntimeFullLoopPreviewResult(artifact=artifact, state_snapshot=state_snapshot)


def render_runtime_full_loop_preview_text(artifact: dict[str, Any]) -> str:
    context_pack = _dict(artifact.get("context_pack"))
    resolved_skill = _dict(artifact.get("resolved_skill"))
    state_feedback = _dict(artifact.get("state_feedback"))
    before = _dict(state_feedback.get("state_before_summary"))
    after = _dict(state_feedback.get("state_after_summary"))
    return "\n".join(
        [
            "EXECUTION HANDOFF",
            "approval -> skill resolution -> context pack -> SDEP handoff -> outcome snapshot",
            "no executor called | no SDEP sent | no feedback state persisted",
            "",
            f"decision_id: {artifact.get('decision_id')}",
            f"trace_ref: {artifact.get('trace_ref')}",
            f"selected_candidate_id: {artifact.get('selected_candidate_id')}",
            "",
            "SKILL RESOLUTION",
            f"- planned_executor: {artifact.get('executor_id')}",
            f"- skill: {artifact.get('skill_id')}",
            f"- skill_source: {_nested(resolved_skill, 'metadata', 'skill_source')}",
            f"- context_pack_id: {artifact.get('context_pack_id')}",
            "",
            "CONTEXT PACK",
            f"- task: {context_pack.get('task')}",
            f"- why_now: {context_pack.get('why_now')}",
            f"- expected_output: {context_pack.get('expected_output')}",
            "",
            "EXECUTION BOUNDARY",
            f"- execution_id: {artifact.get('execution_id')}",
            f"- request_id: {artifact.get('request_id')}",
            "- request shaped but not sent",
            f"- sdep_request_sent: {str(bool(artifact.get('sdep_request_sent'))).lower()}",
            f"- executor_called: {str(bool(artifact.get('executor_called'))).lower()}",
            f"- executed: {str(bool(artifact.get('executed'))).lower()}",
            "",
            "OUTCOME RETURN",
            f"- outcome_id: {artifact.get('outcome_id')}",
            f"- protocol_status: {artifact.get('protocol_status')}",
            f"- task_status: {artifact.get('task_status')}",
            "- source: local fixture response",
            "",
            "STATE FEEDBACK",
            f"- outcomes: {before.get('outcome_count')} -> {after.get('outcome_count')}",
            f"- state_snapshot_updated: {str(bool(artifact.get('state_snapshot_updated'))).lower()}",
            f"- update_mode: {artifact.get('update_mode')}",
            f"- persisted: {str(bool(artifact.get('persisted'))).lower()}",
        ]
    )


def _runtime_skill_catalog(config: dict[str, Any]) -> SkillCatalog:
    base = builtin_fallback_skill_catalog()
    executor_id = str(config.get("executor") or "dry_run")
    executor = ExecutorDescriptor(
        executor_id=f"spice.{_safe_segment(executor_id)}",
        display_name=f"{executor_id} executor hint",
        description="Runtime preview executor descriptor. It is not called by this preview.",
        priority=500,
        capabilities=[
            CapabilityDescriptor(capability_id=f"capability.executor.{_safe_segment(executor_id)}"),
            CapabilityDescriptor(capability_id="work_item_triage"),
            CapabilityDescriptor(capability_id="runtime_context_prepare"),
            CapabilityDescriptor(capability_id="runtime_state_change"),
            CapabilityDescriptor(capability_id="runtime_external_execution"),
        ],
        skills=[
            SkillDescriptor(
                skill_id="runtime.context.prepare",
                display_name="Prepare context",
                source="executor",
                supported_action_types=["context.prepare"],
                required_capabilities=["runtime_context_prepare"],
                side_effect_class="read_only",
                requires_confirmation=False,
                input_schema={"type": "context_pack.v1"},
                output_schema={"type": "context_summary.v1"},
                instructions=[
                    "Use the compressed context pack only.",
                    "Return a concise context summary and next-step recommendation.",
                ],
            ),
            SkillDescriptor(
                skill_id="runtime.work_item.triage",
                display_name="Triage work item",
                source="executor",
                supported_action_types=["item.triage"],
                required_capabilities=["work_item_triage"],
                side_effect_class="state_change",
                requires_confirmation=True,
                input_schema={"type": "context_pack.v1"},
                output_schema={"type": "triage_report.v1"},
                instructions=[
                    "Use the selected work item and compressed context pack only.",
                    "Return a bounded triage plan, risks, and the next safe action.",
                ],
            ),
            SkillDescriptor(
                skill_id="runtime.state.record",
                display_name="Record state",
                source="executor",
                supported_action_types=["state.record", "artifact.draft", "task.split"],
                required_capabilities=["runtime_state_change"],
                side_effect_class="state_change",
                requires_confirmation=True,
                input_schema={"type": "context_pack.v1"},
                output_schema={"type": "state_change_report.v1"},
                instructions=[
                    "Prepare a state-change artifact without external side effects.",
                    "Return proposed updates and follow-up requirements.",
                ],
            ),
            SkillDescriptor(
                skill_id="runtime.intent.execute",
                display_name="Execute intent",
                source="executor",
                supported_action_types=["intent.execute", "capability.use"],
                required_capabilities=["runtime_external_execution"],
                side_effect_class="external_effect",
                requires_confirmation=True,
                input_schema={"type": "context_pack.v1"},
                output_schema={"type": "execution_report.v1"},
                instructions=[
                    "Use the selected candidate and context pack as the execution boundary.",
                    "Return protocol_status, task_status, output, and state_delta.",
                ],
            ),
        ],
        metadata={"preview_only": True},
    )
    return SkillCatalog(
        executors=[executor],
        builtin_skills=base.builtin_skills,
        metadata={"source": RUNTIME_FULL_LOOP_PREVIEW_BUILDER},
    )


def _execution_intent(
    *,
    selected_candidate: GenericCandidate,
    approval: Approval | None,
    policy_result: GenericPolicyResult,
    resolved_skill: Any,
    context_pack: dict[str, Any],
    execution_id: str,
    now: datetime,
) -> ExecutionIntent:
    return ExecutionIntent(
        id=execution_id,
        timestamp=now,
        refs=[
            execution_id,
            policy_result.checkpoint.decision_id,
            policy_result.checkpoint.trace_ref,
            selected_candidate.candidate_id,
            *(selected_candidate.target_refs),
            *([approval.approval_id] if approval else []),
        ],
        intent_type="runtime.execution_request",
        status="planned",
        objective={
            "id": policy_result.checkpoint.decision_id,
            "description": selected_candidate.intent,
        },
        executor_type=resolved_skill.executor_id,
        target={
            "kind": "general_reference",
            "id": selected_candidate.target_refs[0] if selected_candidate.target_refs else selected_candidate.candidate_id,
            "refs": list(selected_candidate.target_refs),
        },
        operation={
            "name": f"spice.runtime.{selected_candidate.action_type}",
            "mode": "sync",
            "dry_run": False,
        },
        input_payload={
            "decision_id": policy_result.checkpoint.decision_id,
            "trace_ref": policy_result.checkpoint.trace_ref,
            "candidate_id": selected_candidate.candidate_id,
            "approval_id": approval.approval_id if approval else None,
            "general_action_type": selected_candidate.action_type,
            "skill_id": resolved_skill.skill_id,
            "executor_id": resolved_skill.executor_id,
            "skill_hint": {
                "skill_id": resolved_skill.skill_id,
                "executor_id": resolved_skill.executor_id,
                "skill_source": resolved_skill.metadata.get("skill_source"),
                "side_effect_class": resolved_skill.side_effect_class,
                "context_pack_id": context_pack.get("context_pack_id"),
            },
            "context_pack_id": context_pack.get("context_pack_id"),
            "context_pack": context_pack,
        },
        parameters={
            "resolved_skill": resolved_skill.to_payload(),
            "selected_candidate": selected_candidate.to_payload(),
            "planning_only": True,
        },
        constraints=list(selected_candidate.constraints_triggered),
        success_criteria=[
            {
                "description": selected_candidate.expected_state_delta.summary
                or f"Complete {selected_candidate.action_type}.",
                "source": "generic_candidate.expected_state_delta",
            }
        ],
        provenance={
            "adapter": RUNTIME_FULL_LOOP_PREVIEW_BUILDER,
            "spice_decision_id": policy_result.checkpoint.decision_id,
            "trace_ref": policy_result.checkpoint.trace_ref,
            "candidate_id": selected_candidate.candidate_id,
            "approval_id": approval.approval_id if approval else None,
            "skill_id": resolved_skill.skill_id,
            "executor_id": resolved_skill.executor_id,
            "context_pack_id": context_pack.get("context_pack_id"),
        },
        metadata={
            "planning_only": True,
            "executor_called": False,
        },
    )


def _sdep_request(intent: ExecutionIntent, *, request_id: str, now: datetime) -> dict[str, Any]:
    request = build_sdep_execute_request(
        intent,
        request_id=request_id,
        metadata={
            "runtime": "spice",
            "adapter": RUNTIME_FULL_LOOP_PREVIEW_BUILDER,
            "planning_only": True,
        },
    )
    request.message_id = f"sdep-msg.runtime.{_hash(intent.id)[:16]}"
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
            "adapter": RUNTIME_FULL_LOOP_PREVIEW_BUILDER,
            "general_action_type": str(intent.input_payload.get("general_action_type", "")),
            "skill_id": str(intent.input_payload.get("skill_id", "")),
            "executor_id": str(intent.input_payload.get("executor_id", "")),
            "context_pack_id": str(intent.input_payload.get("context_pack_id", "")),
            "planning_only": True,
        }
    )
    request.validate()
    return request.to_dict()


def _sdep_response_fixture(
    *,
    sdep_request: dict[str, Any],
    execution_id: str,
    decision_id: str,
    trace_ref: str,
    candidate_id: str,
    approval_id: str | None,
    now: datetime,
) -> dict[str, Any]:
    request_id = str(sdep_request.get("request_id") or "")
    response = {
        "protocol": "sdep",
        "sdep_version": "0.1",
        "message_type": "execute.response",
        "message_id": f"sdep-msg.runtime.response.{_hash(execution_id)[:16]}",
        "request_id": request_id,
        "timestamp": _timestamp(now),
        "responder": {
            "id": "agent.runtime.fixture",
            "name": "Runtime Fixture Executor",
            "version": "0.1",
            "vendor": "Spice",
            "implementation": "read-only-fixture",
            "role": "executor",
        },
        "status": "success",
        "outcome": {
            "execution_id": execution_id,
            "status": "success",
            "outcome_type": "observation",
            "output": {
                "summary": "Planned action completed by read-only fixture.",
                "state_delta": {
                    "updated_refs": [candidate_id],
                    "task_status": "success",
                },
            },
            "artifacts": [],
            "metrics": {},
            "metadata": {
                "adapter": RUNTIME_FULL_LOOP_PREVIEW_BUILDER,
                "fixture": True,
            },
        },
        "traceability": {
            "execution_id": execution_id,
            "spice_decision_id": decision_id,
            "trace_ref": trace_ref,
            "candidate_id": candidate_id,
            "approval_id": approval_id,
        },
        "metadata": {
            "adapter": RUNTIME_FULL_LOOP_PREVIEW_BUILDER,
            "fixture": True,
            "planning_only": False,
        },
    }
    return SDEPExecuteResponse.from_dict(response).to_dict()


def _outcome_record(
    *,
    response_payload: dict[str, Any],
    decision_id: str,
    trace_ref: str,
    candidate_id: str,
    execution_id: str,
    request_id: str,
) -> OutcomeRecord:
    response = SDEPExecuteResponse.from_dict(response_payload)
    output = dict(response.outcome.output)
    return OutcomeRecord(
        outcome_id=f"outcome.runtime.{_hash(f'{request_id}\\n{execution_id}\\n{response.status}\\n{response.outcome.status}')[:16]}",
        decision_id=decision_id,
        trace_ref=trace_ref,
        candidate_id=candidate_id,
        execution_ref=execution_id,
        protocol_status=response.status,
        task_status=response.outcome.status,
        status="observed",
        summary=str(output.get("summary") or f"Task status: {response.outcome.status}"),
        state_delta=dict(output.get("state_delta")) if isinstance(output.get("state_delta"), dict) else {},
        evidence_refs=[response.message_id],
        metadata={
            "adapter": RUNTIME_FULL_LOOP_PREVIEW_BUILDER,
            "approval_id": response.traceability.get("approval_id"),
            "execution_id": execution_id,
            "request_id": request_id,
            "response_message_id": response.message_id,
            "protocol_status": response.status,
            "task_status": response.outcome.status,
            "traceability": dict(response.traceability),
            "responder": response.responder.to_dict(),
            "output": output,
            "fixture": True,
        },
    )


def _outcome_observation(
    *,
    outcome: OutcomeRecord,
    response_payload: dict[str, Any],
    approval_id: str | None,
    now: datetime,
) -> GenericObservation:
    response = SDEPExecuteResponse.from_dict(response_payload)
    metadata = {
        "adapter": RUNTIME_FULL_LOOP_PREVIEW_BUILDER,
        "decision_id": outcome.decision_id,
        "trace_ref": outcome.trace_ref,
        "candidate_id": outcome.candidate_id,
        "approval_id": approval_id,
        "execution_id": outcome.execution_ref,
        "request_id": outcome.metadata.get("request_id"),
        "outcome_id": outcome.outcome_id,
        "protocol_status": outcome.protocol_status,
        "task_status": outcome.task_status,
        "response_message_id": response.message_id,
        "responder": response.responder.to_dict(),
        "output": dict(response.outcome.output),
        "fixture": True,
    }
    return GenericObservation(
        observation_id=f"obs.runtime.outcome.{_hash(outcome.outcome_id)[:16]}",
        kind=ObservationKind.OUTCOME,
        source=ObservationSource(
            provider="runtime_full_loop_preview",
            channel="local_fixture",
            received_at=_timestamp(now),
        ),
        subject=ObservationSubject(
            subject_id=outcome.execution_ref,
            subject_type="execution_outcome",
            title=outcome.summary,
            refs=[
                outcome.decision_id,
                outcome.trace_ref or "",
                outcome.candidate_id or "",
                outcome.execution_ref,
                outcome.outcome_id,
            ],
        ),
        summary=outcome.summary,
        attributes={
            "outcome_id": outcome.outcome_id,
            "decision_id": outcome.decision_id,
            "trace_ref": outcome.trace_ref,
            "candidate_id": outcome.candidate_id,
            "approval_id": approval_id,
            "execution_id": outcome.execution_ref,
            "request_id": outcome.metadata.get("request_id"),
            "protocol_status": outcome.protocol_status,
            "task_status": outcome.task_status,
            "state_delta": dict(outcome.state_delta),
        },
        evidence=[
            ObservationEvidence(
                evidence_id=response.message_id,
                kind="sdep_execute_response_fixture",
                summary="Local SDEP response fixture for full-loop preview.",
                content=response.to_dict(),
            )
        ],
        confidence=ObservationConfidence(score=1.0, level="high"),
        refs=[
            outcome.decision_id,
            outcome.trace_ref or "",
            outcome.candidate_id or "",
            outcome.execution_ref,
            outcome.outcome_id,
        ],
        metadata=metadata,
    )


def _preview_approval(
    policy_result: GenericPolicyResult,
    selected_candidate: GenericCandidate,
    *,
    now: datetime,
) -> Approval | None:
    approval = policy_result.checkpoint.approval
    if approval is None and not selected_candidate.requires_confirmation:
        return None
    if approval is None:
        approval = Approval(
            approval_id=f"approval.runtime.{_hash(f'{policy_result.checkpoint.decision_id}\\n{selected_candidate.candidate_id}')[:16]}",
            decision_id=policy_result.checkpoint.decision_id,
            candidate_id=selected_candidate.candidate_id,
            status="pending",
            requested_at=_timestamp(now),
            execution_allowed=False,
        )
    return Approval(
        approval_id=approval.approval_id,
        decision_id=approval.decision_id,
        candidate_id=approval.candidate_id,
        status="approved",
        mode=approval.mode,
        requested_at=approval.requested_at or _timestamp(now),
        resolved_at=_timestamp(now),
        actor="runtime_preview",
        prompt=approval.prompt,
        response="local_preview_confirm",
        reason=approval.reason,
        execution_allowed=True,
        metadata={
            **dict(approval.metadata),
            "trace_ref": policy_result.checkpoint.trace_ref,
            "local_approval_fixture_used_for_preview": True,
        },
    )


def _unresolved_artifact(
    *,
    policy_result: GenericPolicyResult,
    selected_candidate: GenericCandidate,
    approval: Approval | None,
    resolution: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    artifact = {
        "path_type": "runtime_full_loop_preview",
        "generated_by": RUNTIME_FULL_LOOP_PREVIEW_BUILDER,
        "loop_mode": "full_loop_preview",
        "loop_status": "skill_unresolved",
        "created_at": _timestamp(now),
        "decision_id": policy_result.checkpoint.decision_id,
        "trace_ref": policy_result.checkpoint.trace_ref,
        "selected_candidate_id": selected_candidate.candidate_id,
        "approval_id": approval.approval_id if approval else None,
        "read_only": True,
        "executor_called": False,
        "sdep_request_sent": False,
        "executed": False,
        "execution": None,
        "persisted": False,
        "state_snapshot_updated": False,
        "update_mode": "read_only_snapshot",
        "skill_resolution": resolution,
        "reason": "; ".join(resolution.get("unresolved_reasons", []))
        if isinstance(resolution.get("unresolved_reasons"), list)
        else "skill unresolved",
    }
    artifact["rendered_text"] = render_runtime_full_loop_preview_text(artifact)
    return artifact


def _execution_intent_payload(intent: ExecutionIntent) -> dict[str, Any]:
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


def _selected_candidate(candidates: list[GenericCandidate], candidate_id: str) -> GenericCandidate:
    for candidate in candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    raise ValueError(f"selected candidate not found: {candidate_id}")


def _state_summary(state: GeneralDecisionState) -> dict[str, Any]:
    return {
        "state_id": state.state_id,
        "observation_count": len(state.observations),
        "intent_count": len(state.intents),
        "commitment_count": len(state.commitments),
        "work_item_count": len(state.work_items),
        "capability_count": len(state.capabilities),
        "constraint_count": len(state.constraints),
        "risk_count": len(state.risks),
        "open_loop_count": len(state.open_loops),
        "approval_count": len(state.approvals),
        "outcome_count": len(state.outcomes),
    }


def _stable_id(prefix: str, value: Any, now: datetime) -> str:
    return f"{prefix}.{now.strftime('%Y%m%dT%H%M%S.%fZ')}.{_hash(value)[:16]}"


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _hash(value: Any) -> str:
    return sha256(repr(payload_value(value)).encode("utf-8")).hexdigest()


def _safe_segment(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_") or "unknown"


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current
