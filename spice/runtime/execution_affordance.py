from __future__ import annotations

from typing import Any

from spice.decision.general.candidates import (
    GenericExecutionIntent,
    GenericCandidate,
    is_approval_eligible_executable_candidate,
)
from spice.decision.general.permissions import (
    infer_executor_permission_requirement,
    permission_exceeds,
)
from spice.runtime.executor_runtime import (
    ResolvedExecutorRuntime,
    resolve_executor_runtime_from_config,
)
from spice.runtime.workspace import SpiceWorkspaceConfig


def annotate_execution_affordances(
    candidates: list[GenericCandidate],
    *,
    config: SpiceWorkspaceConfig | dict[str, Any],
) -> list[GenericCandidate]:
    runtime = (
        resolve_executor_runtime_from_config(config)
        if isinstance(config, SpiceWorkspaceConfig)
        else resolve_executor_runtime_from_config(SpiceWorkspaceConfig.from_payload(config))
    )
    return [
        _annotate_candidate(candidate, runtime=runtime)
        for candidate in candidates
    ]


def build_execution_affordance(
    candidate: GenericCandidate,
    *,
    executor_runtime: ResolvedExecutorRuntime,
) -> dict[str, Any]:
    requirement = infer_executor_permission_requirement(candidate)
    candidate_eligible = is_approval_eligible_executable_candidate(candidate)
    executor_ready = executor_runtime.status == "ready"
    escalation_required = permission_exceeds(
        requirement.required_permission,
        executor_runtime.permission_mode,
    )
    escalation_supported = (
        not escalation_required
        or executor_runtime.permission_enforcement == "command_flag"
    )
    blockers = _candidate_execution_blockers(candidate)
    if not executor_ready:
        blockers.append(executor_runtime.detail or f"{executor_runtime.executor_id} is not ready.")
    if escalation_required and not escalation_supported:
        blockers.append(
            f"Executor permission escalation to {requirement.required_permission} is not automated."
        )
    approval_required = bool(
        candidate_eligible
        and (candidate.requires_confirmation or executor_runtime.approval_required)
    )
    executable = bool(candidate_eligible and executor_ready and escalation_supported)
    return {
        "schema_version": "0.1",
        "generated_by": "spice.runtime.execution_affordance",
        "candidate_executable": candidate_eligible,
        "executor_available": executor_ready,
        "executable": executable,
        "blocked": bool(blockers),
        "blocked_reason": blockers[0] if blockers else "",
        "blockers": blockers,
        "executor": {
            "executor_id": executor_runtime.executor_id,
            "requested_executor_id": executor_runtime.requested_executor_id,
            "transport": executor_runtime.transport,
            "status": executor_runtime.status,
            "detail": executor_runtime.detail,
            "command": executor_runtime.command,
            "command_source": executor_runtime.command_source,
            "command_found": executor_runtime.command_found,
            "command_path": executor_runtime.command_path,
            "real_executor": executor_runtime.real_executor,
            "sends_sdep_request": executor_runtime.sends_sdep_request,
        },
        "permission": {
            "required": requirement.required_permission,
            "configured": executor_runtime.permission_mode,
            "reason": requirement.reason,
            "source": requirement.source,
            "side_effect_class": requirement.side_effect_class,
            "escalation_required": escalation_required,
            "escalation_supported": escalation_supported,
            "enforcement": executor_runtime.permission_enforcement,
        },
        "approval": {
            "required": approval_required,
            "candidate_requires_confirmation": bool(candidate.requires_confirmation),
            "executor_approval_required": bool(executor_runtime.approval_required),
            "eligible_for_approval": candidate_eligible,
            "status": "approval_required_on_selection" if candidate_eligible else "not_approval_eligible",
        },
    }


def _annotate_candidate(
    candidate: GenericCandidate,
    *,
    runtime: ResolvedExecutorRuntime,
) -> GenericCandidate:
    metadata = dict(candidate.metadata or {})
    metadata["execution_affordance"] = build_execution_affordance(
        candidate,
        executor_runtime=runtime,
    )
    candidate.metadata = metadata
    return candidate


def _candidate_execution_blockers(candidate: GenericCandidate) -> list[str]:
    blockers: list[str] = []
    if candidate.availability_status == "blocked":
        reason = "; ".join(candidate.why_blocked) or "Candidate availability is blocked."
        blockers.append(reason)
    execution_intent = getattr(candidate, "execution_intent", GenericExecutionIntent())
    if execution_intent.intent_class != "execution_requested":
        blockers.append(
            "Candidate is advisory; execution_intent.intent_class is not execution_requested."
        )
    if not execution_intent.requested:
        blockers.append("Candidate execution_intent.requested is false.")
    if not str(execution_intent.handoff_task or "").strip():
        blockers.append("Candidate execution_intent.handoff_task is empty.")
    if not candidate.requires_confirmation:
        blockers.append("Candidate does not request an approval-gated executor handoff.")
    boundary = candidate.execution_boundary
    if boundary is not None and boundary.requires_confirmation is False:
        blockers.append("Candidate execution boundary does not require confirmation.")
    if not _has_handoff_anchor(candidate):
        blockers.append("Candidate has no concrete executor handoff target.")
    if candidate.action_type == "capability.use" and not candidate.required_capability:
        blockers.append("Candidate uses a capability action but does not name a required capability.")
    return _dedupe(blockers)


def _has_handoff_anchor(candidate: GenericCandidate) -> bool:
    execution_intent = getattr(candidate, "execution_intent", GenericExecutionIntent())
    if str(execution_intent.handoff_task or "").strip():
        return True
    boundary = candidate.execution_boundary
    if candidate.target_refs:
        return True
    if boundary is None:
        return False
    if boundary.target or boundary.protocol:
        return True
    return boundary.mode in {"execution_intent", "capability", "sdep"}


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result
