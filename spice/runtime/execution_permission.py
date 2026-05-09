from __future__ import annotations

from typing import Any

from spice.decision.general.approval import Approval
from spice.decision.general.candidates import GenericCandidate
from spice.decision.general.permissions import (
    ExecutorPermissionRequirement,
    infer_executor_permission_requirement,
    permission_exceeds,
    permission_rank,
)
from spice.runtime.store import LocalJsonStore


def approval_permission_requirement(
    store: LocalJsonStore,
    approval: Approval,
) -> ExecutorPermissionRequirement:
    metadata = dict(approval.metadata)
    existing = metadata.get("permission_requirement")
    if isinstance(existing, dict):
        required = str(
            existing.get("required_permission")
            or metadata.get("required_executor_permission")
            or "read_only"
        )
        return ExecutorPermissionRequirement(
            required_permission=required,
            reason=str(
                existing.get("reason")
                or metadata.get("required_executor_permission_reason")
                or "Permission requirement recorded on approval."
            ),
            source=str(existing.get("source") or "approval_metadata"),
            target_refs=tuple(str(item) for item in _list(existing.get("target_refs"))),
            side_effect_class=str(existing.get("side_effect_class") or ""),
        )

    required = str(metadata.get("required_executor_permission") or "")
    if required:
        return ExecutorPermissionRequirement(
            required_permission=required,
            reason=str(
                metadata.get("required_executor_permission_reason")
                or "Permission requirement recorded on approval."
            ),
            source="approval_metadata",
        )

    candidate = _candidate_for_approval(store, approval)
    if candidate is None:
        return ExecutorPermissionRequirement(
            required_permission="read_only",
            reason="No selected candidate payload was found; defaulting to read-only.",
            source="fallback",
        )
    return infer_executor_permission_requirement(candidate)


def approval_requires_permission_escalation(
    *,
    store: LocalJsonStore,
    approval: Approval,
    current_permission: str,
) -> tuple[bool, ExecutorPermissionRequirement]:
    requirement = approval_permission_requirement(store, approval)
    return (
        permission_exceeds(requirement.required_permission, current_permission),
        requirement,
    )


def permission_mode_rank(permission: str | None) -> int:
    return permission_rank(permission)


def _candidate_for_approval(
    store: LocalJsonStore,
    approval: Approval,
) -> GenericCandidate | None:
    for run_id in store.list_record_ids("runs"):
        payload = store.load_run(run_id)
        if str(payload.get("approval_id") or "") != approval.approval_id:
            continue
        candidate = _candidate_from_payloads(payload, approval.candidate_id)
        if candidate is not None:
            return candidate
    return None


def _candidate_from_payloads(payload: dict[str, Any], candidate_id: str) -> GenericCandidate | None:
    for key in ("candidates", "evaluation_candidates"):
        for item in _list(payload.get(key)):
            if not isinstance(item, dict):
                continue
            if str(item.get("candidate_id") or "") != candidate_id:
                continue
            return GenericCandidate.from_payload(item)
    return None


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []
