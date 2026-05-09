from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from spice.decision.general.candidates import GenericCandidate
from spice.decision.general.types import payload_value


EXECUTOR_PERMISSION_ORDER = {
    "read_only": 0,
    "workspace_write": 1,
    "danger_full_access": 2,
}


@dataclass(frozen=True, slots=True)
class ExecutorPermissionRequirement:
    required_permission: str
    reason: str
    source: str = "spice_inference"
    target_refs: tuple[str, ...] = ()
    side_effect_class: str = ""

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


def infer_executor_permission_requirement(
    candidate: GenericCandidate,
) -> ExecutorPermissionRequirement:
    explicit = _normalize_permission(candidate.metadata.get("required_permission"))
    side_effect = _normalized_side_effect(candidate)
    text = " ".join(
        [
            candidate.action_type,
            candidate.intent,
            candidate.expected_state_delta.summary,
            " ".join(candidate.target_refs),
        ]
    ).lower()

    inferred = "read_only"
    reason = "This candidate can be handled without modifying the workspace."

    if _looks_destructive(text):
        inferred = "danger_full_access"
        reason = "The task appears destructive or may affect files outside the workspace."
    elif side_effect == "external_effect" or _looks_like_workspace_write(text):
        inferred = "workspace_write"
        reason = "The task may create or modify files in the workspace."
    elif side_effect in {"draft", "state_change"}:
        inferred = "read_only"
        reason = "The selected candidate records or drafts context without executor-side writes."

    required = _max_permission(inferred, explicit or "read_only")
    if explicit and permission_rank(explicit) > permission_rank(inferred):
        reason = f"The candidate explicitly requested {explicit}."

    return ExecutorPermissionRequirement(
        required_permission=required,
        reason=reason,
        target_refs=tuple(candidate.target_refs),
        side_effect_class=side_effect,
    )


def permission_rank(permission: str | None) -> int:
    return EXECUTOR_PERMISSION_ORDER.get(_normalize_permission(permission) or "read_only", 0)


def permission_exceeds(required: str | None, current: str | None) -> bool:
    return permission_rank(required) > permission_rank(current)


def _max_permission(left: str, right: str) -> str:
    return left if permission_rank(left) >= permission_rank(right) else right


def _normalize_permission(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"readonly", "read-only", "read_only"}:
        return "read_only"
    if normalized in {"workspace", "workspace-write", "workspace_write"}:
        return "workspace_write"
    if normalized in {"danger", "danger-full-access", "danger_full_access", "full_access"}:
        return "danger_full_access"
    return normalized if normalized in EXECUTOR_PERMISSION_ORDER else ""


def _normalized_side_effect(candidate: GenericCandidate) -> str:
    boundary = candidate.execution_boundary
    values = [
        candidate.side_effect_class,
        boundary.side_effect_class if boundary is not None else "",
        boundary.mode if boundary is not None else "",
    ]
    normalized: list[str] = []
    for value in values:
        token = str(value or "").strip().lower()
        if token in {"external", "external_effect", "execute", "write", "send", "capability"}:
            normalized.append("external_effect")
        elif token in {"draft", "state_change"}:
            normalized.append(token)
        elif token in {"none", "read_only", "read-or-prepare", "read_or_prepare"}:
            normalized.append("read_only")
    if "external_effect" in normalized:
        return "external_effect"
    if "state_change" in normalized:
        return "state_change"
    if "draft" in normalized:
        return "draft"
    if "read_only" in normalized:
        return "read_only"
    return ""


def _looks_like_workspace_write(text: str) -> bool:
    write_tokens = (
        "add ",
        "append ",
        "create ",
        "edit ",
        "fix ",
        "implement ",
        "modify ",
        "patch ",
        "rename ",
        "update ",
        "write ",
    )
    return any(token in text for token in write_tokens)


def _looks_destructive(text: str) -> bool:
    destructive_tokens = (
        "delete ",
        "remove ",
        "rm ",
        "drop ",
        "reset --hard",
        "force push",
        "sudo ",
        "outside this workspace",
        "/users/",
        "/etc/",
        "/usr/",
    )
    return any(token in text for token in destructive_tokens)
