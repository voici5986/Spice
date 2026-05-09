from __future__ import annotations

import re
from typing import Any

from spice.decision.general import GenericCandidate
from spice.executors import SkillCatalog, resolve_skill_for_candidate
from spice.executors.skill_resolver import SkillResolutionResult
from spice.runtime.full_loop_preview import _runtime_skill_catalog
from spice.runtime.workspace import SpiceWorkspaceConfig


def runtime_skill_catalog(config: SpiceWorkspaceConfig | dict[str, Any]) -> SkillCatalog:
    payload = config.to_payload() if isinstance(config, SpiceWorkspaceConfig) else dict(config)
    return _runtime_skill_catalog(payload)


def runtime_skill_catalog_summary(config: SpiceWorkspaceConfig | dict[str, Any]) -> dict[str, Any]:
    catalog = runtime_skill_catalog(config)
    return {
        "source": "runtime_static_catalog",
        "executors": [
            {
                "executor_id": executor.executor_id,
                "display_name": executor.display_name,
                "status": executor.status,
                "capability_ids": executor.capability_ids(),
                "skill_ids": executor.skill_ids(),
            }
            for executor in catalog.executors
        ],
        "builtin_skill_ids": [skill.skill_id for skill in catalog.builtin_skills],
        "user_skill_ids": [skill.skill_id for skill in catalog.user_skills],
        "project_skill_ids": [skill.skill_id for skill in catalog.project_skills],
    }


def resolve_runtime_skill_for_candidate(
    candidate: GenericCandidate,
    *,
    config: SpiceWorkspaceConfig | dict[str, Any],
) -> SkillResolutionResult:
    payload = config.to_payload() if isinstance(config, SpiceWorkspaceConfig) else dict(config)
    executor_id = str(payload.get("executor") or "dry_run")
    return resolve_skill_for_candidate(
        candidate,
        runtime_skill_catalog(payload),
        preferred_executor_ids=(f"spice.{_safe_segment(executor_id)}",),
    )


def annotate_skill_resolutions(
    candidates: list[GenericCandidate],
    *,
    config: SpiceWorkspaceConfig | dict[str, Any],
) -> list[GenericCandidate]:
    annotated: list[GenericCandidate] = []
    for candidate in candidates:
        resolution = resolve_runtime_skill_for_candidate(candidate, config=config)
        metadata = dict(candidate.metadata or {})
        metadata["skill_resolution"] = resolution.to_payload()
        if resolution.resolved_skill is not None:
            metadata["resolved_skill"] = resolution.resolved_skill.to_payload()
        else:
            metadata.pop("resolved_skill", None)
        candidate.metadata = metadata
        annotated.append(candidate)
    return annotated


def selected_skill_resolution_payload(
    candidates: list[GenericCandidate],
    *,
    selected_candidate_id: str,
) -> dict[str, Any]:
    for candidate in candidates:
        if candidate.candidate_id != selected_candidate_id:
            continue
        value = candidate.metadata.get("skill_resolution")
        return dict(value) if isinstance(value, dict) else {}
    return {}


def _safe_segment(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip().lower())
    return text.strip("._-") or "executor"
