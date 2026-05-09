from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from spice.decision.general.candidates import GenericCandidate
from spice.executors.skills import (
    ExecutorDescriptor,
    ResolvedSkill,
    SkillCatalog,
    SkillDescriptor,
    payload_value,
    safe_dataclass_from_payload,
)


RESOLUTION_STATUSES = (
    "resolved",
    "unresolved",
)

_SOURCE_PRIORITY = {
    "user": 0,
    "project": 1,
    "executor": 2,
    "builtin": 3,
}

_SIDE_EFFECT_RANK = {
    "read_only": 0,
    "state_change": 1,
    "external_effect": 2,
}


@dataclass(slots=True)
class SkillResolutionResult:
    status: str
    candidate_id: str
    action_type: str
    resolved_skill: ResolvedSkill | None = None
    unresolved_reasons: list[str] = field(default_factory=list)
    considered_skill_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.status not in RESOLUTION_STATUSES:
            allowed = ", ".join(RESOLUTION_STATUSES)
            raise ValueError(f"skill resolution status must be one of [{allowed}]")
        if not self.candidate_id:
            raise ValueError("skill resolution candidate_id is required")
        if not self.action_type:
            raise ValueError("skill resolution action_type is required")
        if self.status == "resolved" and self.resolved_skill is None:
            raise ValueError("resolved skill resolution requires resolved_skill")
        if self.status == "unresolved" and self.resolved_skill is not None:
            raise ValueError("unresolved skill resolution cannot include resolved_skill")

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SkillResolutionResult":
        item = safe_dataclass_from_payload(cls, payload)
        item.resolved_skill = (
            ResolvedSkill.from_payload(payload["resolved_skill"])
            if isinstance(payload.get("resolved_skill"), dict)
            else None
        )
        item.unresolved_reasons = _string_list(payload.get("unresolved_reasons"))
        item.considered_skill_ids = _string_list(payload.get("considered_skill_ids"))
        item.metadata = dict(payload.get("metadata")) if isinstance(payload.get("metadata"), dict) else {}
        item.validate()
        return item


def resolve_skill_for_candidate(
    candidate: GenericCandidate,
    catalog: SkillCatalog,
    *,
    preferred_executor_ids: list[str] | tuple[str, ...] = (),
    preferred_skill_ids: list[str] | tuple[str, ...] = (),
) -> SkillResolutionResult:
    """Resolve a generic candidate to a skill without planning or executing it."""
    if not isinstance(candidate, GenericCandidate):
        raise ValueError("candidate must be a GenericCandidate")
    if not isinstance(catalog, SkillCatalog):
        raise ValueError("catalog must be a SkillCatalog")

    catalog.validate()
    required_side_effect = candidate_required_side_effect_class(candidate)
    entries = _skill_entries(catalog)
    considered_skill_ids: list[str] = []
    compatible: list[_SkillMatch] = []
    rejection_reasons: list[str] = []

    for entry in entries:
        skill = entry.skill
        if candidate.action_type not in skill.supported_action_types:
            continue
        considered_skill_ids.append(skill.skill_id)

        executor = _resolve_executor(entry, skill, candidate, catalog)
        if executor is None and _needs_executor(skill, candidate):
            rejection_reasons.append(
                f"{skill.skill_id}: no available executor provides required capability"
            )
            continue

        capability_id = _resolved_capability_id(skill, candidate)
        if executor is not None and capability_id:
            capability_ids = set(executor.capability_ids())
            if capability_id not in capability_ids:
                rejection_reasons.append(
                    f"{skill.skill_id}: executor {executor.executor_id} lacks {capability_id}"
                )
                continue

        missing_skill_capability = _missing_skill_capability(skill, executor)
        if missing_skill_capability:
            rejection_reasons.append(
                f"{skill.skill_id}: executor lacks required skill capability {missing_skill_capability}"
            )
            continue

        if not _side_effect_allowed(skill.side_effect_class, required_side_effect):
            rejection_reasons.append(
                f"{skill.skill_id}: side_effect_mismatch "
                f"{skill.side_effect_class} != {required_side_effect}"
            )
            continue

        compatible.append(
            _SkillMatch(
                skill=skill,
                executor=executor,
                capability_id=capability_id,
                required_side_effect=required_side_effect,
                ranking=_ranking(
                    entry=entry,
                    skill=skill,
                    executor=executor,
                    preferred_executor_ids=preferred_executor_ids,
                    preferred_skill_ids=preferred_skill_ids,
                ),
            )
        )

    if not compatible:
        reasons = rejection_reasons or [
            f"no skill supports action_type {candidate.action_type}"
        ]
        return SkillResolutionResult(
            status="unresolved",
            candidate_id=candidate.candidate_id,
            action_type=candidate.action_type,
            unresolved_reasons=reasons,
            considered_skill_ids=considered_skill_ids,
            metadata={
                "required_capability": candidate.required_capability,
                "required_side_effect_class": required_side_effect,
            },
        )

    match = sorted(compatible, key=lambda item: item.ranking)[0]
    executor_id = match.executor.executor_id if match.executor else _virtual_executor_id(match.skill)
    requires_confirmation = bool(candidate.requires_confirmation or match.skill.requires_confirmation)
    resolved = ResolvedSkill(
        executor_id=executor_id,
        skill_id=match.skill.skill_id,
        action_type=candidate.action_type,
        capability_id=match.capability_id,
        side_effect_class=match.skill.side_effect_class,
        requires_confirmation=requires_confirmation,
        input_schema=dict(match.skill.input_schema),
        output_schema=dict(match.skill.output_schema),
        instructions=list(match.skill.instructions),
        resolution_reason=_resolution_reason(candidate, match),
        confidence=1.0,
        metadata={
            "candidate_id": candidate.candidate_id,
            "skill_source": match.skill.source,
            "required_side_effect_class": match.required_side_effect,
            "target_refs": list(candidate.target_refs),
            "requires_confirmation_from_candidate": candidate.requires_confirmation,
            "requires_confirmation_from_skill": match.skill.requires_confirmation,
        },
    )
    resolved.validate()
    return SkillResolutionResult(
        status="resolved",
        candidate_id=candidate.candidate_id,
        action_type=candidate.action_type,
        resolved_skill=resolved,
        considered_skill_ids=considered_skill_ids,
        metadata={
            "selected_skill_source": match.skill.source,
            "selected_executor_priority": match.executor.priority if match.executor else None,
        },
    )


def resolve_skills_for_candidates(
    candidates: list[GenericCandidate],
    catalog: SkillCatalog,
    *,
    preferred_executor_ids: list[str] | tuple[str, ...] = (),
    preferred_skill_ids: list[str] | tuple[str, ...] = (),
) -> list[SkillResolutionResult]:
    return [
        resolve_skill_for_candidate(
            candidate,
            catalog,
            preferred_executor_ids=preferred_executor_ids,
            preferred_skill_ids=preferred_skill_ids,
        )
        for candidate in candidates
    ]


def candidate_required_side_effect_class(candidate: GenericCandidate) -> str:
    boundary_value = getattr(candidate.execution_boundary, "side_effect_class", "") or ""
    action_type = candidate.action_type
    if boundary_value in _SIDE_EFFECT_RANK:
        return boundary_value
    if action_type == "state.record":
        return "state_change"
    if action_type in {"intent.execute", "capability.use"}:
        return "external_effect"
    if action_type in {"artifact.draft", "approval.request", "task.split"}:
        return "state_change"
    value = candidate.side_effect_class or boundary_value
    if value in {"external", "execute", "write", "send"}:
        return "external_effect"
    if value in {"draft", "low", "state_change"}:
        return "state_change"
    return "read_only"


@dataclass(frozen=True, slots=True)
class _SkillEntry:
    skill: SkillDescriptor
    executor: ExecutorDescriptor | None
    source_rank: int
    order: int


@dataclass(frozen=True, slots=True)
class _SkillMatch:
    skill: SkillDescriptor
    executor: ExecutorDescriptor | None
    capability_id: str
    required_side_effect: str
    ranking: tuple[int, int, int, int, str, str]


def _skill_entries(catalog: SkillCatalog) -> list[_SkillEntry]:
    entries: list[_SkillEntry] = []
    order = 0
    for skill in catalog.user_skills:
        entries.append(_SkillEntry(skill, None, _SOURCE_PRIORITY["user"], order))
        order += 1
    for skill in catalog.project_skills:
        entries.append(_SkillEntry(skill, None, _SOURCE_PRIORITY["project"], order))
        order += 1
    for executor in sorted(catalog.executors, key=lambda item: (item.priority, item.executor_id)):
        if executor.status != "available":
            continue
        for skill in sorted(executor.skills, key=lambda item: item.skill_id):
            entries.append(_SkillEntry(skill, executor, _SOURCE_PRIORITY["executor"], order))
            order += 1
    for skill in catalog.builtin_skills:
        entries.append(_SkillEntry(skill, None, _SOURCE_PRIORITY["builtin"], order))
        order += 1
    return entries


def _resolve_executor(
    entry: _SkillEntry,
    skill: SkillDescriptor,
    candidate: GenericCandidate,
    catalog: SkillCatalog,
) -> ExecutorDescriptor | None:
    if entry.executor is not None:
        return entry.executor
    metadata_executor_id = skill.metadata.get("executor_id")
    if isinstance(metadata_executor_id, str) and metadata_executor_id:
        executor = catalog.find_executor(metadata_executor_id)
        if executor and executor.status == "available":
            return executor
        return None
    required_capabilities = _required_capability_ids(skill, candidate)
    if not required_capabilities:
        return None
    return _find_executor_with_capabilities(catalog, required_capabilities)


def _find_executor_with_capabilities(
    catalog: SkillCatalog,
    capability_ids: list[str],
) -> ExecutorDescriptor | None:
    required = {item for item in capability_ids if item}
    for executor in sorted(catalog.executors, key=lambda item: (item.priority, item.executor_id)):
        if executor.status != "available":
            continue
        if required.issubset(set(executor.capability_ids())):
            return executor
    return None


def _needs_executor(skill: SkillDescriptor, candidate: GenericCandidate) -> bool:
    return bool(candidate.required_capability or skill.required_capabilities)


def _resolved_capability_id(skill: SkillDescriptor, candidate: GenericCandidate) -> str:
    if candidate.required_capability:
        return candidate.required_capability
    if len(skill.required_capabilities) == 1:
        return skill.required_capabilities[0]
    return ""


def _required_capability_ids(skill: SkillDescriptor, candidate: GenericCandidate) -> list[str]:
    values: list[str] = []
    if candidate.required_capability:
        values.append(candidate.required_capability)
    values.extend(skill.required_capabilities)
    return list(dict.fromkeys(value for value in values if value))


def _missing_skill_capability(
    skill: SkillDescriptor,
    executor: ExecutorDescriptor | None,
) -> str:
    if not skill.required_capabilities:
        return ""
    if executor is None:
        return skill.required_capabilities[0]
    capability_ids = set(executor.capability_ids())
    for capability_id in skill.required_capabilities:
        if capability_id not in capability_ids:
            return capability_id
    return ""


def _side_effect_allowed(skill_side_effect: str, required_side_effect: str) -> bool:
    return skill_side_effect == required_side_effect


def _ranking(
    *,
    entry: _SkillEntry,
    skill: SkillDescriptor,
    executor: ExecutorDescriptor | None,
    preferred_executor_ids: list[str] | tuple[str, ...],
    preferred_skill_ids: list[str] | tuple[str, ...],
) -> tuple[int, int, int, int, str, str]:
    executor_id = executor.executor_id if executor else _virtual_executor_id(skill)
    skill_preference = _preference_rank(skill.skill_id, preferred_skill_ids)
    executor_preference = _preference_rank(executor_id, preferred_executor_ids)
    executor_priority = executor.priority if executor else 1000
    return (
        skill_preference,
        executor_preference,
        entry.source_rank,
        executor_priority,
        skill.skill_id,
        executor_id,
    )


def _preference_rank(value: str, preferred_values: list[str] | tuple[str, ...]) -> int:
    try:
        return list(preferred_values).index(value)
    except ValueError:
        return len(preferred_values) + 1


def _virtual_executor_id(skill: SkillDescriptor) -> str:
    return f"spice.{skill.source or 'skill'}"


def _resolution_reason(candidate: GenericCandidate, match: _SkillMatch) -> str:
    parts = [
        f"matched action_type {candidate.action_type}",
        f"skill source {match.skill.source}",
    ]
    if match.capability_id:
        parts.append(f"capability {match.capability_id}")
    parts.append(f"side effect == {match.required_side_effect}")
    return "; ".join(parts)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item.strip()]
