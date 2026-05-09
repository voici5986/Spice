from __future__ import annotations

from dataclasses import replace
from typing import Any

from spice.decision.general.candidates import GenericCandidate
from spice.llm.simulation_runner import LLMSimulationResult


DEFAULT_SIMULATION_TARGET_LIMIT = 3

_RUNTIME_GUARDRAIL_ACTION_TYPES = frozenset(
    {
        "approval.request",
        "artifact.draft",
        "context.prepare",
        "item.ignore",
        "state.observe_more",
        "state.record",
        "task.split",
        "time.defer",
        "user.clarify",
    }
)


def select_simulation_targets(
    candidates: list[GenericCandidate],
    *,
    explicit_options: list[str] | None = None,
    limit: int = DEFAULT_SIMULATION_TARGET_LIMIT,
) -> list[GenericCandidate]:
    """Pick the small set of candidates worth sending to LLM simulation."""

    if limit <= 0:
        return []

    selected: list[GenericCandidate] = []
    selected_ids: set[str] = set()

    def add(candidate: GenericCandidate) -> None:
        if len(selected) >= limit:
            return
        if candidate.candidate_id in selected_ids:
            return
        if candidate.availability_status == "blocked":
            return
        selected.append(candidate)
        selected_ids.add(candidate.candidate_id)

    if explicit_options:
        for candidate in candidates:
            if _is_explicit_option_candidate(candidate, option_count=len(explicit_options)):
                add(candidate)

    for candidate in candidates:
        if _is_decision_candidate(candidate) and not _is_runtime_guardrail(candidate):
            add(candidate)

    for candidate in candidates:
        if not _is_runtime_guardrail(candidate):
            add(candidate)

    return selected


def merge_simulation_result_candidates(
    candidates: list[GenericCandidate],
    simulation_result: LLMSimulationResult,
) -> LLMSimulationResult:
    """Merge simulated metadata back into the full candidate list by candidate_id."""

    simulations_by_candidate_id: dict[str, dict[str, Any]] = {}
    for candidate in simulation_result.candidates:
        simulation = candidate.metadata.get("llm_simulation")
        if isinstance(simulation, dict):
            simulations_by_candidate_id[candidate.candidate_id] = dict(simulation)

    merged = [
        _clone_with_simulation(candidate, simulations_by_candidate_id.get(candidate.candidate_id))
        for candidate in candidates
    ]
    return replace(simulation_result, candidates=merged)


def _clone_with_simulation(
    candidate: GenericCandidate,
    simulation: dict[str, Any] | None,
) -> GenericCandidate:
    if simulation is None:
        return candidate
    payload = candidate.to_payload()
    metadata = dict(payload.get("metadata")) if isinstance(payload.get("metadata"), dict) else {}
    metadata["llm_simulation"] = dict(simulation)
    payload["metadata"] = metadata
    return GenericCandidate.from_payload(payload)


def _is_explicit_option_candidate(candidate: GenericCandidate, *, option_count: int) -> bool:
    metadata = candidate.metadata or {}
    if metadata.get("source") == "explicit_options":
        return True
    if metadata.get("candidate_source") == "explicit_options":
        return True
    explicit_index = _metadata_int(metadata.get("explicit_option_index"))
    return explicit_index is not None and 1 <= explicit_index <= option_count


def _is_decision_candidate(candidate: GenericCandidate) -> bool:
    metadata = candidate.metadata or {}
    return (
        candidate.candidate_kind == "decision"
        or metadata.get("candidate_kind") == "decision"
        or metadata.get("candidate_source") in {"llm_generator", "explicit_options"}
        or metadata.get("source") == "explicit_options"
    )


def _is_runtime_guardrail(candidate: GenericCandidate) -> bool:
    return candidate.action_type in _RUNTIME_GUARDRAIL_ACTION_TYPES


def _metadata_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None
