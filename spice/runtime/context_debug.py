from __future__ import annotations

from pathlib import Path
from typing import Any

from spice.decision.general import load_general_state
from spice.decision.general.types import payload_value
from spice.protocols import WorldState
from spice.runtime.session import load_or_create_session
from spice.runtime.store import LocalJsonStore
from spice.runtime.workspace import (
    load_workspace_config,
    load_workspace_context_compiler,
    require_workspace,
)


def compile_workspace_decision_context_payload(
    *,
    project_root: str | Path = ".",
    session_id: str | None = None,
) -> dict[str, Any]:
    """Compile the same general decision context exposed by the TUI /context command."""

    paths = require_workspace(project_root)
    store = LocalJsonStore(paths)
    config = load_workspace_config(project_root)
    config_payload = config.to_payload()
    state_payload = store.load_state()
    world_state = _world_state_from_workspace_payload(state_payload)
    general_state = load_general_state(world_state)
    active_session_id = session_id or config.active_session_id
    session = load_or_create_session(store, session_id=active_session_id)
    frame = _active_decision_frame_from_general_state(general_state)
    compiler = load_workspace_context_compiler(project_root, config=config)
    context = compiler.compile_general_decision_context(
        world_state,
        general_state,
        current_intent=_context_current_intent(frame),
        active_decision_frame=frame,
        session=payload_value(session),
        config=config_payload,
        domain="general",
    )
    return payload_value(context)


def render_decision_context_text(payload: dict[str, Any]) -> str:
    frame = _mapping(payload.get("active_decision_frame"))
    return "\n".join(
        [
            "COMPILED DECISION CONTEXT",
            f"context_id: {payload.get('id') or ''}",
            f"context_type: {payload.get('context_type') or ''}",
            f"current_intent: {_shorten(str(_mapping(payload.get('current_intent')).get('text') or ''), 120)}",
            f"active_decision: {frame.get('decision_id') or ''}",
            f"selected: {_context_selected_summary(frame)}",
            f"recent_decisions: {len(_list(payload.get('recent_decisions')))}",
            f"recent_approvals: {len(_list(payload.get('recent_approvals')))}",
            f"recent_outcomes: {len(_list(payload.get('recent_outcomes')))}",
            f"retrieved_memory: {len(_list(payload.get('retrieved_memory')))}",
            f"executor: {_executor_context_summary(_mapping(payload.get('executor_affordance')))}",
            f"summary: {_summary_context_summary(_mapping(payload.get('session_summary')))}",
            f"session: {_session_context_summary(_mapping(payload.get('session_summary')))}",
            f"workspace: {_workspace_context_summary(_mapping(payload.get('workspace_context')))}",
            "Use `spice context --json` to inspect the exact payload.",
        ]
    )


def _active_decision_frame_from_general_state(general_state: Any) -> dict[str, Any]:
    metadata = getattr(general_state, "metadata", None)
    if not isinstance(metadata, dict):
        return {}
    frame = metadata.get("active_decision_frame")
    return dict(frame) if isinstance(frame, dict) else {}


def _context_current_intent(frame: dict[str, Any]) -> dict[str, Any]:
    raw_input = frame.get("input") if isinstance(frame.get("input"), dict) else {}
    text = str(raw_input.get("text") or "").strip()
    return {
        "text": text,
        "source": str(frame.get("source") or "context_debug"),
        "kind": "context_debug",
        "run_intent_mode": str(frame.get("run_intent_mode") or ""),
        "display_language": str(frame.get("display_language") or ""),
        "decision_id": str(frame.get("decision_id") or ""),
        "run_id": str(frame.get("run_id") or ""),
    }


def _world_state_from_workspace_payload(payload: dict[str, Any]) -> WorldState:
    world_payload = payload.get("world_state")
    if not isinstance(world_payload, dict):
        raise ValueError("Workspace state must contain a world_state object.")
    return WorldState(
        id=str(world_payload.get("id") or "worldstate.local"),
        schema_version=str(world_payload.get("schema_version", "0.1")),
        status=str(world_payload.get("status", "current")),
        entities=_mapping(world_payload.get("entities")),
        relations=_list_of_mappings(world_payload.get("relations")),
        goals=_list_of_mappings(world_payload.get("goals")),
        constraints=_list_of_mappings(world_payload.get("constraints")),
        signals=_list_of_mappings(world_payload.get("signals")),
        risks=_list_of_mappings(world_payload.get("risks")),
        active_intents=_list_of_mappings(world_payload.get("active_intents")),
        recent_outcomes=_list_of_mappings(world_payload.get("recent_outcomes")),
        resources=_mapping(world_payload.get("resources")),
        confidence=_mapping(world_payload.get("confidence")),
        provenance=_mapping(world_payload.get("provenance")),
        domain_state=_mapping(world_payload.get("domain_state")),
    )


def _context_selected_summary(frame: dict[str, Any]) -> str:
    selected = frame.get("selected") if isinstance(frame.get("selected"), dict) else {}
    label = str(selected.get("label") or "").strip()
    title = str(selected.get("title") or selected.get("recommended_action") or "").strip()
    candidate_id = str(selected.get("candidate_id") or frame.get("selected_candidate_id") or "").strip()
    parts = [part for part in [label, _shorten(title, 80), candidate_id] if part]
    return " | ".join(parts)


def _executor_context_summary(payload: dict[str, Any]) -> str:
    parts = [
        str(payload.get("executor") or ""),
        str(payload.get("status") or ""),
        str(payload.get("permission_mode") or ""),
    ]
    return " ".join(part for part in parts if part).strip()


def _session_context_summary(payload: dict[str, Any]) -> str:
    parts = []
    session_id = str(payload.get("session_id") or "")
    if session_id:
        parts.append(session_id)
    decisions = payload.get("decision_count")
    if decisions is not None:
        parts.append(f"decisions={decisions}")
    return " ".join(part for part in parts if part).strip()


def _summary_context_summary(payload: dict[str, Any]) -> str:
    rolling = payload.get("rolling_summary")
    if not isinstance(rolling, dict):
        return "none"
    summary_type = str(rolling.get("summary_type") or "deterministic")
    updated = str(rolling.get("updated_at") or "")
    model = rolling.get("model") if isinstance(rolling.get("model"), dict) else {}
    model_id = str(model.get("model_id") or "")
    parts = [summary_type]
    if model_id:
        parts.append(model_id)
    if updated:
        parts.append(f"updated={updated}")
    return " ".join(parts)


def _workspace_context_summary(payload: dict[str, Any]) -> str:
    parts = [
        f"memory={payload.get('memory_provider')}" if payload.get("memory_provider") is not None else "",
        f"compiler={payload.get('context_compiler')}" if payload.get("context_compiler") is not None else "",
        f"summary={payload.get('memory_summary_provider')}" if payload.get("memory_summary_provider") is not None else "",
        f"executor={payload.get('executor')}" if payload.get("executor") is not None else "",
    ]
    return " ".join(part for part in parts if part)


def _shorten(text: str, limit: int) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 1)].rstrip() + "…"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _list_of_mappings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]
