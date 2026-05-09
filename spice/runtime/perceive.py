from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from spice.decision.general import (
    GenericObservation,
    ObservationConfidence,
    ObservationEvidence,
    ObservationKind,
    ObservationSource,
    ObservationSubject,
    load_general_state,
    reduce_generic_observations,
    store_general_state,
)
from spice.decision.general.types import payload_value
from spice.perception.providers.open_chronicle import (
    OpenChroniclePerceptionProvider,
    OpenChronicleResult,
)
from spice.perception.providers.poll import PollPerceptionProvider, PollResult
from spice.protocols import WorldState
from spice.runtime.providers import RuntimeProviderDescriptor
from spice.runtime.run_once import RunOnceResult, run_decision_loop_from_observations
from spice.runtime.store import LocalJsonStore
from spice.runtime.workspace import SpiceWorkspacePaths, load_workspace_config, require_workspace


PERCEPTION_BUILDER = "spice.runtime.perceive"
PROVIDER_STATE_FILENAMES = {
    "open_chronicle": "open_chronicle_state.json",
    "poll": "poll_state.json",
}
SUPPORTED_PERCEPTION_PROVIDERS = frozenset({"open_chronicle", "poll"})
PerceptionResult = PollResult | OpenChronicleResult


@dataclass(slots=True)
class PerceiveResult:
    artifact: dict[str, Any]
    rendered_text: str
    perception_path: Path
    state_path: Path


def perceive_once(
    *,
    project_root: str | Path = ".",
    provider: str | None = None,
    poll_url: str | None = None,
    poll_command: str | None = None,
    openchronicle_mcp_url: str | None = None,
    openchronicle_since_minutes: int | None = None,
    openchronicle_context_limit: int | None = None,
    allow_command_poll: bool | None = None,
    decide_on_change: bool | None = None,
    timeout_seconds: int | None = None,
    now: datetime | None = None,
) -> PerceiveResult:
    created = now or datetime.now(timezone.utc)
    paths = require_workspace(project_root)
    store = LocalJsonStore(paths)
    config = load_workspace_config(project_root)
    provider_id = provider or _configured_provider(
        config,
        poll_url=poll_url,
        poll_command=poll_command,
        openchronicle_mcp_url=openchronicle_mcp_url,
        openchronicle_since_minutes=openchronicle_since_minutes,
        openchronicle_context_limit=openchronicle_context_limit,
    )
    if provider_id not in SUPPORTED_PERCEPTION_PROVIDERS:
        valid = ", ".join(sorted(SUPPORTED_PERCEPTION_PROVIDERS))
        raise ValueError(f"Unsupported perception provider: {provider_id}. Supported values: {valid}.")
    trigger_mode = (
        "decision_on_change"
        if decide_on_change is True
        else str(getattr(config, "perception_trigger_mode", "state_only") or "state_only")
    )
    if trigger_mode not in {"state_only", "decision_on_change"}:
        raise ValueError("perception_trigger_mode must be one of: state_only, decision_on_change.")
    timeout = timeout_seconds if timeout_seconds is not None else int(config.perception_poll_timeout)
    provider_state = _load_provider_state(paths, provider_id)
    previous_hashes = _dict(provider_state.get("hashes"))
    perception_provider = _build_perception_provider(
        provider_id,
        config=config,
        poll_url=poll_url,
        poll_command=poll_command,
        openchronicle_mcp_url=openchronicle_mcp_url,
        openchronicle_since_minutes=openchronicle_since_minutes,
        openchronicle_context_limit=openchronicle_context_limit,
        allow_command_poll=allow_command_poll,
        timeout_seconds=timeout,
        previous_hashes=previous_hashes,
        now=created,
    )
    results = perception_provider.poll_results()
    observations = [
        result.observation for result in results if result.observation is not None
    ]

    for result in results:
        previous_hashes[f"{result.source_type}:{result.source_value}"] = result.content_hash
    provider_state_payload = {
        "schema_version": f"spice.perception.{provider_id}_state.v1",
        "updated_at": _timestamp(created),
        "provider": provider_id,
        "hashes": previous_hashes,
        "sources": {
            f"{result.source_type}:{result.source_value}": {
                "source_type": result.source_type,
                "source_value": result.source_value,
                "content_hash": result.content_hash,
                "changed": result.changed,
                "last_observation_id": (
                    result.observation.observation_id if result.observation is not None else ""
                ),
            }
            for result in results
        },
    }
    _save_provider_state(paths, provider_id, provider_state_payload)

    state_payload = store.load_state()
    state_before_hash = _hash(state_payload)[:12]
    world_state = _world_state_from_workspace_payload(state_payload)
    state_before = load_general_state(world_state)
    decision_result: RunOnceResult | None = None
    decision_triggered = bool(observations) and trigger_mode == "decision_on_change"
    if decision_triggered:
        trigger_observations = [
            *observations,
            _decision_trigger_observation(
                results,
                input_text=_decision_trigger_text(results),
                now=created,
            ),
        ]
        decision_result = run_decision_loop_from_observations(
            trigger_observations,
            input_text=_decision_trigger_text(results),
            project_root=project_root,
            now=created,
            persist=True,
            full_loop_preview=True,
            run_intent_mode="act",
            source="perception_trigger",
            input_kind=f"{provider_id}_signal",
            input_source=provider_id,
            path_type="perception_trigger_run_once",
            generated_by="spice.runtime.perceive",
            decision_prefix="decision.perception",
            trace_prefix="trace.perception",
            run_prefix="run.perception",
            perception_descriptor=_perception_descriptor(provider_id),
        )
        state_after_payload = store.load_state()
        state_after_hash = _hash(state_after_payload)[:12]
        state_after = load_general_state(_world_state_from_workspace_payload(state_after_payload))
    else:
        state_after = reduce_generic_observations(state_before, observations)
        state_after_payload = _workspace_state_payload(world_state, state_after)
        state_after_hash = _hash(state_after_payload)[:12]
        if observations:
            store.save_state(state_after_payload)

    perception_id = _make_perception_id(results, created, store=store, provider_id=provider_id)
    perception_path = store.record_path("perception", perception_id)
    artifact = {
        "path_type": f"runtime_perception_{provider_id}",
        "generated_by": PERCEPTION_BUILDER,
        "created_at": _timestamp(created),
        "perception_id": perception_id,
        "provider": provider_id,
        "source": provider_id,
        "source_count": len(results),
        "observation_count": len(observations),
        "changed_count": sum(1 for result in results if result.changed),
        "deduped_count": sum(1 for result in results if not result.changed),
        "trigger_mode": trigger_mode,
        "decision_triggered": decision_triggered,
        "triggered_run": decision_result.artifact if decision_result is not None else None,
        "decision_id": decision_result.artifact.get("decision_id") if decision_result is not None else None,
        "run_id": decision_result.artifact.get("run_id") if decision_result is not None else None,
        "selected_candidate_id": (
            decision_result.artifact.get("selected_candidate_id") if decision_result is not None else None
        ),
        "approval_id": decision_result.artifact.get("approval_id") if decision_result is not None else None,
        "rendered_decision_text": (
            decision_result.rendered_text if decision_result is not None else None
        ),
        "executor_called": False,
        "sdep_request_sent": False,
        "executed": False,
        "state_updated": bool(observations),
        "persisted": True,
        "state_before_ref": f"{_workspace_relative(paths.state)}#before:{state_before_hash}",
        "state_after_ref": f"{_workspace_relative(paths.state)}#after:{state_after_hash}",
        "provider_state_ref": _workspace_relative(_provider_state_path(paths, provider_id)),
        "store_paths": {
            "perception": _workspace_relative(perception_path),
            "state": _workspace_relative(paths.state),
            "provider_state": _workspace_relative(_provider_state_path(paths, provider_id)),
        },
        "results": [result.to_payload() for result in results],
        "observations": [observation.to_payload() for observation in observations],
        "state_before_summary": _state_summary(state_before),
        "state_after_summary": _state_summary(state_after),
    }
    artifact["rendered_text"] = render_perceive_text(artifact)
    perception_path = store.save_perception(perception_id, artifact)
    return PerceiveResult(
        artifact=artifact,
        rendered_text=artifact["rendered_text"],
        perception_path=perception_path,
        state_path=paths.state,
    )


def perceive_watch(
    *,
    project_root: str | Path = ".",
    provider: str | None = None,
    poll_url: str | None = None,
    poll_command: str | None = None,
    openchronicle_mcp_url: str | None = None,
    openchronicle_since_minutes: int | None = None,
    openchronicle_context_limit: int | None = None,
    allow_command_poll: bool | None = None,
    decide_on_change: bool | None = None,
    timeout_seconds: int | None = None,
    interval_seconds: int | None = None,
    max_iterations: int | None = None,
) -> list[PerceiveResult]:
    config = load_workspace_config(project_root)
    interval = interval_seconds if interval_seconds is not None else int(config.perception_poll_interval)
    if interval <= 0:
        raise ValueError("perception poll interval must be positive.")
    results: list[PerceiveResult] = []
    iteration = 0
    while True:
        results.append(
            perceive_once(
                project_root=project_root,
                provider=provider,
                poll_url=poll_url,
                poll_command=poll_command,
                openchronicle_mcp_url=openchronicle_mcp_url,
                openchronicle_since_minutes=openchronicle_since_minutes,
                openchronicle_context_limit=openchronicle_context_limit,
                allow_command_poll=allow_command_poll,
                decide_on_change=decide_on_change,
                timeout_seconds=timeout_seconds,
            )
        )
        iteration += 1
        if max_iterations is not None and iteration >= max_iterations:
            break
        time.sleep(interval)
    return results


def render_perceive_text(artifact: dict[str, Any]) -> str:
    provider = str(artifact.get("provider") or "perception")
    lines = [
        "SPICE PERCEPTION",
        f"{provider} -> GenericObservation -> General state",
        (
            f"decision_triggered: {str(bool(artifact.get('decision_triggered'))).lower()} "
            "| executor_called: false | sdep_request_sent: false"
        ),
        "",
        f"perception_id: {artifact.get('perception_id')}",
        f"provider: {artifact.get('provider')}",
        f"observation_count: {artifact.get('observation_count')}",
        f"changed_count: {artifact.get('changed_count')}",
        f"deduped_count: {artifact.get('deduped_count')}",
        f"trigger_mode: {artifact.get('trigger_mode')}",
        f"decision_triggered: {str(bool(artifact.get('decision_triggered'))).lower()}",
        "",
        "STATE",
        f"- state_updated: {str(bool(artifact.get('state_updated'))).lower()}",
        f"- persisted: {str(bool(artifact.get('persisted'))).lower()}",
        f"- state_after_ref: {artifact.get('state_after_ref')}",
    ]
    observations = artifact.get("observations")
    if isinstance(observations, list) and observations:
        lines.extend(["", "OBSERVATIONS"])
        for observation in observations[:5]:
            if isinstance(observation, dict):
                lines.append(f"- {observation.get('observation_id')}: {observation.get('summary')}")
    if artifact.get("decision_triggered"):
        lines.extend(
            [
                "",
                "TRIGGERED DECISION",
                f"- run_id: {artifact.get('run_id')}",
                f"- decision_id: {artifact.get('decision_id')}",
                f"- approval_id: {artifact.get('approval_id')}",
                "- executor_called: false",
                "- sdep_request_sent: false",
            ]
        )
    return "\n".join(lines)


def _load_provider_state(paths: SpiceWorkspacePaths, provider_id: str) -> dict[str, Any]:
    path = _provider_state_path(paths, provider_id)
    if not path.exists():
        return {
            "schema_version": f"spice.perception.{provider_id}_state.v1",
            "provider": provider_id,
            "hashes": {},
            "sources": {},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _save_provider_state(paths: SpiceWorkspacePaths, provider_id: str, payload: dict[str, Any]) -> Path:
    path = _provider_state_path(paths, provider_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _provider_state_path(paths: SpiceWorkspacePaths, provider_id: str) -> Path:
    filename = PROVIDER_STATE_FILENAMES.get(provider_id)
    if filename is None:
        filename = f"{provider_id}_state.json"
    return paths.perceptions_dir / filename


def _decision_trigger_text(results: list[PerceptionResult]) -> str:
    changed = [result for result in results if result.changed]
    previews = []
    for result in changed[:3]:
        previews.append(f"{result.source_type} {result.source_value}: {result.content[:200]}")
    summary = " | ".join(" ".join(preview.split()) for preview in previews)
    if not summary:
        summary = "new perception signal"
    return f"Review new perception signal and decide whether action is needed: {summary}"


def _decision_trigger_observation(
    results: list[PerceptionResult],
    *,
    input_text: str,
    now: datetime,
) -> GenericObservation:
    changed = [result for result in results if result.changed]
    changed_refs = [
        f"{result.source_type}:{result.source_value}" for result in changed
    ]
    seed = [input_text, changed_refs, _timestamp(now)]
    slug = _hash(seed)[:12]
    providers = sorted(
        {
            str((result.metadata or {}).get("provider") or (result.metadata or {}).get("source") or "")
            for result in changed
        }
    )
    provider = "open_chronicle" if any("open_chronicle" in item for item in providers) else "poll"
    intent_id = f"intent.perception.{provider}.{slug}"
    return GenericObservation(
        observation_id=f"obs.perception.trigger.intent.{slug}",
        kind=ObservationKind.INTENT,
        source=ObservationSource(
            provider=provider,
            channel="perception_trigger",
            actor="spice",
            received_at=_timestamp(now),
            metadata={"trigger_mode": "decision_on_change"},
        ),
        subject=ObservationSubject(
            subject_id=intent_id,
            subject_type="intent",
            title="Review new poll signal",
            refs=[intent_id, *changed_refs],
        ),
        summary=input_text,
        attributes={
            "intent_id": intent_id,
            "desired_outcome": "Decide whether action is needed for new poll signals.",
            "original_text": input_text,
            "urgency": "medium",
            "status": "active",
            "target_refs": changed_refs,
            "derived_from": f"{provider}_perception",
        },
        evidence=[
            ObservationEvidence(
                evidence_id=f"evidence.perception.trigger.intent.{slug}",
                kind="perception_trigger",
                summary="Synthetic intent created from changed perception observations.",
                content=input_text,
                metadata={"changed_refs": changed_refs},
            )
        ],
        confidence=ObservationConfidence(score=0.8, level="high"),
        refs=changed_refs,
        metadata={
            "source": provider,
            "trigger_mode": "decision_on_change",
            "changed_refs": changed_refs,
        },
    )


def _perception_descriptor(provider_id: str) -> dict[str, Any]:
    if provider_id == "open_chronicle":
        return RuntimeProviderDescriptor(
            provider_id="open_chronicle",
            provider_type="perception",
            implementation="spice.perception.providers.open_chronicle.OpenChroniclePerceptionProvider",
            metadata={
                "input": "Open Chronicle MCP current_context and recent_activity",
                "output": "GenericObservation[]",
                "external_calls": True,
                "trigger_capable": True,
            },
        ).to_payload()
    return RuntimeProviderDescriptor(
        provider_id="poll",
        provider_type="perception",
        implementation="spice.perception.providers.poll.PollPerceptionProvider",
        metadata={
            "input": "url or command poll source",
            "output": "GenericObservation[]",
            "external_calls": True,
            "command_poll_requires_opt_in": True,
            "trigger_capable": True,
        },
    ).to_payload()


def _configured_provider(
    config: Any,
    *,
    poll_url: str | None,
    poll_command: str | None,
    openchronicle_mcp_url: str | None,
    openchronicle_since_minutes: int | None,
    openchronicle_context_limit: int | None,
) -> str:
    if poll_url is not None or poll_command is not None:
        return "poll"
    if (
        openchronicle_mcp_url is not None
        or openchronicle_since_minutes is not None
        or openchronicle_context_limit is not None
    ):
        return "open_chronicle"
    return str(getattr(config, "perception_provider", "manual") or "manual")


def _build_perception_provider(
    provider_id: str,
    *,
    config: Any,
    poll_url: str | None,
    poll_command: str | None,
    openchronicle_mcp_url: str | None,
    openchronicle_since_minutes: int | None,
    openchronicle_context_limit: int | None,
    allow_command_poll: bool | None,
    timeout_seconds: int,
    previous_hashes: dict[str, str],
    now: datetime,
) -> Any:
    if provider_id == "poll":
        url = poll_url if poll_url is not None else config.perception_poll_url
        command = poll_command if poll_command is not None else config.perception_poll_command
        allow_command = (
            allow_command_poll
            if allow_command_poll is not None
            else _as_bool(config.perception_allow_command_poll)
        )
        return PollPerceptionProvider(
            url=url,
            command=command,
            allow_command_poll=allow_command,
            timeout_seconds=timeout_seconds,
            previous_hashes=previous_hashes,
            now=now,
        )
    if provider_id == "open_chronicle":
        return OpenChroniclePerceptionProvider(
            mcp_url=(
                openchronicle_mcp_url
                if openchronicle_mcp_url is not None
                else str(getattr(config, "openchronicle_mcp_url", "") or "")
            ),
            since_minutes=(
                openchronicle_since_minutes
                if openchronicle_since_minutes is not None
                else int(getattr(config, "openchronicle_since_minutes", "15") or "15")
            ),
            context_limit=(
                openchronicle_context_limit
                if openchronicle_context_limit is not None
                else int(getattr(config, "openchronicle_context_limit", "5") or "5")
            ),
            timeout_seconds=timeout_seconds,
            previous_hashes=previous_hashes,
            now=now,
        )
    raise ValueError(f"Unsupported perception provider: {provider_id}.")


def _workspace_state_payload(
    world_state: WorldState,
    state: Any,
) -> dict[str, Any]:
    store_general_state(world_state, state)
    return {
        "schema_version": "spice.workspace.state.v1",
        "world_state": payload_value(world_state),
    }


def _world_state_from_workspace_payload(payload: dict[str, Any]) -> WorldState:
    world_payload = payload.get("world_state")
    if not isinstance(world_payload, dict):
        raise ValueError("Workspace state must contain a world_state object.")
    return WorldState(
        id=str(world_payload.get("id") or "worldstate.local"),
        schema_version=str(world_payload.get("schema_version", "0.1")),
        status=str(world_payload.get("status", "current")),
        entities=_dict(world_payload.get("entities")),
        relations=_list_of_dicts(world_payload.get("relations")),
        goals=_list_of_dicts(world_payload.get("goals")),
        constraints=_list_of_dicts(world_payload.get("constraints")),
        resources=_dict(world_payload.get("resources")),
        risks=_list_of_dicts(world_payload.get("risks")),
        signals=_list_of_dicts(world_payload.get("signals")),
        active_intents=_list_of_dicts(world_payload.get("active_intents")),
        recent_outcomes=_list_of_dicts(world_payload.get("recent_outcomes")),
        confidence=_dict(world_payload.get("confidence")),
        provenance=_dict(world_payload.get("provenance")),
        domain_state=_dict(world_payload.get("domain_state")),
    )


def _state_summary(state: Any) -> dict[str, int]:
    return {
        "observations": len(getattr(state, "observations", []) or []),
        "signals": len(getattr(state, "signals", []) or []),
        "intents": len(getattr(state, "intents", []) or []),
        "work_items": len(getattr(state, "work_items", []) or []),
        "outcomes": len(getattr(state, "outcomes", []) or []),
    }


def _make_perception_id(
    results: list[PerceptionResult],
    created: datetime,
    *,
    store: LocalJsonStore,
    provider_id: str,
) -> str:
    seed = [
        _timestamp(created),
        [
            {
                "source_type": result.source_type,
                "source_value": result.source_value,
                "content_hash": result.content_hash,
                "changed": result.changed,
            }
            for result in results
        ],
    ]
    base = f"perception.{provider_id}.{created.strftime('%Y%m%dT%H%M%S%fZ')}.{_hash(seed)[:12]}"
    candidate = base
    suffix = 2
    while store.record_path("perception", candidate).exists():
        candidate = f"{base}.{suffix}"
        suffix += 1
    return candidate


def _workspace_relative(path: Path) -> str:
    parts = path.parts
    if ".spice" in parts:
        index = parts.index(".spice")
        return str(Path(*parts[index:]))
    return str(path)


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _hash(value: Any) -> str:
    return sha256(json.dumps(payload_value(value), sort_keys=True).encode("utf-8")).hexdigest()


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
