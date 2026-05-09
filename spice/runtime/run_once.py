from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from spice.decision.compare import render_compare_text
from spice.decision.compare_payload import normalize_compare_payload
from spice.decision.general import (
    GeneralDecisionState,
    GenericObservation,
    GenericPolicyAdapter,
    generate_generic_candidates,
    load_general_state,
    reduce_generic_observations,
    store_general_state,
)
from spice.decision.general.types import payload_value
from spice.llm.candidate_expander import (
    build_explicit_option_candidates,
    expand_candidates_from_runtime_config,
    extract_explicit_options,
    merge_expanded_candidates,
)
from spice.llm.simulation_runner import simulate_candidates_from_runtime_config
from spice.runtime.active_decision_frame import (
    attach_active_decision_frame,
    build_active_decision_frame,
)
from spice.runtime.execution_affordance import annotate_execution_affordances
from spice.protocols import WorldState
from spice.runtime.full_loop_preview import build_runtime_full_loop_preview
from spice.language import detect_display_language
from spice.runtime.memory_writeback import (
    skipped_general_decision_memory_writeback,
    write_general_decision_memory,
)
from spice.runtime.providers import ManualInputProvider, default_runtime_provider_descriptors
from spice.runtime.session import DEFAULT_SESSION_ID, append_run_to_session, load_or_create_session
from spice.runtime.skill_resolution import (
    annotate_skill_resolutions,
    selected_skill_resolution_payload,
)
from spice.runtime.simulation_targets import (
    merge_simulation_result_candidates,
    select_simulation_targets,
)
from spice.runtime.store import LocalJsonStore
from spice.runtime.workspace import (
    SpiceWorkspaceConfig,
    SpiceWorkspacePaths,
    load_workspace_context_compiler,
    load_workspace_env,
    load_workspace_memory_provider,
    workspace_paths,
)


@dataclass(slots=True)
class RunOnceResult:
    artifact: dict[str, Any]
    rendered_text: str
    run_path: Path
    decision_path: Path
    approval_path: Path | None
    session_path: Path
    state_path: Path


def run_once(
    intent: str,
    *,
    project_root: str | Path = ".",
    now: datetime | None = None,
    use_bars: bool = False,
    persist: bool = True,
    session_id: str = DEFAULT_SESSION_ID,
    full_loop_preview: bool = True,
    run_intent_mode: str = "auto",
) -> RunOnceResult:
    """Run one manual-intent General decision loop.

    This is the first product-facing runtime entrance. It persists local state
    and artifacts by default, renders a full read-only execution handoff
    preview, but does not send SDEP or call an executor.
    """

    text = intent.strip()
    if not text:
        raise ValueError("run --once requires a non-empty intent.")

    created = now or datetime.now(timezone.utc)
    paths = workspace_paths(project_root)
    _require_workspace(paths)
    load_workspace_env(project_root)
    config = _load_config(paths)
    perception_provider = ManualInputProvider()
    observations = perception_provider.collect_observations(text, config=config, now=created)
    return run_decision_loop_from_observations(
        observations,
        input_text=text,
        project_root=project_root,
        now=created,
        use_bars=use_bars,
        persist=persist,
        session_id=session_id,
        full_loop_preview=full_loop_preview,
        run_intent_mode=run_intent_mode,
        source="manual_intent",
        input_kind="manual_intent",
        input_source="cli",
        path_type="manual_intent_run_once",
        generated_by="spice.runtime.run_once",
        decision_prefix="decision.manual",
        trace_prefix="trace.manual",
        run_prefix="run.manual",
        perception_descriptor=perception_provider.descriptor().to_payload(),
    )


def run_decision_loop_from_observations(
    observations: list[GenericObservation],
    *,
    input_text: str,
    project_root: str | Path = ".",
    now: datetime | None = None,
    use_bars: bool = False,
    persist: bool = True,
    session_id: str = DEFAULT_SESSION_ID,
    full_loop_preview: bool = True,
    run_intent_mode: str = "auto",
    source: str = "observations",
    input_kind: str = "observations",
    input_source: str = "runtime",
    path_type: str = "observation_run_once",
    generated_by: str = "spice.runtime.run_once",
    decision_prefix: str = "decision.observation",
    trace_prefix: str = "trace.observation",
    run_prefix: str = "run.observation",
    perception_descriptor: dict[str, Any] | None = None,
) -> RunOnceResult:
    """Run the product decision loop from already-normalized observations."""

    text = input_text.strip()
    if not text:
        raise ValueError("run_decision_loop_from_observations requires non-empty input_text.")
    normalized_observations = list(observations)
    if not normalized_observations:
        raise ValueError("run_decision_loop_from_observations requires at least one observation.")
    mode = _normalize_run_intent_mode(run_intent_mode)
    display_language = detect_display_language(text)
    effective_full_loop_preview = full_loop_preview and mode != "advise"

    created = now or datetime.now(timezone.utc)
    paths = workspace_paths(project_root)
    _require_workspace(paths)
    load_workspace_env(project_root)
    store = LocalJsonStore(paths)
    config = _load_config(paths)
    workspace_config = SpiceWorkspaceConfig.from_payload(config)
    session = load_or_create_session(store, session_id=session_id, now=created)
    state_payload = store.load_state()
    state_before_hash = _hash(state_payload)[:12]
    world_state = _world_state_from_workspace_payload(state_payload)
    state_before = load_general_state(world_state)

    state_after = reduce_generic_observations(state_before, normalized_observations)
    context_compiler = load_workspace_context_compiler(
        project_root,
        config=workspace_config,
    )
    decision_context = context_compiler.compile_general_decision_context(
        world_state,
        state_after,
        current_intent={
            "text": text,
            "source": source,
            "kind": input_kind,
            "run_intent_mode": mode,
            "display_language": display_language,
        },
        session=payload_value(session),
        config=config,
        domain="general",
    )
    candidates = generate_generic_candidates(state_after)
    explicit_options = extract_explicit_options(text)
    candidate_expansion = expand_candidates_from_runtime_config(
        config=config,
        state=state_after,
        intent_text=text,
        rule_candidates=candidates,
        display_language=display_language,
        explicit_options=explicit_options,
        decision_context=decision_context,
    )
    candidates = merge_expanded_candidates(candidates, candidate_expansion)
    if explicit_options:
        candidates = _merge_explicit_option_fallback_candidates(
            candidates,
            build_explicit_option_candidates(
                intent_text=text,
                state=state_after,
                explicit_options=explicit_options,
            ),
            explicit_options=explicit_options,
        )
    simulation_targets = select_simulation_targets(
        candidates,
        explicit_options=explicit_options,
    )
    simulation_context = context_compiler.compile_general_simulation_context(
        world_state,
        state_after,
        current_intent=decision_context.current_intent,
        candidates=simulation_targets,
        active_decision_frame=decision_context.active_decision_frame,
        session=payload_value(session),
        config=config,
        domain="general",
    )
    candidate_simulation = simulate_candidates_from_runtime_config(
        config=config,
        state=state_after,
        intent_text=text,
        candidates=simulation_targets,
        display_language=display_language,
        simulation_context=simulation_context,
    )
    candidate_simulation = merge_simulation_result_candidates(candidates, candidate_simulation)
    candidates = annotate_execution_affordances(
        candidate_simulation.candidates,
        config=config,
    )
    candidates = annotate_skill_resolutions(candidates, config=config)
    if not candidates:
        raise ValueError("No generic candidates were generated for this intent.")
    focused_candidates = _focus_candidates_for_observations(
        candidates,
        observations=normalized_observations,
    )
    candidate_selection = _candidate_selection_for_run_mode(
        focused_candidates,
        mode=mode,
        explicit_options=explicit_options,
    )
    evaluation_candidates = list(focused_candidates)
    handoff_blocked = bool(candidate_selection["handoff_blocked"])
    handoff_blockers = list(candidate_selection["handoff_blockers"])
    selection_candidate_ids = candidate_selection["selection_candidate_ids"]

    decision_id = _stable_id(decision_prefix, text, created)
    trace_ref = _stable_id(trace_prefix, text, created)
    policy = GenericPolicyAdapter.from_decision_profile(paths.decision_profile)
    policy_result = policy.evaluate(
        state_after,
        candidates=evaluation_candidates,
        decision_id=decision_id,
        trace_ref=trace_ref,
        run_intent_mode=mode,
        selection_candidate_ids=selection_candidate_ids,
        selection_pool_reason=candidate_selection["selection_pool_reason"],
    )
    checkpoint = policy_result.checkpoint
    state_after.decision_checkpoints.append(checkpoint)
    state_after.trace_refs.append(trace_ref)
    if checkpoint.approval is not None:
        state_after.approvals.append(checkpoint.approval)

    runtime_warnings = _runtime_warning_payloads(
        candidate_expansion=candidate_expansion.to_payload(),
        candidate_simulation=candidate_simulation.to_payload(),
    )
    compare_payload = _attach_compare_warnings(
        policy_result.compare_payload,
        runtime_warnings,
    )
    selected = compare_payload["selected_recommendation"]
    run_id = _make_run_id(
        text,
        created,
        store=store,
        prefix=run_prefix,
    )
    approval_id = checkpoint.approval.approval_id if checkpoint.approval else None
    selected_skill_resolution = selected_skill_resolution_payload(
        evaluation_candidates,
        selected_candidate_id=checkpoint.selected_candidate_id,
    )
    selection_pool = {
        "kind": candidate_selection["selection_pool_kind"],
        "candidate_ids": sorted(selection_candidate_ids or []),
        "reason": candidate_selection["selection_pool_reason"],
    }
    active_decision_frame = build_active_decision_frame(
        compare_payload=compare_payload,
        run_id=run_id,
        session_id=session.session_id,
        input_text=text,
        created_at=_timestamp(created),
        run_intent_mode=mode,
        display_language=display_language,
        approval_id=approval_id,
        selection_pool=selection_pool,
        handoff_blocked=handoff_blocked,
        handoff_blockers=handoff_blockers,
        source=source,
    )
    attach_active_decision_frame(state_after, active_decision_frame)

    state_after_payload = _workspace_state_payload(world_state, state_after)
    state_after_hash = _hash(state_after_payload)[:12]
    persisted_at = _timestamp(created) if persist else None
    if persist:
        store.save_state(state_after_payload)

    decision_card = render_compare_text(compare_payload, use_bars=use_bars)
    run_path = store.record_path("run", run_id)
    decision_path = store.record_path("decision", decision_id)
    approval_path = store.record_path("approval", approval_id) if approval_id else None
    session_path = store.record_path("session", session.session_id)
    state_before_ref = f"{_workspace_relative(paths.state)}#before:{state_before_hash}"
    state_after_ref = (
        f"{_workspace_relative(paths.state)}#after:{state_after_hash}"
        if persist
        else f"preview:{_workspace_relative(paths.state)}#after:{state_after_hash}"
    )
    store_paths = {
        "run": _workspace_relative(run_path),
        "decision": _workspace_relative(decision_path),
        "state": _workspace_relative(paths.state),
        "session": _workspace_relative(session_path),
    }
    if approval_path is not None:
        store_paths["approval"] = _workspace_relative(approval_path)
    artifact = {
        "path_type": path_type,
        "generated_by": generated_by,
        "created_at": _timestamp(created),
        "run_id": run_id,
        "session_id": session.session_id,
        "loop_mode": "full_loop_preview" if effective_full_loop_preview else "decision_only",
        "run_intent_mode": mode,
        "display_language": display_language,
        "handoff_required": mode == "act",
        "handoff_blocked": handoff_blocked,
        "handoff_blockers": handoff_blockers,
        "handoff_eligible_candidate_count": len(_approval_eligible_executable_candidates(candidates)),
        "selection_pool": selection_pool,
        "active_decision_frame": active_decision_frame,
        "evaluated_candidate_count": len(evaluation_candidates),
        "source": source,
        "decision_id": decision_id,
        "trace_ref": trace_ref,
        "selected_candidate_id": checkpoint.selected_candidate_id,
        "skill_resolution": selected_skill_resolution,
        "skill_resolution_status": selected_skill_resolution.get("status"),
        "skill_id": _selected_skill_id(selected_skill_resolution),
        "executor_id": _selected_skill_executor_id(selected_skill_resolution),
        "approval_id": approval_id,
        "state_before_ref": state_before_ref,
        "state_after_ref": state_after_ref,
        "store_paths": store_paths,
        "input": {
            "kind": input_kind,
            "text": text,
            "source": input_source,
            "display_language": display_language,
            "observation_ids": [
                observation.observation_id for observation in normalized_observations
            ],
        },
        "config": {
            "llm_provider": config.get("llm_provider"),
            "llm_model": config.get("llm_model"),
            "llm_candidate_expand": config.get("llm_candidate_expand"),
            "llm_simulation": config.get("llm_simulation"),
            "executor": config.get("executor"),
            "permission_mode": config.get("permission_mode"),
            "perception_provider": config.get("perception_provider"),
            "store": config.get("store"),
            "memory_provider": config.get("memory_provider"),
            "memory_path": config.get("memory_path"),
            "context_compiler": config.get("context_compiler"),
        },
        "providers": {
            **default_runtime_provider_descriptors(),
            **(
                {"perception": perception_descriptor}
                if isinstance(perception_descriptor, dict)
                else {}
            ),
        },
        "read_only_execution": True,
        "executor_called": False,
        "sdep_request_sent": False,
        "executed": False,
        "execution": None,
        "state_persisted": persist,
        "persisted": persist,
        "persist_mode": "active_state" if persist else "no_persist",
        "persisted_at": persisted_at,
        "artifacts_persisted": True,
        "observations": [observation.to_payload() for observation in normalized_observations],
        "state_before_summary": _state_summary(state_before),
        "state_after_summary": _state_summary(state_after),
        "context_refs": {
            "decision_context_id": decision_context.id,
            "simulation_context_id": simulation_context.id,
        },
        "compiled_context": {
            "decision_context": payload_value(decision_context),
            "simulation_context": payload_value(simulation_context),
        },
        "warnings": runtime_warnings,
        "raw_model_outputs": _raw_model_outputs(
            candidate_expansion=candidate_expansion.to_payload(),
            candidate_simulation=candidate_simulation.to_payload(),
        ),
        "llm_candidate_expansion": candidate_expansion.to_payload(),
        "llm_simulation": candidate_simulation.to_payload(),
        "candidates": [candidate.to_payload() for candidate in candidates],
        "evaluated_candidates": [
            candidate.to_payload() for candidate in evaluation_candidates
        ],
        "candidate_summary": _candidate_summary(
            candidates,
            selected_candidate_id=checkpoint.selected_candidate_id,
        ),
        "compare_payload": compare_payload,
        "rendered_text": render_run_once_text(
            intent=text,
            artifact={
                "run_id": run_id,
                "decision_id": decision_id,
                "trace_ref": trace_ref,
                "selected": selected,
                "state_after_summary": _state_summary(state_after),
                "candidate_count": len(candidates),
                "approval_id": approval_id,
                "executor_called": False,
                "sdep_request_sent": False,
                "state_persisted": persist,
                "persist_mode": "active_state" if persist else "no_persist",
                "run_intent_mode": mode,
                "handoff_required": mode == "act",
                "handoff_blocked": handoff_blocked,
                "handoff_blockers": handoff_blockers,
                "decision_card": decision_card,
                "llm_candidate_expansion": candidate_expansion.to_payload(),
                "llm_simulation": candidate_simulation.to_payload(),
            },
        ),
        "decision": policy_result.to_payload(),
        "approval": checkpoint.approval.to_payload() if checkpoint.approval else None,
    }
    if effective_full_loop_preview:
        preview = build_runtime_full_loop_preview(
            state=state_after,
            candidates=evaluation_candidates,
            policy_result=policy_result,
            config=config,
            now=created,
        )
        artifact["full_loop_preview"] = preview.artifact
        artifact["skill_id"] = preview.artifact.get("skill_id")
        artifact["executor_id"] = preview.artifact.get("executor_id")
        artifact["context_pack_id"] = preview.artifact.get("context_pack_id")
        artifact["execution_id"] = preview.artifact.get("execution_id")
        artifact["request_id"] = preview.artifact.get("request_id")
        artifact["outcome_id"] = preview.artifact.get("outcome_id")
        artifact["runtime_state_feedback"] = preview.artifact.get("state_feedback")
        artifact["rendered_text"] = render_run_once_full_loop_text(
            intent=text,
            artifact=artifact,
            decision_card=decision_card,
        )
    artifact["rendered_text"] = render_run_once_text(
        intent=text,
        artifact={
            "run_id": run_id,
            "decision_id": decision_id,
            "trace_ref": trace_ref,
            "selected": selected,
            "state_after_summary": artifact["state_after_summary"],
            "candidate_count": len(candidates),
            "approval_id": artifact["approval_id"],
            "executor_called": False,
            "sdep_request_sent": False,
            "state_persisted": persist,
            "persist_mode": artifact["persist_mode"],
            "run_intent_mode": mode,
            "handoff_required": mode == "act",
            "handoff_blocked": artifact["handoff_blocked"],
            "handoff_blockers": artifact["handoff_blockers"],
            "decision_card": decision_card,
            "llm_candidate_expansion": artifact["llm_candidate_expansion"],
            "llm_simulation": artifact["llm_simulation"],
        },
    ) if not effective_full_loop_preview else artifact["rendered_text"]

    saved_decision_path = store.save_decision(decision_id, artifact["decision"])
    saved_approval_path = (
        store.save_approval(artifact["approval_id"], artifact["approval"])
        if artifact["approval_id"] and isinstance(artifact["approval"], dict)
        else None
    )
    saved_session = append_run_to_session(store, session, artifact, now=created)
    artifact["session"] = saved_session.to_payload()
    if persist:
        memory_provider = load_workspace_memory_provider(
            project_root,
            config=workspace_config,
        )
        artifact["memory_writeback"] = write_general_decision_memory(
            memory_provider,
            artifact=artifact,
            config=workspace_config.to_payload(),
        )
    else:
        artifact["memory_writeback"] = skipped_general_decision_memory_writeback(
            reason="persist=false",
        )
    saved_run_path = store.save_run(run_id, artifact)
    return RunOnceResult(
        artifact=artifact,
        rendered_text=str(artifact["rendered_text"]),
        run_path=saved_run_path,
        decision_path=saved_decision_path,
        approval_path=saved_approval_path,
        session_path=session_path,
        state_path=paths.state,
    )


def build_manual_intent_observations(
    intent: str,
    *,
    config: dict[str, Any],
    now: datetime,
) -> list[GenericObservation]:
    return ManualInputProvider().collect_observations(intent, config=config, now=now)


def render_run_once_text(*, intent: str, artifact: dict[str, Any]) -> str:
    selected = artifact["selected"]
    summary = artifact["state_after_summary"]
    lines = [
        "SPICE RUN ONCE",
        "manual intent -> General state -> candidates -> decision card",
        "no executor called | no SDEP sent",
        "",
        f"mode: {artifact.get('run_intent_mode', 'auto')}",
        f"intent: {intent}",
        f"run_id: {artifact['run_id']}",
        f"decision_id: {artifact['decision_id']}",
        f"trace_ref: {artifact['trace_ref']}",
        "",
        "GENERAL STATE",
        f"- observations: {summary['observation_count']}",
        f"- intents: {summary['intent_count']}",
        f"- capabilities: {summary['capability_count']}",
        f"- constraints: {summary['constraint_count']}",
        "",
        "SELECTED DECISION",
        f"- {selected.get('title')} ({selected.get('candidate_id')})",
        f"- action: {selected.get('action')}",
        f"- approval_id: {artifact.get('approval_id')}",
        "",
        "BOUNDARY",
        f"- handoff_required: {str(bool(artifact.get('handoff_required'))).lower()}",
        f"- handoff_blocked: {str(bool(artifact.get('handoff_blocked'))).lower()}",
        f"- executor_called: {str(bool(artifact['executor_called'])).lower()}",
        f"- sdep_request_sent: {str(bool(artifact['sdep_request_sent'])).lower()}",
        f"- state_persisted: {str(bool(artifact['state_persisted'])).lower()}",
        f"- persist_mode: {artifact['persist_mode']}",
        "",
        "LLM",
        _llm_runtime_status_line("candidate_expansion", artifact.get("llm_candidate_expansion")),
        _llm_runtime_status_line("simulation", artifact.get("llm_simulation")),
        "",
        artifact["decision_card"],
    ]
    return "\n".join(lines)


def render_run_once_full_loop_text(
    *,
    intent: str,
    artifact: dict[str, Any],
    decision_card: str,
) -> str:
    preview = artifact.get("full_loop_preview")
    preview_text = (
        preview.get("rendered_text")
        if isinstance(preview, dict) and isinstance(preview.get("rendered_text"), str)
        else ""
    )
    lines = [
        "SPICE DECISION LOOP",
        "manual intent -> state -> candidates -> decision -> approval -> execution handoff -> outcome snapshot",
        "no executor called | no SDEP sent | no state feedback persisted",
        "",
        f"mode: {artifact.get('run_intent_mode', 'auto')}",
        f"intent: {intent}",
        f"run_id: {artifact['run_id']}",
        f"session_id: {artifact['session_id']}",
        f"decision_id: {artifact['decision_id']}",
        f"trace_ref: {artifact['trace_ref']}",
        "",
        "DECISION CARD",
        _llm_runtime_status_line("candidate_expansion", artifact.get("llm_candidate_expansion")),
        _llm_runtime_status_line("simulation", artifact.get("llm_simulation")),
        "",
        decision_card,
    ]
    if preview_text:
        lines.extend(["", preview_text])
    lines.extend(
        [
            "",
            "RUNTIME PERSISTENCE",
            f"- handoff_required: {str(bool(artifact.get('handoff_required'))).lower()}",
            f"- handoff_blocked: {str(bool(artifact.get('handoff_blocked'))).lower()}",
            f"- active_state_persisted: {str(bool(artifact['state_persisted'])).lower()}",
            f"- persist_mode: {artifact['persist_mode']}",
            "- full_loop_feedback_persisted: false",
        ]
    )
    return "\n".join(lines)


def _require_workspace(paths: SpiceWorkspacePaths) -> None:
    missing = [
        path
        for path in (paths.config, paths.decision_profile, paths.state)
        if not path.exists()
    ]
    if missing:
        rendered = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Spice workspace is not initialized. Missing: {rendered}. Run `spice setup` first."
        )


def _workspace_state_payload(
    world_state: WorldState,
    state: GeneralDecisionState,
) -> dict[str, Any]:
    store_general_state(world_state, state)
    return {
        "schema_version": "spice.workspace.state.v1",
        "world_state": payload_value(world_state),
    }


def _load_config(paths: SpiceWorkspacePaths) -> dict[str, Any]:
    payload = json.loads(paths.config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Workspace config must be a JSON object: {paths.config}")
    return payload


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


def _state_summary(state: GeneralDecisionState) -> dict[str, int | str]:
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
        "decision_checkpoint_count": len(state.decision_checkpoints),
        "outcome_count": len(state.outcomes),
    }


def _candidate_summary(
    candidates: list[Any],
    *,
    selected_candidate_id: str,
) -> dict[str, Any]:
    by_action_type: dict[str, int] = {}
    by_availability_status: dict[str, int] = {}
    for candidate in candidates:
        by_action_type[candidate.action_type] = by_action_type.get(candidate.action_type, 0) + 1
        status = candidate.availability_status
        by_availability_status[status] = by_availability_status.get(status, 0) + 1
    return {
        "total": len(candidates),
        "selected_candidate_id": selected_candidate_id,
        "selected_present": any(
            candidate.candidate_id == selected_candidate_id for candidate in candidates
        ),
        "by_action_type": by_action_type,
        "by_availability_status": by_availability_status,
    }


def _normalize_run_intent_mode(value: str) -> str:
    mode = (value or "auto").strip().lower()
    if mode not in {"auto", "advise", "act"}:
        raise ValueError("run_intent_mode must be one of: auto, advise, act.")
    return mode


def _candidate_selection_for_run_mode(
    candidates: list[Any],
    *,
    mode: str,
    explicit_options: list[str] | None = None,
) -> dict[str, Any]:
    explicit_choice_ids = _explicit_choice_candidate_ids(
        candidates,
        explicit_options=explicit_options,
    )
    selection_scope = [
        candidate
        for candidate in candidates
        if not explicit_choice_ids or candidate.candidate_id in explicit_choice_ids
    ]
    decision_ids = _decision_candidate_ids(selection_scope)

    if mode != "act":
        if explicit_choice_ids:
            return {
                "selection_candidate_ids": explicit_choice_ids,
                "selection_pool_kind": "explicit_choice",
                "selection_pool_reason": (
                    "Candidate is visible for comparison but excluded because the user provided "
                    "explicit options and Spice is selecting within that option set."
                ),
                "handoff_blocked": False,
                "handoff_blockers": [],
            }
        if decision_ids:
            return {
                "selection_candidate_ids": decision_ids,
                "selection_pool_kind": "decision_candidates",
                "selection_pool_reason": (
                    "Candidate is visible for comparison but excluded because decision candidates "
                    "are available; runtime actions are guardrails and fallback choices."
                ),
                "handoff_blocked": False,
                "handoff_blockers": [],
            }
        return {
            "selection_candidate_ids": None,
            "selection_pool_kind": "all_candidates",
            "selection_pool_reason": "",
            "handoff_blocked": False,
            "handoff_blockers": [],
        }

    executable = _approval_eligible_executable_candidates(selection_scope)
    executable_decision = [
        candidate for candidate in executable if _is_decision_candidate(candidate)
    ]
    if executable_decision:
        return {
            "selection_candidate_ids": {
                candidate.candidate_id for candidate in executable_decision
            },
            "selection_pool_kind": "executable_decision_candidates",
            "selection_pool_reason": (
                "Candidate is visible for comparison but excluded because /act found "
                "approval-eligible executable decision candidates."
            ),
            "handoff_blocked": False,
            "handoff_blockers": [],
        }
    if executable:
        return {
            "selection_candidate_ids": {candidate.candidate_id for candidate in executable},
            "selection_pool_kind": "executable_candidates",
            "selection_pool_reason": (
                "Candidate is visible for comparison but excluded from the /act executable "
                "selection pool."
            ),
            "handoff_blocked": False,
            "handoff_blockers": [],
        }

    if decision_ids:
        return {
            "selection_candidate_ids": decision_ids,
            "selection_pool_kind": "decision_candidates_handoff_blocked",
            "selection_pool_reason": (
                "Candidate is visible for comparison but excluded because decision candidates "
                "are available; runtime actions are guardrails and fallback choices."
            ),
            "handoff_blocked": True,
            "handoff_blockers": [
                "Decision candidates were available, but none were approval-eligible executable candidates.",
                "Spice will select among decision candidates instead of falling back to runtime meta-actions.",
            ],
        }

    if explicit_choice_ids:
        return {
            "selection_candidate_ids": explicit_choice_ids,
            "selection_pool_kind": "explicit_choice_handoff_blocked",
            "selection_pool_reason": (
                "Candidate is visible for comparison but excluded because the user provided "
                "explicit options and Spice is selecting within that option set."
            ),
            "handoff_blocked": True,
            "handoff_blockers": [
                "The user provided explicit options, but none were approval-eligible executable candidates.",
                "Spice will select among the explicit options instead of falling back to meta-actions.",
            ],
        }

    return {
        "selection_candidate_ids": None,
        "selection_pool_kind": "all_candidates_handoff_blocked",
        "selection_pool_reason": "",
        "handoff_blocked": True,
        "handoff_blockers": [
            "No approval-eligible executable candidate was available for /act.",
            "Spice will show the best non-executable decision instead of fabricating an approval.",
        ],
    }


def _candidates_for_run_mode(candidates: list[Any], *, mode: str) -> list[Any]:
    selection = _candidate_selection_for_run_mode(candidates, mode=mode)
    candidate_ids = selection["selection_candidate_ids"]
    if candidate_ids is None:
        return list(candidates)
    return [candidate for candidate in candidates if candidate.candidate_id in candidate_ids]


def _explicit_choice_candidate_ids(
    candidates: list[Any],
    *,
    explicit_options: list[str] | None,
) -> set[str]:
    if not explicit_options:
        return set()
    option_count = len(explicit_options)
    candidate_ids: set[str] = set()
    for candidate in candidates:
        metadata = getattr(candidate, "metadata", {}) or {}
        source = metadata.get("source")
        candidate_source = metadata.get("candidate_source")
        explicit_index = _metadata_int(metadata.get("explicit_option_index"))
        if source == "explicit_options" or candidate_source == "explicit_options":
            candidate_ids.add(candidate.candidate_id)
            continue
        if explicit_index is not None and 1 <= explicit_index <= option_count:
            candidate_ids.add(candidate.candidate_id)
    return candidate_ids


def _decision_candidate_ids(candidates: list[Any]) -> set[str]:
    return {
        candidate.candidate_id
        for candidate in candidates
        if _is_decision_candidate(candidate)
        and getattr(candidate, "availability_status", "") != "blocked"
    }


def _is_decision_candidate(candidate: Any) -> bool:
    metadata = getattr(candidate, "metadata", {}) or {}
    return (
        getattr(candidate, "candidate_kind", "") == "decision"
        or metadata.get("candidate_kind") == "decision"
        or metadata.get("candidate_source") in {"llm_generator", "explicit_options"}
        or metadata.get("source") == "explicit_options"
    )


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


def _approval_eligible_executable_candidates(candidates: list[Any]) -> list[Any]:
    return [
        candidate
        for candidate in candidates
        if _candidate_approval_eligible_from_affordance(candidate)
    ]


def _candidate_approval_eligible_from_affordance(candidate: Any) -> bool:
    metadata = getattr(candidate, "metadata", {}) or {}
    execution_affordance = dict(metadata.get("execution_affordance", {}) or {})
    if (
        str(execution_affordance.get("schema_version") or "") != "0.1"
        or str(execution_affordance.get("generated_by") or "")
        != "spice.runtime.execution_affordance"
    ):
        return False
    approval = dict(execution_affordance.get("approval", {}) or {})
    return bool(
        approval.get("eligible_for_approval")
        and approval.get("required")
        and execution_affordance.get("candidate_executable")
        and execution_affordance.get("executor_available")
        and execution_affordance.get("executable")
    )


def _runtime_warning_payloads(
    *,
    candidate_expansion: dict[str, Any],
    candidate_simulation: dict[str, Any],
) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    if _llm_stage_fell_back(candidate_expansion):
        warnings.append(
            {
                "source": "llm_candidate_expansion",
                "message": "LLM candidate expansion fell back to deterministic candidates.",
                "reason": _fallback_reason(candidate_expansion),
            }
        )
    if _llm_stage_fell_back(candidate_simulation):
        warnings.append(
            {
                "source": "llm_simulation",
                "message": "LLM simulation fell back to deterministic candidate metadata.",
                "reason": _fallback_reason(candidate_simulation),
            }
        )
    return warnings


def _llm_stage_fell_back(payload: dict[str, Any]) -> bool:
    return bool(payload.get("enabled")) and str(payload.get("status") or "") == "fallback"


def _fallback_reason(payload: dict[str, Any]) -> str:
    return str(payload.get("error") or "model output could not be used").strip()


def _attach_compare_warnings(
    compare_payload: dict[str, Any],
    warnings: list[dict[str, str]],
) -> dict[str, Any]:
    if not warnings:
        return compare_payload
    payload = dict(compare_payload)
    existing = list(payload.get("warnings") or [])
    payload["warnings"] = existing + warnings
    return normalize_compare_payload(payload)


def _raw_model_outputs(
    *,
    candidate_expansion: dict[str, Any],
    candidate_simulation: dict[str, Any],
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    expansion_raw = str(candidate_expansion.get("raw_output") or "")
    simulation_raw = str(candidate_simulation.get("raw_output") or "")
    if expansion_raw:
        outputs["llm_candidate_expansion"] = expansion_raw
    if simulation_raw:
        outputs["llm_simulation"] = simulation_raw
    return outputs


def _merge_candidates(candidates: list[Any], additions: list[Any]) -> list[Any]:
    merged = list(candidates)
    seen = {getattr(candidate, "candidate_id", "") for candidate in merged}
    for candidate in additions:
        candidate_id = getattr(candidate, "candidate_id", "")
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        merged.append(candidate)
    return merged


def _merge_explicit_option_fallback_candidates(
    candidates: list[Any],
    fallback_candidates: list[Any],
    *,
    explicit_options: list[str],
) -> list[Any]:
    existing_indices = _explicit_option_indices(candidates, option_count=len(explicit_options))
    additions = [
        candidate
        for candidate in fallback_candidates
        if _metadata_int((getattr(candidate, "metadata", {}) or {}).get("explicit_option_index"))
        not in existing_indices
    ]
    return _merge_candidates(candidates, additions)


def _explicit_option_indices(candidates: list[Any], *, option_count: int) -> set[int]:
    indices: set[int] = set()
    for candidate in candidates:
        metadata = getattr(candidate, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            continue
        explicit_index = _metadata_int(metadata.get("explicit_option_index"))
        if explicit_index is not None and 1 <= explicit_index <= option_count:
            indices.add(explicit_index)
    return indices


def _focus_candidates_for_observations(
    candidates: list[Any],
    *,
    observations: list[GenericObservation],
) -> list[Any]:
    current_refs = _current_observation_refs(observations)
    if not current_refs:
        return list(candidates)
    focused = [
        candidate
        for candidate in candidates
        if _is_current_llm_candidate(candidate)
        or set(getattr(candidate, "target_refs", []) or []).intersection(current_refs)
    ]
    return focused or list(candidates)


def _is_current_llm_candidate(candidate: Any) -> bool:
    metadata = getattr(candidate, "metadata", {}) or {}
    return isinstance(metadata, dict) and (
        metadata.get("source") == "llm_candidate_expander"
        or metadata.get("candidate_source") == "llm_generator"
    )


def _current_observation_refs(observations: list[GenericObservation]) -> set[str]:
    refs: set[str] = set()
    for observation in observations:
        refs.add(observation.observation_id)
        subject = observation.subject
        if subject.subject_id:
            refs.add(subject.subject_id)
        refs.update(subject.refs or [])
        refs.update(observation.refs or [])
        attrs = observation.attributes or {}
        for key, value in attrs.items():
            if key == "target_refs" and isinstance(value, list):
                refs.update(str(item) for item in value if item)
            elif key.endswith("_id") and value:
                refs.add(str(value))
    return refs


def _llm_runtime_status_line(label: str, payload: Any) -> str:
    if not isinstance(payload, dict):
        return f"- {label}: unavailable"
    if not payload.get("enabled"):
        return f"- {label}: disabled"
    status = str(payload.get("status") or "unknown")
    provider = str(payload.get("model_provider") or "")
    model = str(payload.get("model_id") or "")
    prefix = f"- {label}: {status}"
    if provider or model:
        prefix += f" ({provider}/{model})"
    error = str(payload.get("error") or "").strip()
    if error:
        prefix += f" - fallback reason: {error}"
    return prefix


def _is_handoff_candidate(candidate: Any) -> bool:
    if getattr(candidate, "availability_status", "") == "blocked":
        return True
    action_type = getattr(candidate, "action_type", "")
    if action_type in {
        "intent.execute",
        "capability.use",
        "item.triage",
        "artifact.draft",
        "task.split",
    }:
        return True
    side_effect = _candidate_side_effect(candidate)
    return side_effect in {"state_change", "external_effect"}


def _candidate_side_effect(candidate: Any) -> str:
    action_type = getattr(candidate, "action_type", "")
    boundary = getattr(candidate, "execution_boundary", None)
    boundary_value = getattr(boundary, "side_effect_class", "") if boundary else ""
    value = getattr(candidate, "side_effect_class", "") or boundary_value
    if action_type in {"intent.execute", "capability.use"}:
        return "external_effect"
    if action_type in {"artifact.draft", "item.triage", "task.split"}:
        return "state_change"
    if value in {"external", "execute", "write", "send", "external_effect"}:
        return "external_effect"
    if value in {"draft", "low", "state_change"}:
        return "state_change"
    return "read_only"


def _stable_id(prefix: str, text: str, now: datetime) -> str:
    return f"{prefix}.{now.strftime('%Y%m%dT%H%M%S.%fZ')}.{_hash(text)[:12]}"


def _make_run_id(
    text: str,
    now: datetime,
    *,
    store: LocalJsonStore,
    prefix: str = "run.manual",
) -> str:
    base = f"{prefix}.{now.strftime('%Y%m%dT%H%M%S.%fZ')}.{_hash(text)[:12]}"
    candidate = base
    counter = 2
    while store.record_path("run", candidate).exists():
        candidate = f"{base}.{counter}"
        counter += 1
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


def _selected_skill_id(skill_resolution: dict[str, Any]) -> str | None:
    resolved = skill_resolution.get("resolved_skill")
    if not isinstance(resolved, dict):
        return None
    return str(resolved.get("skill_id") or "") or None


def _selected_skill_executor_id(skill_resolution: dict[str, Any]) -> str | None:
    resolved = skill_resolution.get("resolved_skill")
    if not isinstance(resolved, dict):
        return None
    return str(resolved.get("executor_id") or "") or None


def _safe_slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_") or "unknown"


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]
