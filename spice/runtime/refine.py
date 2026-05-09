from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from spice.decision.compare import render_compare_text
from spice.decision.general import (
    GenericCandidate,
    GenericPolicyAdapter,
    load_general_state,
    store_general_state,
)
from spice.decision.general.candidates import (
    EstimatedCost,
    ExecutionBoundary,
    ExpectedStateDelta,
    GenericExecutionIntent,
    RiskProfile,
)
from spice.decision.general.types import payload_value
from spice.llm.candidate_expander import (
    LLMCandidateExpansionResult,
    expand_candidates_from_runtime_config,
    merge_expanded_candidates,
)
from spice.llm.simulation_runner import simulate_candidates_from_runtime_config
from spice.runtime.active_decision_frame import (
    attach_active_decision_frame,
    build_active_decision_frame,
)
from spice.runtime.execution_affordance import annotate_execution_affordances
from spice.runtime.full_loop_preview import build_runtime_full_loop_preview
from spice.runtime.memory_writeback import (
    skipped_general_decision_memory_writeback,
    write_general_decision_memory,
)
from spice.runtime.run_once import (
    _candidate_summary,
    _candidate_selection_for_run_mode,
    _hash,
    _attach_compare_warnings,
    _load_config,
    _normalize_run_intent_mode,
    _raw_model_outputs,
    _require_workspace,
    _runtime_warning_payloads,
    _selected_skill_executor_id,
    _selected_skill_id,
    _state_summary,
    _timestamp,
    _workspace_relative,
    _workspace_state_payload,
    _world_state_from_workspace_payload,
)
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
    load_workspace_memory_provider,
    workspace_paths,
)


@dataclass(slots=True)
class RefineResult:
    artifact: dict[str, Any]
    rendered_text: str
    run_path: Path
    decision_path: Path
    approval_path: Path | None
    session_path: Path
    state_path: Path


def refine_decision(
    refinement: str,
    *,
    project_root: str | Path = ".",
    session_id: str = DEFAULT_SESSION_ID,
    run_id: str | None = None,
    now: datetime | None = None,
    use_bars: bool = False,
    persist: bool = True,
    full_loop_preview: bool = True,
    run_intent_mode: str | None = None,
) -> RefineResult:
    text = refinement.strip()
    if not text:
        raise ValueError("refine requires non-empty feedback text.")

    created = now or datetime.now(timezone.utc)
    paths = workspace_paths(project_root)
    _require_workspace(paths)
    store = LocalJsonStore(paths)
    config = _load_config(paths)
    workspace_config = SpiceWorkspaceConfig.from_payload(config)
    session = load_or_create_session(store, session_id=session_id, now=created)
    parent_run_id = run_id or session.last_run_id
    if not parent_run_id:
        raise ValueError("No prior run found for refinement. Run `spice decide \"...\"` first.")
    parent = store.load_run(parent_run_id)
    parent_candidates = _load_parent_candidates(parent)
    parent_intent = _parent_intent(parent)
    mode = _normalize_run_intent_mode(run_intent_mode or str(parent.get("run_intent_mode") or "auto"))
    effective_full_loop_preview = full_loop_preview and mode != "advise"

    state_payload = store.load_state()
    state_before_hash = _hash(state_payload)[:12]
    world_state = _world_state_from_workspace_payload(state_payload)
    state = load_general_state(world_state)
    state_before_summary = _state_summary(state)
    display_language = str(parent.get("display_language") or "en")
    context_compiler = load_workspace_context_compiler(
        project_root,
        config=workspace_config,
    )
    refined_intent = f"{parent_intent}\n\nRefinement: {text}"
    decision_context = context_compiler.compile_general_decision_context(
        world_state,
        state,
        current_intent={
            "text": refined_intent,
            "source": "manual_refinement",
            "kind": "manual_refinement",
            "run_intent_mode": mode,
            "display_language": display_language,
            "parent_run_id": parent_run_id,
        },
        session=payload_value(session),
        config=config,
        domain="general",
    )

    expansion = _expand_refinement_candidates(
        config=config,
        state=state,
        parent_intent=parent_intent,
        refinement=text,
        parent_candidates=parent_candidates,
        decision_context=decision_context,
    )
    candidates = merge_expanded_candidates(parent_candidates, expansion)
    if not expansion.candidates:
        fallback = _fallback_refinement_candidate(
            parent=parent,
            parent_candidates=parent_candidates,
            refinement=text,
        )
        candidates = merge_expanded_candidates(
            candidates,
            LLMCandidateExpansionResult(
                enabled=False,
                status="manual_refinement_candidate",
                candidates=[fallback],
                proposed_count=1,
                accepted_count=1,
            ),
        )
    simulation_targets = select_simulation_targets(candidates)
    simulation_context = context_compiler.compile_general_simulation_context(
        world_state,
        state,
        current_intent=decision_context.current_intent,
        candidates=simulation_targets,
        active_decision_frame=decision_context.active_decision_frame,
        session=payload_value(session),
        config=config,
        domain="general",
    )
    candidate_simulation = simulate_candidates_from_runtime_config(
        config=config,
        state=state,
        intent_text=refined_intent,
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
    evaluation_candidates = list(candidates)
    candidate_selection = _candidate_selection_for_run_mode(candidates, mode=mode)
    selection_candidate_ids = candidate_selection["selection_candidate_ids"]
    if not evaluation_candidates:
        raise ValueError("No candidates remain after applying the refinement.")

    decision_id = _stable_refine_id("decision.refine", parent_run_id, text, created)
    trace_ref = _stable_refine_id("trace.refine", parent_run_id, text, created)
    policy = GenericPolicyAdapter.from_decision_profile(paths.decision_profile)
    policy_result = policy.evaluate(
        state,
        candidates=evaluation_candidates,
        decision_id=decision_id,
        trace_ref=trace_ref,
        run_intent_mode=mode,
        selection_candidate_ids=selection_candidate_ids,
        selection_pool_reason=candidate_selection["selection_pool_reason"],
    )
    checkpoint = policy_result.checkpoint
    state.decision_checkpoints.append(checkpoint)
    state.trace_refs.append(trace_ref)
    if checkpoint.approval is not None:
        state.approvals.append(checkpoint.approval)

    runtime_warnings = _runtime_warning_payloads(
        candidate_expansion=expansion.to_payload(),
        candidate_simulation=candidate_simulation.to_payload(),
    )
    compare_payload = _attach_compare_warnings(
        policy_result.compare_payload,
        runtime_warnings,
    )
    run_id_value = _make_refine_run_id(parent_run_id, text, created, store=store)
    approval_id = checkpoint.approval.approval_id if checkpoint.approval else None
    selected_skill_resolution = selected_skill_resolution_payload(
        evaluation_candidates,
        selected_candidate_id=checkpoint.selected_candidate_id,
    )
    handoff_blocked = bool(candidate_selection["handoff_blocked"])
    handoff_blockers = list(candidate_selection["handoff_blockers"])
    selection_pool = {
        "kind": candidate_selection["selection_pool_kind"],
        "candidate_ids": sorted(selection_candidate_ids or []),
        "reason": candidate_selection["selection_pool_reason"],
    }
    active_decision_frame = build_active_decision_frame(
        compare_payload=compare_payload,
        run_id=run_id_value,
        session_id=session.session_id,
        input_text=text,
        created_at=_timestamp(created),
        run_intent_mode=mode,
        display_language=display_language,
        approval_id=approval_id,
        selection_pool=selection_pool,
        handoff_blocked=handoff_blocked,
        handoff_blockers=handoff_blockers,
        source="manual_refinement",
        parent_run_id=parent_run_id,
    )
    attach_active_decision_frame(state, active_decision_frame)

    state_after_payload = _workspace_state_payload(world_state, state)
    state_after_hash = _hash(state_after_payload)[:12]
    persisted_at = _timestamp(created) if persist else None
    if persist:
        store.save_state(state_after_payload)

    decision_card = render_compare_text(compare_payload, use_bars=use_bars)
    run_path = store.record_path("run", run_id_value)
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

    artifact: dict[str, Any] = {
        "path_type": "manual_intent_refine",
        "generated_by": "spice.runtime.refine",
        "created_at": _timestamp(created),
        "run_id": run_id_value,
        "session_id": session.session_id,
        "loop_mode": "full_loop_preview" if effective_full_loop_preview else "decision_only",
        "run_intent_mode": mode,
        "source": "manual_refinement",
        "parent_run_id": parent_run_id,
        "parent_decision_id": parent.get("decision_id"),
        "parent_trace_ref": parent.get("trace_ref"),
        "decision_id": decision_id,
        "trace_ref": trace_ref,
        "selected_candidate_id": checkpoint.selected_candidate_id,
        "skill_resolution": selected_skill_resolution,
        "skill_resolution_status": selected_skill_resolution.get("status"),
        "skill_id": _selected_skill_id(selected_skill_resolution),
        "executor_id": _selected_skill_executor_id(selected_skill_resolution),
        "approval_id": approval_id,
        "handoff_blocked": handoff_blocked,
        "handoff_blockers": handoff_blockers,
        "selection_pool": selection_pool,
        "active_decision_frame": active_decision_frame,
        "state_before_ref": state_before_ref,
        "state_after_ref": state_after_ref,
        "store_paths": store_paths,
        "input": {
            "kind": "manual_refinement",
            "text": text,
            "parent_intent": parent_intent,
            "parent_run_id": parent_run_id,
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
        "state_before_summary": state_before_summary,
        "state_after_summary": _state_summary(state),
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
            candidate_expansion=expansion.to_payload(),
            candidate_simulation=candidate_simulation.to_payload(),
        ),
        "llm_candidate_expansion": expansion.to_payload(),
        "llm_simulation": candidate_simulation.to_payload(),
        "candidates": [candidate.to_payload() for candidate in candidates],
        "evaluated_candidates": [candidate.to_payload() for candidate in evaluation_candidates],
        "candidate_summary": _candidate_summary(
            candidates,
            selected_candidate_id=checkpoint.selected_candidate_id,
        ),
        "compare_payload": compare_payload,
        "decision": policy_result.to_payload(),
        "approval": checkpoint.approval.to_payload() if checkpoint.approval else None,
        "refinement": {
            "text": text,
            "parent_run_id": parent_run_id,
            "parent_selected_candidate_id": parent.get("selected_candidate_id"),
            "added_candidate_count": max(0, len(candidates) - len(parent_candidates)),
            "llm_status": expansion.status,
            "fallback_used": not bool(expansion.candidates),
        },
    }
    if effective_full_loop_preview:
        preview = build_runtime_full_loop_preview(
            state=state,
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
    artifact["rendered_text"] = render_refine_text(
        refinement=text,
        artifact=artifact,
        decision_card=decision_card,
    )

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
    saved_run_path = store.save_run(run_id_value, artifact)
    return RefineResult(
        artifact=artifact,
        rendered_text=str(artifact["rendered_text"]),
        run_path=saved_run_path,
        decision_path=saved_decision_path,
        approval_path=saved_approval_path,
        session_path=session_path,
        state_path=paths.state,
    )


def render_refine_text(
    *,
    refinement: str,
    artifact: dict[str, Any],
    decision_card: str,
) -> str:
    lines = [
        "SPICE REFINE",
        "previous decision -> refined candidate set -> updated decision card",
        "no executor called | no SDEP sent",
        "",
        f"refinement: {refinement}",
        f"parent_run_id: {artifact['parent_run_id']}",
        f"run_id: {artifact['run_id']}",
        f"decision_id: {artifact['decision_id']}",
        f"trace_ref: {artifact['trace_ref']}",
        f"selected_candidate_id: {artifact['selected_candidate_id']}",
        f"approval_id: {artifact.get('approval_id') or 'none'}",
        "",
        "REFINEMENT",
        f"- llm_status: {artifact['refinement']['llm_status']}",
        f"- fallback_used: {str(bool(artifact['refinement']['fallback_used'])).lower()}",
        f"- added_candidate_count: {artifact['refinement']['added_candidate_count']}",
        "",
        "BOUNDARY",
        f"- executor_called: {str(bool(artifact['executor_called'])).lower()}",
        f"- sdep_request_sent: {str(bool(artifact['sdep_request_sent'])).lower()}",
        f"- state_persisted: {str(bool(artifact['state_persisted'])).lower()}",
        f"- persist_mode: {artifact['persist_mode']}",
        "",
        "UPDATED DECISION CARD",
        decision_card,
    ]
    return "\n".join(lines)


def _expand_refinement_candidates(
    *,
    config: dict[str, Any],
    state: Any,
    parent_intent: str,
    refinement: str,
    parent_candidates: list[GenericCandidate],
    decision_context: Any | None = None,
) -> LLMCandidateExpansionResult:
    prompt = f"{parent_intent}\n\nUser refinement: {refinement}"
    return expand_candidates_from_runtime_config(
        config=config,
        state=state,
        intent_text=prompt,
        rule_candidates=parent_candidates,
        decision_context=decision_context,
    )


def _load_parent_candidates(parent: dict[str, Any]) -> list[GenericCandidate]:
    raw = parent.get("candidates")
    if not isinstance(raw, list) or not raw:
        raw = parent.get("evaluated_candidates")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Parent run does not contain candidates to refine.")
    return [GenericCandidate.from_payload(item) for item in raw if isinstance(item, dict)]


def _parent_intent(parent: dict[str, Any]) -> str:
    input_payload = parent.get("input")
    if isinstance(input_payload, dict):
        text = str(input_payload.get("text") or "")
        if text.strip():
            return text.strip()
    return str(parent.get("intent_text") or parent.get("decision_id") or "previous decision")


def _fallback_refinement_candidate(
    *,
    parent: dict[str, Any],
    parent_candidates: list[GenericCandidate],
    refinement: str,
) -> GenericCandidate:
    selected_id = str(parent.get("selected_candidate_id") or "")
    selected = _candidate_by_id(parent_candidates, selected_id) or parent_candidates[0]
    action_type = selected.action_type
    side_effect_class = selected.side_effect_class
    requires_confirmation = selected.requires_confirmation
    candidate_id = "candidate.refine." + _slug(
        f"{parent.get('run_id', '')}.{action_type}.{refinement}.{len(parent_candidates)}"
    )[:80]
    if action_type in {"intent.execute", "capability.use"}:
        side_effect_class = "external_effect"
        requires_confirmation = True
    elif action_type in {"artifact.draft", "item.triage", "task.split"}:
        side_effect_class = "state_change"
        requires_confirmation = True
    selected_execution_intent = getattr(selected, "execution_intent", GenericExecutionIntent())
    execution_requested = bool(selected_execution_intent.requested)
    execution_intent = GenericExecutionIntent(
        intent_class=(
            selected_execution_intent.intent_class
            if selected_execution_intent.intent_class in {"advisory", "execution_requested"}
            else ("execution_requested" if execution_requested else "advisory")
        ),
        requested=execution_requested,
        handoff_task=refinement if execution_requested else "",
        reason=(
            "Manual refinement inherits execution intent from the parent candidate."
            if execution_requested
            else "Manual refinement is advisory unless the parent candidate requested execution."
        ),
        required_permission_hint=selected_execution_intent.required_permission_hint,
        side_effect_class=side_effect_class if execution_requested else "none",
        metadata={"parent_candidate_id": selected.candidate_id},
    )
    return GenericCandidate(
        candidate_id=candidate_id,
        action_type=action_type,
        intent=refinement,
        candidate_kind=_candidate_kind_from_parent(selected),
        target_refs=list(selected.target_refs),
        required_capability=selected.required_capability,
        execution_intent=execution_intent,
        estimated_cost=EstimatedCost(time_minutes=15, attention="medium"),
        risk_profile=RiskProfile(
            level=selected.risk_profile.level or "unknown",
            risk_refs=list(selected.risk_profile.risk_refs),
            summary="Manual refinement candidate; risk inherits the previous selected option.",
            uncertainty="medium",
        ),
        reversibility=selected.reversibility,
        requires_confirmation=requires_confirmation,
        expected_state_delta=ExpectedStateDelta(
            updates_refs=list(selected.target_refs),
            summary=f"Consider user refinement: {refinement}",
        ),
        execution_boundary=ExecutionBoundary(
            mode=selected.execution_boundary.mode,
            target=selected.execution_boundary.target,
            protocol="",
            required_capability=selected.required_capability,
            requires_confirmation=requires_confirmation,
            side_effect_class=side_effect_class,
            metadata={"source": "manual_refinement"},
        ),
        constraints_triggered=[],
        why_available=["User refinement added this option to the candidate set."],
        why_blocked=[],
        side_effect_class=side_effect_class,
        availability_status="needs_confirmation" if requires_confirmation else "available",
        metadata={
            "source": "manual_refinement",
            "candidate_kind": _candidate_kind_from_parent(selected),
            "candidate_source": "manual_refinement",
            "execution_intent": execution_intent.to_payload(),
            "parent_candidate_id": selected.candidate_id,
            "refinement_text": refinement,
        },
    )


def _candidate_kind_from_parent(candidate: GenericCandidate) -> str:
    kind = str(getattr(candidate, "candidate_kind", "") or "").strip()
    if kind in {"decision", "runtime_action", "execution_handoff"}:
        return kind
    metadata = getattr(candidate, "metadata", {}) or {}
    kind = str(metadata.get("candidate_kind") or "").strip()
    if kind in {"decision", "runtime_action", "execution_handoff"}:
        return kind
    if metadata.get("candidate_source") in {"llm_generator", "explicit_options"}:
        return "decision"
    return "runtime_action"


def _candidate_by_id(
    candidates: list[GenericCandidate],
    candidate_id: str,
) -> GenericCandidate | None:
    for candidate in candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    return None


def _stable_refine_id(prefix: str, parent_run_id: str, text: str, now: datetime) -> str:
    return f"{prefix}.{now.strftime('%Y%m%dT%H%M%S.%fZ')}.{_digest(parent_run_id + text)[:12]}"


def _make_refine_run_id(
    parent_run_id: str,
    text: str,
    now: datetime,
    *,
    store: LocalJsonStore,
) -> str:
    base = f"run.refine.{now.strftime('%Y%m%dT%H%M%S.%fZ')}.{_digest(parent_run_id + text)[:12]}"
    candidate = base
    counter = 2
    while store.record_path("run", candidate).exists():
        candidate = f"{base}.{counter}"
        counter += 1
    return candidate


def _digest(value: Any) -> str:
    return sha256(json.dumps(payload_value(value), sort_keys=True).encode("utf-8")).hexdigest()


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower() or "candidate"
