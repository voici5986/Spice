from __future__ import annotations

import json
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
    OutcomeRecord,
    load_general_state,
    reduce_generic_observations,
    store_general_state,
)
from spice.decision.general.types import payload_value
from spice.protocols import WorldState
from spice.protocols.sdep import SDEPExecuteRequest, SDEPExecuteResponse
from spice.runtime.approval_flow import APPROVAL_APPROVED, approval_from_payload
from spice.runtime.memory_writeback import write_general_reflection_memory
from spice.runtime.session import SessionRecord
from spice.runtime.store import LocalJsonStore
from spice.runtime.workspace import (
    SpiceWorkspacePaths,
    load_workspace_config,
    load_workspace_memory_provider,
    workspace_paths,
)


DRY_RUN_EXECUTOR_BUILDER = "spice.runtime.dry_run_executor"


@dataclass(slots=True)
class DryRunExecutionResult:
    artifact: dict[str, Any]
    rendered_text: str
    outcome_path: Path
    run_path: Path
    session_path: Path | None
    state_path: Path


def execute_dry_run_approval(
    approval_id: str,
    *,
    project_root: str | Path = ".",
    now: datetime | None = None,
) -> DryRunExecutionResult:
    """Apply an approved local approval through the dry-run executor bridge.

    This consumes the runtime full-loop preview SDEP request, returns a local
    execute.response-shaped payload, records an outcome, and updates local
    General state. It does not call a real executor or send SDEP over a transport.
    """

    created = now or datetime.now(timezone.utc)
    paths = workspace_paths(project_root)
    _require_workspace(paths)
    store = LocalJsonStore(paths)
    approval = approval_from_payload(store.load_approval(approval_id))
    if approval.status != APPROVAL_APPROVED or not approval.execution_allowed:
        raise ValueError(
            f"Approval {approval_id} must be approved before dry-run execution."
        )

    run_id, run_payload = _find_run_for_approval(store, approval_id)
    preview = _dict(run_payload.get("full_loop_preview"))
    sdep_request = _dict(preview.get("sdep_request"))
    if not sdep_request:
        raise ValueError(
            f"Run {run_id} does not contain a full-loop SDEP request preview."
        )
    request = SDEPExecuteRequest.from_dict(sdep_request)

    _validate_approval_matches_request(
        approval_payload=approval.to_payload(),
        run_payload=run_payload,
        request_payload=request.to_dict(),
    )

    state_payload = store.load_state()
    state_before_hash = _hash(state_payload)[:12]
    world_state = _world_state_from_workspace_payload(state_payload)
    state_before = load_general_state(world_state)

    sdep_response = build_dry_run_sdep_response(
        request.to_dict(),
        approval_payload=approval.to_payload(),
        now=created,
    )
    outcome_record = build_outcome_record_from_response(
        response_payload=sdep_response,
        request_payload=request.to_dict(),
    )
    outcome_observation = build_outcome_observation(
        outcome=outcome_record,
        response_payload=sdep_response,
        now=created,
    )
    state_after = reduce_generic_observations(state_before, [outcome_observation])
    state_after_payload = _workspace_state_payload(world_state, state_after)
    state_after_hash = _hash(state_after_payload)[:12]
    store.save_state(state_after_payload)

    outcome_artifact = {
        "path_type": "runtime_dry_run_outcome",
        "generated_by": DRY_RUN_EXECUTOR_BUILDER,
        "created_at": _timestamp(created),
        "decision_id": outcome_record.decision_id,
        "trace_ref": outcome_record.trace_ref,
        "candidate_id": outcome_record.candidate_id,
        "approval_id": approval.approval_id,
        "execution_id": outcome_record.execution_ref,
        "request_id": outcome_record.metadata.get("request_id"),
        "outcome_id": outcome_record.outcome_id,
        "protocol_status": outcome_record.protocol_status,
        "task_status": outcome_record.task_status,
        "dry_run": True,
        "real_executor_called": False,
        "sdep_request_sent": False,
        "executed": False,
        "state_updated": True,
        "persisted": True,
        "sdep_response": sdep_response,
        "outcome_record": outcome_record.to_payload(),
        "outcome_observation": outcome_observation.to_payload(),
    }
    outcome_path = store.save_outcome(outcome_record.outcome_id, outcome_artifact)

    session_path = _update_session(store, run_payload, outcome_record, created=created)
    store_paths = {
        "run": _workspace_relative(store.record_path("run", run_id)),
        "approval": _workspace_relative(store.record_path("approval", approval.approval_id)),
        "outcome": _workspace_relative(outcome_path),
        "state": _workspace_relative(paths.state),
    }
    if session_path is not None:
        store_paths["session"] = _workspace_relative(session_path)

    artifact = {
        "path_type": "runtime_dry_run_execution",
        "generated_by": DRY_RUN_EXECUTOR_BUILDER,
        "created_at": _timestamp(created),
        "run_id": run_id,
        "session_id": run_payload.get("session_id"),
        "approval_id": approval.approval_id,
        "decision_id": outcome_record.decision_id,
        "trace_ref": outcome_record.trace_ref,
        "selected_candidate_id": outcome_record.candidate_id,
        "candidate_id": outcome_record.candidate_id,
        "skill_id": _nested(request.to_dict(), "execution", "metadata", "skill_id"),
        "executor_id": _nested(request.to_dict(), "execution", "metadata", "executor_id"),
        "context_pack_id": _nested(
            request.to_dict(),
            "execution",
            "metadata",
            "context_pack_id",
        ),
        "execution_id": outcome_record.execution_ref,
        "request_id": outcome_record.metadata.get("request_id"),
        "outcome_id": outcome_record.outcome_id,
        "protocol_status": outcome_record.protocol_status,
        "task_status": outcome_record.task_status,
        "executor_provider": "dry_run",
        "dry_run": True,
        "dry_run_executor_called": True,
        "executor_called": False,
        "real_executor_called": False,
        "sdep_request_sent": False,
        "executed": False,
        "execution": None,
        "outcome": outcome_artifact,
        "state_updated": True,
        "persisted": True,
        "state_before_ref": f"{_workspace_relative(paths.state)}#before:{state_before_hash}",
        "state_after_ref": f"{_workspace_relative(paths.state)}#after:{state_after_hash}",
        "store_paths": store_paths,
        "sdep_request": request.to_dict(),
        "sdep_response": sdep_response,
        "outcome_record": outcome_record.to_payload(),
        "outcome_observation": outcome_observation.to_payload(),
    }
    artifact["rendered_text"] = render_dry_run_execution_text(artifact)
    workspace_config = load_workspace_config(project_root)
    memory_provider = load_workspace_memory_provider(
        project_root,
        config=workspace_config,
    )
    artifact["memory_writeback"] = write_general_reflection_memory(
        memory_provider,
        decision_artifact=run_payload,
        execution_artifact=artifact,
        config=workspace_config.to_payload(),
    )

    run_payload["dry_run_execution"] = artifact
    run_payload["outcome_id"] = outcome_record.outcome_id
    run_payload["state_after_ref"] = artifact["state_after_ref"]
    run_payload["state_updated_by_dry_run"] = True
    run_payload["store_paths"] = {
        **_dict(run_payload.get("store_paths")),
        **store_paths,
    }
    run_path = store.save_run(run_id, run_payload)

    return DryRunExecutionResult(
        artifact=artifact,
        rendered_text=artifact["rendered_text"],
        outcome_path=outcome_path,
        run_path=run_path,
        session_path=session_path,
        state_path=paths.state,
    )


def build_dry_run_sdep_response(
    sdep_request: dict[str, Any],
    *,
    approval_payload: dict[str, Any],
    now: datetime | None = None,
) -> dict[str, Any]:
    request = SDEPExecuteRequest.from_dict(sdep_request)
    created = now or datetime.now(timezone.utc)
    traceability = dict(request.traceability)
    execution = request.execution.to_dict()
    execution_id = str(traceability.get("execution_id") or "")
    decision_id = str(
        traceability.get("spice_decision_id")
        or approval_payload.get("decision_id")
        or ""
    )
    candidate_id = str(traceability.get("candidate_id") or approval_payload.get("candidate_id") or "")
    approval_id = str(traceability.get("approval_id") or approval_payload.get("approval_id") or "")
    context_pack = _dict(execution.get("input")).get("context_pack")
    context = _dict(context_pack)
    task = str(context.get("task") or execution.get("action_type") or "planned action")
    response = {
        "protocol": "sdep",
        "sdep_version": "0.1",
        "message_type": "execute.response",
        "message_id": f"sdep-msg.dry-run.{_hash([request.request_id, execution_id])[:16]}",
        "request_id": request.request_id,
        "timestamp": _timestamp(created),
        "responder": {
            "id": "spice.dry_run_executor",
            "name": "Spice Dry-run Executor",
            "version": "0.1",
            "vendor": "Spice",
            "implementation": "local-dry-run",
            "role": "executor",
        },
        "status": "success",
        "outcome": {
            "execution_id": execution_id,
            "status": "success",
            "outcome_type": "observation",
            "output": {
                "summary": f"Would execute: {task}",
                "dry_run": True,
                "state_delta": {
                    "updated_refs": [candidate_id] if candidate_id else [],
                    "task_status": "success",
                    "dry_run": True,
                },
                "expected_output": context.get("expected_output"),
                "context_pack_id": context.get("context_pack_id"),
            },
            "artifacts": [],
            "metrics": {},
            "metadata": {
                "adapter": DRY_RUN_EXECUTOR_BUILDER,
                "dry_run": True,
                "real_executor_called": False,
            },
        },
        "traceability": {
            "execution_id": execution_id,
            "spice_decision_id": decision_id,
            "trace_ref": traceability.get("trace_ref"),
            "candidate_id": candidate_id,
            "approval_id": approval_id,
            "skill_id": _nested(execution, "metadata", "skill_id"),
            "context_pack_id": _nested(execution, "metadata", "context_pack_id"),
        },
        "metadata": {
            "adapter": DRY_RUN_EXECUTOR_BUILDER,
            "dry_run": True,
            "real_executor_called": False,
            "sdep_request_sent": False,
        },
    }
    return SDEPExecuteResponse.from_dict(response).to_dict()


def build_outcome_record_from_response(
    *,
    response_payload: dict[str, Any],
    request_payload: dict[str, Any],
) -> OutcomeRecord:
    response = SDEPExecuteResponse.from_dict(response_payload)
    request = SDEPExecuteRequest.from_dict(request_payload)
    traceability = dict(response.traceability)
    execution_id = str(traceability.get("execution_id") or response.outcome.execution_id)
    output = dict(response.outcome.output)
    return OutcomeRecord(
        outcome_id=f"outcome.dry_run.{_hash([response.request_id, execution_id, response.status, response.outcome.status])[:16]}",
        decision_id=str(traceability.get("spice_decision_id") or ""),
        trace_ref=str(traceability.get("trace_ref") or ""),
        candidate_id=str(traceability.get("candidate_id") or ""),
        execution_ref=execution_id,
        protocol_status=response.status,
        task_status=response.outcome.status,
        status="observed",
        summary=str(output.get("summary") or f"Task status: {response.outcome.status}"),
        state_delta=dict(output.get("state_delta")) if isinstance(output.get("state_delta"), dict) else {},
        evidence_refs=[response.message_id],
        metadata={
            "adapter": DRY_RUN_EXECUTOR_BUILDER,
            "approval_id": traceability.get("approval_id"),
            "execution_id": execution_id,
            "request_id": response.request_id,
            "request_message_id": request.message_id,
            "response_message_id": response.message_id,
            "protocol_status": response.status,
            "task_status": response.outcome.status,
            "traceability": traceability,
            "responder": response.responder.to_dict(),
            "output": output,
            "dry_run": True,
            "real_executor_called": False,
        },
    )


def build_outcome_observation(
    *,
    outcome: OutcomeRecord,
    response_payload: dict[str, Any],
    now: datetime | None = None,
) -> GenericObservation:
    created = now or datetime.now(timezone.utc)
    response = SDEPExecuteResponse.from_dict(response_payload)
    approval_id = outcome.metadata.get("approval_id")
    metadata = {
        "adapter": DRY_RUN_EXECUTOR_BUILDER,
        "decision_id": outcome.decision_id,
        "trace_ref": outcome.trace_ref,
        "candidate_id": outcome.candidate_id,
        "approval_id": approval_id,
        "execution_id": outcome.execution_ref,
        "request_id": outcome.metadata.get("request_id"),
        "outcome_id": outcome.outcome_id,
        "protocol_status": outcome.protocol_status,
        "task_status": outcome.task_status,
        "response_message_id": response.message_id,
        "responder": response.responder.to_dict(),
        "output": dict(response.outcome.output),
        "dry_run": True,
        "real_executor_called": False,
    }
    return GenericObservation(
        observation_id=f"obs.dry_run.outcome.{_hash(outcome.outcome_id)[:16]}",
        kind=ObservationKind.OUTCOME,
        source=ObservationSource(
            provider="dry_run_executor",
            channel="local_runtime",
            received_at=_timestamp(created),
        ),
        subject=ObservationSubject(
            subject_id=outcome.execution_ref,
            subject_type="execution_outcome",
            title=outcome.summary,
            refs=[
                outcome.decision_id,
                outcome.trace_ref or "",
                outcome.candidate_id or "",
                outcome.execution_ref,
                outcome.outcome_id,
            ],
        ),
        summary=outcome.summary,
        attributes={
            "outcome_id": outcome.outcome_id,
            "decision_id": outcome.decision_id,
            "trace_ref": outcome.trace_ref,
            "candidate_id": outcome.candidate_id,
            "approval_id": approval_id,
            "execution_ref": outcome.execution_ref,
            "execution_id": outcome.execution_ref,
            "request_id": outcome.metadata.get("request_id"),
            "protocol_status": outcome.protocol_status,
            "task_status": outcome.task_status,
            "state_delta": dict(outcome.state_delta),
        },
        evidence=[
            ObservationEvidence(
                evidence_id=response.message_id,
                kind="sdep_execute_response_dry_run",
                summary="Local dry-run SDEP execute.response.",
                content=response.to_dict(),
            )
        ],
        confidence=ObservationConfidence(score=1.0, level="high"),
        refs=[
            outcome.decision_id,
            outcome.trace_ref or "",
            outcome.candidate_id or "",
            outcome.execution_ref,
            outcome.outcome_id,
        ],
        metadata=metadata,
    )


def render_dry_run_execution_text(artifact: dict[str, Any]) -> str:
    return "\n".join(
        [
            "SPICE DRY-RUN EXECUTION",
            "approved decision -> SDEP request -> local dry-run response -> outcome -> state update",
            "no real executor called | no SDEP sent outside this process",
            "",
            f"approval_id: {artifact.get('approval_id')}",
            f"decision_id: {artifact.get('decision_id')}",
            f"trace_ref: {artifact.get('trace_ref')}",
            f"candidate_id: {artifact.get('candidate_id')}",
            "",
            "EXECUTION HANDOFF",
            f"- planned_executor: {artifact.get('executor_id')}",
            f"- skill: {artifact.get('skill_id')}",
            f"- context_pack_id: {artifact.get('context_pack_id')}",
            f"- execution_id: {artifact.get('execution_id')}",
            f"- request_id: {artifact.get('request_id')}",
            "",
            "DRY-RUN RESULT",
            f"- outcome_id: {artifact.get('outcome_id')}",
            f"- protocol_status: {artifact.get('protocol_status')}",
            f"- task_status: {artifact.get('task_status')}",
            "- dry_run_executor_called: true",
            "- real_executor_called: false",
            "- sdep_request_sent: false",
            "- executed: false",
            "",
            "STATE",
            "- state_updated: true",
            "- persisted: true",
            f"- state_after_ref: {artifact.get('state_after_ref')}",
        ]
    )


def _find_run_for_approval(store: LocalJsonStore, approval_id: str) -> tuple[str, dict[str, Any]]:
    matches: list[tuple[str, dict[str, Any]]] = []
    for run_id in store.list_record_ids("runs"):
        payload = store.load_run(run_id)
        if payload.get("approval_id") == approval_id:
            matches.append((run_id, payload))
    if not matches:
        raise ValueError(f"No run artifact found for approval {approval_id}.")
    return sorted(matches, key=lambda item: str(item[1].get("created_at") or item[0]))[-1]


def _validate_approval_matches_request(
    *,
    approval_payload: dict[str, Any],
    run_payload: dict[str, Any],
    request_payload: dict[str, Any],
) -> None:
    traceability = _dict(request_payload.get("traceability"))
    approval_id = str(approval_payload.get("approval_id") or "")
    decision_id = str(approval_payload.get("decision_id") or "")
    candidate_id = str(approval_payload.get("candidate_id") or "")
    if approval_id != str(run_payload.get("approval_id") or ""):
        raise ValueError("Approved approval does not match the run artifact.")
    if decision_id != str(run_payload.get("decision_id") or ""):
        raise ValueError("Approved approval does not match the run decision_id.")
    if decision_id != str(traceability.get("spice_decision_id") or ""):
        raise ValueError("Approved approval does not match the SDEP request decision_id.")
    if candidate_id != str(traceability.get("candidate_id") or ""):
        raise ValueError("Approved approval does not match the SDEP request candidate_id.")
    if approval_id != str(traceability.get("approval_id") or ""):
        raise ValueError("Approved approval does not match the SDEP request approval_id.")


def _update_session(
    store: LocalJsonStore,
    run_payload: dict[str, Any],
    outcome: OutcomeRecord,
    *,
    created: datetime,
) -> Path | None:
    session_id = run_payload.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        return None
    try:
        payload = store.load_session(session_id)
    except FileNotFoundError:
        return None
    session = SessionRecord.from_payload(payload)
    metadata = dict(session.metadata)
    metadata["last_outcome_id"] = outcome.outcome_id
    metadata["last_execution_id"] = outcome.execution_ref
    metadata["last_task_status"] = outcome.task_status
    updated = SessionRecord(
        session_id=session.session_id,
        created_at=session.created_at,
        updated_at=_timestamp(created),
        status=session.status,
        run_ids=list(session.run_ids),
        decision_ids=list(session.decision_ids),
        approval_ids=list(session.approval_ids),
        active_state_ref=".spice/state/state.json",
        last_run_id=session.last_run_id,
        last_decision_id=session.last_decision_id,
        last_trace_ref=session.last_trace_ref,
        pending_approval_ids=list(session.pending_approval_ids),
        metadata=metadata,
    )
    return store.save_session(session_id, updated.to_payload())


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


def _require_workspace(paths: SpiceWorkspacePaths) -> None:
    missing = [path for path in (paths.config, paths.state) if not path.exists()]
    if missing:
        rendered = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Spice workspace is not initialized. Missing: {rendered}. Run `spice setup` first."
        )


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


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current
