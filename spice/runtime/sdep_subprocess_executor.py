from __future__ import annotations

import json
import shlex
import subprocess
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


SDEP_SUBPROCESS_EXECUTOR_BUILDER = "spice.runtime.sdep_subprocess_executor"


@dataclass(slots=True)
class SDEPSubprocessExecutionResult:
    artifact: dict[str, Any]
    rendered_text: str
    outcome_path: Path
    run_path: Path
    session_path: Path | None
    state_path: Path


def execute_sdep_subprocess_approval(
    approval_id: str,
    *,
    command: str | list[str],
    project_root: str | Path = ".",
    timeout_seconds: int = 120,
    now: datetime | None = None,
    executor_provider_id: str = "sdep_subprocess",
    real_executor_called: bool = False,
) -> SDEPSubprocessExecutionResult:
    created = now or datetime.now(timezone.utc)
    paths = workspace_paths(project_root)
    _require_workspace(paths)
    store = LocalJsonStore(paths)
    approval = approval_from_payload(store.load_approval(approval_id))
    if approval.status != APPROVAL_APPROVED or not approval.execution_allowed:
        raise ValueError(
            f"Approval {approval_id} must be approved before SDEP subprocess execution."
        )

    run_id, run_payload = _find_run_for_approval(store, approval_id)
    preview = _dict(run_payload.get("full_loop_preview"))
    sdep_request_payload = _dict(preview.get("sdep_request"))
    if not sdep_request_payload:
        raise ValueError(
            f"Run {run_id} does not contain a planned SDEP execute.request."
        )
    request = SDEPExecuteRequest.from_dict(sdep_request_payload)
    request_payload = request.to_dict()
    _validate_approval_matches_request(
        approval_payload=approval.to_payload(),
        run_payload=run_payload,
        request_payload=request_payload,
    )

    argv = _normalize_command(command)
    response_payload = _invoke_subprocess(
        argv,
        request_payload=request_payload,
        timeout_seconds=timeout_seconds,
    )
    response = SDEPExecuteResponse.from_dict(response_payload)
    response_payload = response.to_dict()
    _validate_response_matches_request(
        response_payload=response_payload,
        request_payload=request_payload,
        approval_payload=approval.to_payload(),
    )

    state_payload = store.load_state()
    state_before_hash = _hash(state_payload)[:12]
    world_state = _world_state_from_workspace_payload(state_payload)
    state_before = load_general_state(world_state)

    outcome_record = build_subprocess_outcome_record(
        response_payload=response_payload,
        request_payload=request_payload,
        command=argv,
        executor_provider_id=executor_provider_id,
        real_executor_called=real_executor_called,
    )
    outcome_observation = build_subprocess_outcome_observation(
        outcome=outcome_record,
        response_payload=response_payload,
        command=argv,
        now=created,
        executor_provider_id=executor_provider_id,
        real_executor_called=real_executor_called,
    )
    state_after = reduce_generic_observations(state_before, [outcome_observation])
    state_after_payload = _workspace_state_payload(world_state, state_after)
    state_after_hash = _hash(state_after_payload)[:12]
    store.save_state(state_after_payload)

    outcome_artifact = {
        "path_type": "runtime_sdep_subprocess_outcome",
        "generated_by": SDEP_SUBPROCESS_EXECUTOR_BUILDER,
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
        "executor_provider": executor_provider_id,
        "executor_command": _command_summary(argv),
        "transport_executor_called": True,
        "real_executor_called": real_executor_called,
        "sdep_request_sent": True,
        "executed": True,
        "protocol_executed": True,
        "state_updated": True,
        "persisted": True,
        "sdep_request": request_payload,
        "sdep_response": response_payload,
        "outcome_record": outcome_record.to_payload(),
        "outcome_observation": outcome_observation.to_payload(),
    }
    outcome_path = store.save_outcome(outcome_record.outcome_id, outcome_artifact)

    session_path = _update_session(
        store,
        run_payload,
        outcome_record,
        created=created,
        executor_provider_id=executor_provider_id,
    )
    store_paths = {
        "run": _workspace_relative(store.record_path("run", run_id)),
        "approval": _workspace_relative(store.record_path("approval", approval.approval_id)),
        "outcome": _workspace_relative(outcome_path),
        "state": _workspace_relative(paths.state),
    }
    if session_path is not None:
        store_paths["session"] = _workspace_relative(session_path)

    artifact = {
        "path_type": "runtime_sdep_subprocess_execution",
        "generated_by": SDEP_SUBPROCESS_EXECUTOR_BUILDER,
        "created_at": _timestamp(created),
        "run_id": run_id,
        "session_id": run_payload.get("session_id"),
        "approval_id": approval.approval_id,
        "decision_id": outcome_record.decision_id,
        "trace_ref": outcome_record.trace_ref,
        "selected_candidate_id": outcome_record.candidate_id,
        "candidate_id": outcome_record.candidate_id,
        "skill_id": _nested(request_payload, "execution", "metadata", "skill_id"),
        "executor_id": _nested(request_payload, "execution", "metadata", "executor_id"),
        "context_pack_id": _nested(request_payload, "execution", "metadata", "context_pack_id"),
        "execution_id": outcome_record.execution_ref,
        "request_id": outcome_record.metadata.get("request_id"),
        "outcome_id": outcome_record.outcome_id,
        "protocol_status": outcome_record.protocol_status,
        "task_status": outcome_record.task_status,
        "executor_provider": executor_provider_id,
        "executor_command": _command_summary(argv),
        "transport_executor_called": True,
        "executor_called": True,
        "real_executor_called": real_executor_called,
        "sdep_request_sent": True,
        "executed": True,
        "protocol_executed": True,
        "execution_status": "sdep_response_received",
        "execution": None,
        "outcome": outcome_artifact,
        "state_updated": True,
        "persisted": True,
        "state_before_ref": f"{_workspace_relative(paths.state)}#before:{state_before_hash}",
        "state_after_ref": f"{_workspace_relative(paths.state)}#after:{state_after_hash}",
        "store_paths": store_paths,
        "sdep_request": request_payload,
        "sdep_response": response_payload,
        "outcome_record": outcome_record.to_payload(),
        "outcome_observation": outcome_observation.to_payload(),
    }
    artifact["rendered_text"] = render_sdep_subprocess_execution_text(artifact)
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

    run_payload["sdep_subprocess_execution"] = artifact
    run_payload["executor_execution"] = artifact
    provider_execution_key = f"{_safe_slug(executor_provider_id)}_execution"
    run_payload[provider_execution_key] = artifact
    run_payload["executor_provider"] = executor_provider_id
    run_payload["executor_command"] = _command_summary(argv)
    run_payload["sdep_request_sent"] = True
    run_payload["executor_called"] = True
    run_payload["transport_executor_called"] = True
    run_payload["real_executor_called"] = real_executor_called
    run_payload["executed"] = True
    run_payload["protocol_executed"] = True
    run_payload["execution_status"] = "sdep_response_received"
    run_payload["protocol_status"] = outcome_record.protocol_status
    run_payload["task_status"] = outcome_record.task_status
    run_payload["outcome_id"] = outcome_record.outcome_id
    run_payload["state_after_ref"] = artifact["state_after_ref"]
    run_payload["store_paths"] = {
        **_dict(run_payload.get("store_paths")),
        **store_paths,
    }
    run_path = store.save_run(run_id, run_payload)

    return SDEPSubprocessExecutionResult(
        artifact=artifact,
        rendered_text=artifact["rendered_text"],
        outcome_path=outcome_path,
        run_path=run_path,
        session_path=session_path,
        state_path=paths.state,
    )


def build_subprocess_outcome_record(
    *,
    response_payload: dict[str, Any],
    request_payload: dict[str, Any],
    command: list[str],
    executor_provider_id: str = "sdep_subprocess",
    real_executor_called: bool = False,
) -> OutcomeRecord:
    response = SDEPExecuteResponse.from_dict(response_payload)
    request = SDEPExecuteRequest.from_dict(request_payload)
    traceability = dict(response.traceability)
    execution_id = str(traceability.get("execution_id") or response.outcome.execution_id)
    output = dict(response.outcome.output)
    return OutcomeRecord(
        outcome_id=f"outcome.{_safe_slug(executor_provider_id)}.{_hash([response.request_id, execution_id, response.status, response.outcome.status])[:16]}",
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
            "adapter": SDEP_SUBPROCESS_EXECUTOR_BUILDER,
            "executor_provider": executor_provider_id,
            "executor_command": _command_summary(command),
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
            "transport_executor_called": True,
            "real_executor_called": real_executor_called,
        },
    )


def build_subprocess_outcome_observation(
    *,
    outcome: OutcomeRecord,
    response_payload: dict[str, Any],
    command: list[str],
    now: datetime | None = None,
    executor_provider_id: str = "sdep_subprocess",
    real_executor_called: bool = False,
) -> GenericObservation:
    created = now or datetime.now(timezone.utc)
    response = SDEPExecuteResponse.from_dict(response_payload)
    approval_id = outcome.metadata.get("approval_id")
    metadata = {
        "adapter": SDEP_SUBPROCESS_EXECUTOR_BUILDER,
        "executor_provider": executor_provider_id,
        "executor_command": _command_summary(command),
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
        "transport_executor_called": True,
        "real_executor_called": real_executor_called,
    }
    return GenericObservation(
        observation_id=f"obs.{_safe_slug(executor_provider_id)}.outcome.{_hash(outcome.outcome_id)[:16]}",
        kind=ObservationKind.OUTCOME,
        source=ObservationSource(
            provider=f"{executor_provider_id}_executor",
            channel="local_subprocess",
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
                kind="sdep_execute_response_subprocess",
                summary="SDEP execute.response received from local subprocess.",
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


def render_sdep_subprocess_execution_text(artifact: dict[str, Any]) -> str:
    return "\n".join(
        [
            "SPICE SDEP SUBPROCESS EXECUTION",
            "approved decision -> SDEP request -> subprocess response -> outcome -> state update",
            "SDEP sent to local subprocess | real external executor not assumed",
            "",
            f"approval_id: {artifact.get('approval_id')}",
            f"decision_id: {artifact.get('decision_id')}",
            f"trace_ref: {artifact.get('trace_ref')}",
            f"candidate_id: {artifact.get('candidate_id')}",
            "",
            "EXECUTOR TRANSPORT",
            f"- executor_provider: {artifact.get('executor_provider')}",
            f"- command: {artifact.get('executor_command')}",
            f"- sdep_request_sent: {str(bool(artifact.get('sdep_request_sent'))).lower()}",
            f"- transport_executor_called: {str(bool(artifact.get('transport_executor_called'))).lower()}",
            f"- real_executor_called: {str(bool(artifact.get('real_executor_called'))).lower()}",
            "",
            "EXECUTION HANDOFF",
            f"- planned_executor: {artifact.get('executor_id')}",
            f"- skill: {artifact.get('skill_id')}",
            f"- context_pack_id: {artifact.get('context_pack_id')}",
            f"- execution_id: {artifact.get('execution_id')}",
            f"- request_id: {artifact.get('request_id')}",
            "",
            "OUTCOME",
            f"- outcome_id: {artifact.get('outcome_id')}",
            f"- protocol_status: {artifact.get('protocol_status')}",
            f"- task_status: {artifact.get('task_status')}",
            "",
            "STATE",
            "- state_updated: true",
            "- persisted: true",
            f"- state_after_ref: {artifact.get('state_after_ref')}",
        ]
    )


def _invoke_subprocess(
    command: list[str],
    *,
    request_payload: dict[str, Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")
    try:
        completed = subprocess.run(
            command,
            input=json.dumps(request_payload, ensure_ascii=True),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"SDEP subprocess executor timed out after {timeout_seconds} seconds."
        ) from exc
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(
            f"SDEP subprocess executor exited with code {completed.returncode}: {stderr}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("SDEP subprocess executor stdout was not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("SDEP subprocess executor stdout must be a JSON object.")
    return payload


def _validate_response_matches_request(
    *,
    response_payload: dict[str, Any],
    request_payload: dict[str, Any],
    approval_payload: dict[str, Any],
) -> None:
    request = SDEPExecuteRequest.from_dict(request_payload)
    response = SDEPExecuteResponse.from_dict(response_payload)
    request_trace = dict(request.traceability)
    response_trace = dict(response.traceability)
    expected = {
        "request_id": request.request_id,
        "execution_id": str(request_trace.get("execution_id") or ""),
        "spice_decision_id": str(request_trace.get("spice_decision_id") or ""),
        "trace_ref": str(request_trace.get("trace_ref") or ""),
        "candidate_id": str(request_trace.get("candidate_id") or ""),
        "approval_id": str(request_trace.get("approval_id") or ""),
    }
    actual = {
        "request_id": response.request_id,
        "execution_id": str(response_trace.get("execution_id") or response.outcome.execution_id or ""),
        "spice_decision_id": str(response_trace.get("spice_decision_id") or ""),
        "trace_ref": str(response_trace.get("trace_ref") or ""),
        "candidate_id": str(response_trace.get("candidate_id") or ""),
        "approval_id": str(response_trace.get("approval_id") or ""),
    }
    for key, expected_value in expected.items():
        if not expected_value:
            raise ValueError(f"SDEP request missing required attribution: {key}.")
        if actual.get(key) != expected_value:
            raise ValueError(f"SDEP response attribution mismatch for {key}.")
    if actual["approval_id"] != str(approval_payload.get("approval_id") or ""):
        raise ValueError("SDEP response attribution mismatch for approval_id.")


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


def _find_run_for_approval(store: LocalJsonStore, approval_id: str) -> tuple[str, dict[str, Any]]:
    matches: list[tuple[str, dict[str, Any]]] = []
    for run_id in store.list_record_ids("runs"):
        payload = store.load_run(run_id)
        if payload.get("approval_id") == approval_id:
            matches.append((run_id, payload))
    if not matches:
        raise ValueError(f"No run artifact found for approval {approval_id}.")
    return sorted(matches, key=lambda item: str(item[1].get("created_at") or item[0]))[-1]


def _update_session(
    store: LocalJsonStore,
    run_payload: dict[str, Any],
    outcome: OutcomeRecord,
    *,
    created: datetime,
    executor_provider_id: str,
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
    metadata["last_executor_provider"] = executor_provider_id
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


def _normalize_command(command: str | list[str]) -> list[str]:
    if isinstance(command, str):
        argv = shlex.split(command)
    else:
        argv = [str(item) for item in command]
    if not argv:
        raise ValueError("SDEP subprocess command must be non-empty.")
    return argv


def _command_summary(command: list[str]) -> str:
    return " ".join(shlex.quote(item) for item in command)


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


def _safe_slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_") or "unknown"
