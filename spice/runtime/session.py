from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from spice.decision.general.types import payload_value
from spice.runtime.store import LocalJsonStore

DEFAULT_SESSION_ID = "session.default"


@dataclass(slots=True)
class SessionRecord:
    session_id: str = DEFAULT_SESSION_ID
    created_at: str = ""
    updated_at: str = ""
    status: str = "active"
    run_ids: list[str] = field(default_factory=list)
    decision_ids: list[str] = field(default_factory=list)
    approval_ids: list[str] = field(default_factory=list)
    active_state_ref: str | None = None
    last_run_id: str | None = None
    last_decision_id: str | None = None
    last_trace_ref: str | None = None
    pending_approval_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SessionRecord":
        if not isinstance(payload, dict):
            raise ValueError("Session payload must be a dict.")
        return cls(
            session_id=str(payload.get("session_id") or DEFAULT_SESSION_ID),
            created_at=str(payload.get("created_at") or ""),
            updated_at=str(payload.get("updated_at") or ""),
            status=str(payload.get("status") or "active"),
            run_ids=_strings(payload.get("run_ids")),
            decision_ids=_strings(payload.get("decision_ids")),
            approval_ids=_strings(payload.get("approval_ids")),
            active_state_ref=_optional_string(payload.get("active_state_ref")),
            last_run_id=_optional_string(payload.get("last_run_id")),
            last_decision_id=_optional_string(payload.get("last_decision_id")),
            last_trace_ref=_optional_string(payload.get("last_trace_ref")),
            pending_approval_ids=_strings(payload.get("pending_approval_ids")),
            metadata=dict(payload.get("metadata")) if isinstance(payload.get("metadata"), dict) else {},
        )


@dataclass(slots=True)
class SessionTimelineEntry:
    timestamp: str
    session_id: str
    run_id: str
    decision_id: str | None = None
    intent: str | None = None
    selected: str | None = None
    approval_status: str | None = None
    execution_status: str | None = None
    outcome_id: str | None = None
    task_status: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(slots=True)
class SessionSearchMatch:
    session_id: str
    run_id: str
    decision_id: str | None
    timestamp: str
    summary: str

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


def load_or_create_session(
    store: LocalJsonStore,
    *,
    session_id: str = DEFAULT_SESSION_ID,
    now: datetime | None = None,
) -> SessionRecord:
    try:
        return SessionRecord.from_payload(store.load_session(session_id))
    except FileNotFoundError:
        created_at = _timestamp(now or datetime.now(timezone.utc))
        return SessionRecord(
            session_id=session_id,
            created_at=created_at,
            updated_at=created_at,
            metadata={
                "created_by": "spice runtime",
                "role": "decision loop session",
            },
        )


def append_run_to_session(
    store: LocalJsonStore,
    session: SessionRecord,
    run_artifact: dict[str, Any],
    *,
    now: datetime | None = None,
) -> SessionRecord:
    if not isinstance(run_artifact, dict):
        raise ValueError("run_artifact must be a dict.")
    run_id = _required_string(run_artifact, "run_id")
    decision_id = _required_string(run_artifact, "decision_id")
    trace_ref = _optional_string(run_artifact.get("trace_ref"))
    approval_id = _optional_string(run_artifact.get("approval_id"))
    state_ref = _optional_string(run_artifact.get("state_after_ref"))
    updated_at = _timestamp(now or datetime.now(timezone.utc))

    updated = SessionRecord(
        session_id=session.session_id,
        created_at=session.created_at or updated_at,
        updated_at=updated_at,
        status=session.status or "active",
        run_ids=_append_unique(session.run_ids, run_id),
        decision_ids=_append_unique(session.decision_ids, decision_id),
        approval_ids=_append_unique(session.approval_ids, approval_id) if approval_id else list(session.approval_ids),
        active_state_ref=state_ref or session.active_state_ref,
        last_run_id=run_id,
        last_decision_id=decision_id,
        last_trace_ref=trace_ref,
        pending_approval_ids=_pending_approval_ids(session.pending_approval_ids, run_artifact),
        metadata=dict(session.metadata),
    )
    store.save_session(updated.session_id, updated.to_payload())
    return updated


def list_sessions(
    store: LocalJsonStore,
    *,
    include_archived: bool = False,
) -> list[SessionRecord]:
    sessions: list[SessionRecord] = []
    for session_id in store.list_record_ids("sessions"):
        session = SessionRecord.from_payload(store.load_session(session_id))
        if include_archived or session.status != "archived":
            sessions.append(session)
    return sessions


def archive_session(
    store: LocalJsonStore,
    session_id: str,
    *,
    now: datetime | None = None,
) -> SessionRecord:
    session = SessionRecord.from_payload(store.load_session(session_id))
    updated = SessionRecord(
        session_id=session.session_id,
        created_at=session.created_at,
        updated_at=_timestamp(now or datetime.now(timezone.utc)),
        status="archived",
        run_ids=list(session.run_ids),
        decision_ids=list(session.decision_ids),
        approval_ids=list(session.approval_ids),
        active_state_ref=session.active_state_ref,
        last_run_id=session.last_run_id,
        last_decision_id=session.last_decision_id,
        last_trace_ref=session.last_trace_ref,
        pending_approval_ids=list(session.pending_approval_ids),
        metadata=dict(session.metadata),
    )
    store.save_session(updated.session_id, updated.to_payload())
    return updated


def delete_session(
    store: LocalJsonStore,
    session_id: str,
    *,
    cascade: bool = False,
) -> dict[str, Any]:
    session = SessionRecord.from_payload(store.load_session(session_id))
    deleted: dict[str, list[str]] = {
        "sessions": [],
        "runs": [],
        "decisions": [],
        "approvals": [],
        "outcomes": [],
    }
    if cascade:
        outcome_ids: list[str] = []
        for run_id in session.run_ids:
            try:
                run_payload = store.load_run(run_id)
            except FileNotFoundError:
                continue
            outcome_ids.extend(_run_outcome_ids(run_payload))
        for kind, ids in (
            ("runs", session.run_ids),
            ("decisions", session.decision_ids),
            ("approvals", session.approval_ids),
            ("outcomes", _unique(outcome_ids)),
        ):
            for record_id in ids:
                try:
                    store.delete_record(kind, record_id)
                except FileNotFoundError:
                    continue
                deleted[kind].append(record_id)
    store.delete_record("sessions", session.session_id)
    deleted["sessions"].append(session.session_id)
    return {
        "session_id": session.session_id,
        "cascade": cascade,
        "deleted": deleted,
    }


def build_session_timeline(
    store: LocalJsonStore,
    session: SessionRecord,
) -> list[SessionTimelineEntry]:
    entries: list[SessionTimelineEntry] = []
    for run_id in session.run_ids:
        try:
            run_payload = store.load_run(run_id)
        except FileNotFoundError:
            entries.append(
                SessionTimelineEntry(
                    timestamp="",
                    session_id=session.session_id,
                    run_id=run_id,
                    execution_status="missing_run_artifact",
                )
            )
            continue
        entries.append(_timeline_entry_from_run(session.session_id, run_payload))
    return entries


def search_sessions(
    store: LocalJsonStore,
    keyword: str,
    *,
    include_archived: bool = False,
) -> list[SessionSearchMatch]:
    needle = keyword.strip().lower()
    if not needle:
        raise ValueError("keyword must be non-empty.")
    matches: list[SessionSearchMatch] = []
    for session in list_sessions(store, include_archived=include_archived):
        for run_id in session.run_ids:
            try:
                run_payload = store.load_run(run_id)
            except FileNotFoundError:
                continue
            haystack = _searchable_run_text(run_payload)
            if needle not in haystack.lower():
                continue
            entry = _timeline_entry_from_run(session.session_id, run_payload)
            matches.append(
                SessionSearchMatch(
                    session_id=session.session_id,
                    run_id=entry.run_id,
                    decision_id=entry.decision_id,
                    timestamp=entry.timestamp,
                    summary=_entry_summary(entry),
                )
            )
    return matches


def session_stats(store: LocalJsonStore) -> dict[str, Any]:
    sessions = list_sessions(store, include_archived=True)
    approval_counts: dict[str, int] = {}
    for approval_id in store.list_record_ids("approvals"):
        try:
            approval = store.load_approval(approval_id)
        except FileNotFoundError:
            continue
        status = str(approval.get("status") or "unknown")
        approval_counts[status] = approval_counts.get(status, 0) + 1
    total_decisions = sum(len(session.decision_ids) for session in sessions)
    most_active = max(sessions, key=lambda item: len(item.run_ids), default=None)
    return {
        "total_sessions": len(sessions),
        "active_sessions": sum(1 for session in sessions if session.status != "archived"),
        "archived_sessions": sum(1 for session in sessions if session.status == "archived"),
        "total_runs": sum(len(session.run_ids) for session in sessions),
        "total_decisions": total_decisions,
        "approval_counts": approval_counts,
        "approve_reject_ratio": _approve_reject_ratio(approval_counts),
        "most_active_session": most_active.session_id if most_active else None,
        "most_active_session_runs": len(most_active.run_ids) if most_active else 0,
        "average_decision_time_seconds": None,
    }


def render_session_list(
    sessions: list[SessionRecord],
    *,
    include_archived: bool = False,
) -> str:
    lines = ["SPICE SESSIONS"]
    if not include_archived:
        lines.append("archived sessions hidden; use --all to include them")
    if not sessions:
        lines.append("- no sessions found")
        return "\n".join(lines)
    for session in sessions:
        lines.append(
            "- "
            f"{session.session_id} "
            f"status={session.status} "
            f"runs={len(session.run_ids)} "
            f"pending_approvals={len(session.pending_approval_ids)} "
            f"last_run={session.last_run_id or 'none'}"
        )
    return "\n".join(lines)


def render_session_current(
    session_id: str,
    session: SessionRecord | None,
) -> str:
    lines = [
        "SPICE CURRENT SESSION",
        f"active_session_id: {session_id}",
    ]
    if session is None:
        lines.append("status: missing")
        lines.append("hint: run `spice run --session-id <id>` to create it, or `spice session switch <id>` to switch.")
        return "\n".join(lines)
    lines.extend(
        [
            f"status: {session.status}",
            f"runs: {len(session.run_ids)}",
            f"pending approvals: {len(session.pending_approval_ids)}",
            f"last_run: {session.last_run_id or 'none'}",
        ]
    )
    return "\n".join(lines)


def render_session_resume(session: SessionRecord) -> str:
    lines = [
        "SPICE SESSION RESUME",
        f"session_id: {session.session_id}",
        f"status: {session.status}",
        f"runs: {len(session.run_ids)}",
        f"decisions: {len(session.decision_ids)}",
        f"pending approvals: {len(session.pending_approval_ids)}",
        f"active_state_ref: {session.active_state_ref or 'none'}",
        "",
        "LAST DECISION",
        f"- run_id: {session.last_run_id or 'none'}",
        f"- decision_id: {session.last_decision_id or 'none'}",
        f"- trace_ref: {session.last_trace_ref or 'none'}",
    ]
    if session.pending_approval_ids:
        lines.extend(["", "PENDING APPROVALS"])
        lines.extend(f"- {approval_id}" for approval_id in session.pending_approval_ids[:5])
    lines.extend(
        [
            "",
            "RESUME COMMANDS",
            f"- spice run --session-id {session.session_id}",
            f"- spice run --session-id {session.session_id} --act",
            f"- spice run --once \"...\" --session-id {session.session_id}",
        ]
    )
    return "\n".join(lines)


def render_session_timeline(entries: list[SessionTimelineEntry]) -> str:
    lines = ["SPICE SESSION TIMELINE"]
    if not entries:
        lines.append("- no runs found")
        return "\n".join(lines)
    for entry in entries:
        lines.append(
            f"- {entry.timestamp or 'unknown-time'}  "
            f"decision: {entry.intent or entry.decision_id or entry.run_id}  "
            f"-> {entry.approval_status or 'no_approval'}"
            f" -> {entry.execution_status or 'not_executed'}"
            f"{' -> ' + entry.task_status if entry.task_status else ''}"
        )
    return "\n".join(lines)


def render_session_search(matches: list[SessionSearchMatch], keyword: str) -> str:
    lines = ["SPICE SESSION SEARCH", f"keyword: {keyword}"]
    if not matches:
        lines.append("- no matches")
        return "\n".join(lines)
    for match in matches:
        lines.append(
            f"- {match.timestamp or 'unknown-time'}  "
            f"{match.session_id}  {match.run_id}  {match.summary}"
        )
    return "\n".join(lines)


def render_session_stats(stats: dict[str, Any]) -> str:
    approvals = stats.get("approval_counts")
    if not isinstance(approvals, dict):
        approvals = {}
    lines = [
        "SPICE SESSION STATS",
        f"total_sessions: {stats.get('total_sessions', 0)}",
        f"active_sessions: {stats.get('active_sessions', 0)}",
        f"archived_sessions: {stats.get('archived_sessions', 0)}",
        f"total_runs: {stats.get('total_runs', 0)}",
        f"total_decisions: {stats.get('total_decisions', 0)}",
        f"most_active_session: {stats.get('most_active_session') or 'none'}",
        f"approve_reject_ratio: {stats.get('approve_reject_ratio')}",
        "approval_counts:",
    ]
    if approvals:
        lines.extend(f"- {key}: {value}" for key, value in sorted(approvals.items()))
    else:
        lines.append("- none")
    lines.append("average_decision_time_seconds: unavailable")
    return "\n".join(lines)


def render_session_delete_result(result: dict[str, Any]) -> str:
    deleted = result.get("deleted")
    if not isinstance(deleted, dict):
        deleted = {}
    lines = [
        "SPICE SESSION DELETED",
        f"session_id: {result.get('session_id')}",
        f"cascade: {str(bool(result.get('cascade'))).lower()}",
    ]
    for kind in ("sessions", "runs", "decisions", "approvals", "outcomes"):
        values = deleted.get(kind)
        count = len(values) if isinstance(values, list) else 0
        lines.append(f"- {kind}: {count}")
    return "\n".join(lines)


def _pending_approval_ids(
    existing: list[str],
    run_artifact: dict[str, Any],
) -> list[str]:
    approval_id = _optional_string(run_artifact.get("approval_id"))
    if not approval_id:
        return list(existing)
    approval = run_artifact.get("approval")
    if isinstance(approval, dict) and str(approval.get("status") or "") == "pending":
        return _append_unique(existing, approval_id)
    return [item for item in existing if item != approval_id]


def _timeline_entry_from_run(
    session_id: str,
    run_payload: dict[str, Any],
) -> SessionTimelineEntry:
    approval = run_payload.get("approval")
    if not isinstance(approval, dict):
        approval = {}
    return SessionTimelineEntry(
        timestamp=str(run_payload.get("created_at") or ""),
        session_id=session_id,
        run_id=str(run_payload.get("run_id") or ""),
        decision_id=_optional_string(run_payload.get("decision_id")),
        intent=_run_intent(run_payload),
        selected=_selected_title(run_payload),
        approval_status=_optional_string(approval.get("status")),
        execution_status=_execution_status(run_payload),
        outcome_id=_first(_run_outcome_ids(run_payload)),
        task_status=_task_status(run_payload),
    )


def _run_intent(run_payload: dict[str, Any]) -> str | None:
    input_payload = run_payload.get("input")
    if isinstance(input_payload, dict):
        text = _optional_string(input_payload.get("text"))
        if text:
            return text
    observations = run_payload.get("observations")
    if isinstance(observations, list):
        for observation in observations:
            if not isinstance(observation, dict):
                continue
            attributes = observation.get("attributes")
            if isinstance(attributes, dict):
                text = _optional_string(attributes.get("text"))
                if text:
                    return text
    return None


def _selected_title(run_payload: dict[str, Any]) -> str | None:
    selected = run_payload.get("selected")
    if isinstance(selected, dict):
        title = _optional_string(selected.get("title"))
        if title:
            return title
    compare = run_payload.get("compare_payload")
    if isinstance(compare, dict):
        selected_payload = compare.get("selected_recommendation")
        if isinstance(selected_payload, dict):
            return _optional_string(selected_payload.get("title"))
    return None


def _execution_status(run_payload: dict[str, Any]) -> str | None:
    for key in ("sdep_subprocess_execution", "dry_run_execution"):
        payload = run_payload.get(key)
        if isinstance(payload, dict):
            status = _optional_string(payload.get("execution_status"))
            if status:
                return status
    return _optional_string(run_payload.get("execution_status"))


def _task_status(run_payload: dict[str, Any]) -> str | None:
    for key in ("sdep_subprocess_execution", "dry_run_execution"):
        payload = run_payload.get(key)
        if isinstance(payload, dict):
            status = _optional_string(payload.get("task_status"))
            if status:
                return status
    return _optional_string(run_payload.get("task_status"))


def _run_outcome_ids(run_payload: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for key in ("outcome_id",):
        value = _optional_string(run_payload.get(key))
        if value:
            ids.append(value)
    for key in ("sdep_subprocess_execution", "dry_run_execution"):
        payload = run_payload.get(key)
        if isinstance(payload, dict):
            value = _optional_string(payload.get("outcome_id"))
            if value:
                ids.append(value)
    return _unique(ids)


def _searchable_run_text(run_payload: dict[str, Any]) -> str:
    fields = [
        _run_intent(run_payload),
        _selected_title(run_payload),
        _optional_string(run_payload.get("rendered_text")),
        _optional_string(run_payload.get("decision_id")),
        _optional_string(run_payload.get("trace_ref")),
        _optional_string(run_payload.get("run_id")),
    ]
    return "\n".join(field for field in fields if field)


def _entry_summary(entry: SessionTimelineEntry) -> str:
    parts = [
        entry.intent or entry.selected or entry.decision_id or entry.run_id,
        entry.approval_status,
        entry.execution_status,
        entry.task_status,
    ]
    return " -> ".join(part for part in parts if part)


def _approve_reject_ratio(approval_counts: dict[str, int]) -> float | None:
    approved = approval_counts.get("approved", 0)
    rejected = approval_counts.get("rejected", 0)
    total = approved + rejected
    if total == 0:
        return None
    return round(approved / total, 4)


def _first(values: list[str]) -> str | None:
    return values[0] if values else None


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _append_unique(values: list[str], value: str | None) -> list[str]:
    result = list(values)
    if value and value not in result:
        result.append(value)
    return result


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"run artifact missing required session field: {key}")
    return value


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()
