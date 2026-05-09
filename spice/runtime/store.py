from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spice.runtime.workspace import SpiceWorkspacePaths, workspace_paths


@dataclass(slots=True)
class LocalJsonStore:
    paths: SpiceWorkspacePaths

    @classmethod
    def from_project_root(cls, project_root: str | Path = ".") -> "LocalJsonStore":
        return cls(paths=workspace_paths(project_root))

    def load_state(self) -> dict[str, Any]:
        return _read_json(self.paths.state)

    def save_state(self, payload: dict[str, Any]) -> Path:
        return _write_json(self.paths.state, payload)

    def save_session(self, session_id: str, payload: dict[str, Any]) -> Path:
        return self._save_record(self.paths.sessions_dir, session_id, payload)

    def load_session(self, session_id: str) -> dict[str, Any]:
        return self._load_record(self.paths.sessions_dir, session_id)

    def save_run(self, run_id: str, payload: dict[str, Any]) -> Path:
        return self._save_record(self.paths.runs_dir, run_id, payload)

    def load_run(self, run_id: str) -> dict[str, Any]:
        return self._load_record(self.paths.runs_dir, run_id)

    def save_decision(self, decision_id: str, payload: dict[str, Any]) -> Path:
        return self._save_record(self.paths.decisions_dir, decision_id, payload)

    def load_decision(self, decision_id: str) -> dict[str, Any]:
        return self._load_record(self.paths.decisions_dir, decision_id)

    def save_approval(self, approval_id: str, payload: dict[str, Any]) -> Path:
        return self._save_record(self.paths.approvals_dir, approval_id, payload)

    def load_approval(self, approval_id: str) -> dict[str, Any]:
        return self._load_record(self.paths.approvals_dir, approval_id)

    def save_outcome(self, outcome_id: str, payload: dict[str, Any]) -> Path:
        return self._save_record(self.paths.outcomes_dir, outcome_id, payload)

    def load_outcome(self, outcome_id: str) -> dict[str, Any]:
        return self._load_record(self.paths.outcomes_dir, outcome_id)

    def save_perception(self, perception_id: str, payload: dict[str, Any]) -> Path:
        return self._save_record(self.paths.perceptions_dir, perception_id, payload)

    def load_perception(self, perception_id: str) -> dict[str, Any]:
        return self._load_record(self.paths.perceptions_dir, perception_id)

    def list_record_ids(self, kind: str) -> list[str]:
        directory = self._directory_for_kind(kind)
        if not directory.exists():
            return []
        return sorted(path.stem for path in directory.glob("*.json") if path.is_file())

    def record_path(self, kind: str, record_id: str) -> Path:
        if not record_id:
            raise ValueError("record_id must be non-empty")
        return self._directory_for_kind(kind) / f"{_safe_filename(record_id)}.json"

    def delete_record(self, kind: str, record_id: str) -> Path:
        path = self.record_path(kind, record_id)
        if not path.exists():
            raise FileNotFoundError(f"Local JSON store file does not exist: {path}")
        path.unlink()
        return path

    def _save_record(self, directory: Path, record_id: str, payload: dict[str, Any]) -> Path:
        if not record_id:
            raise ValueError("record_id must be non-empty")
        return _write_json(directory / f"{_safe_filename(record_id)}.json", payload)

    def _load_record(self, directory: Path, record_id: str) -> dict[str, Any]:
        if not record_id:
            raise ValueError("record_id must be non-empty")
        return _read_json(directory / f"{_safe_filename(record_id)}.json")

    def _directory_for_kind(self, kind: str) -> Path:
        mapping = {
            "session": self.paths.sessions_dir,
            "sessions": self.paths.sessions_dir,
            "run": self.paths.runs_dir,
            "runs": self.paths.runs_dir,
            "decision": self.paths.decisions_dir,
            "decisions": self.paths.decisions_dir,
            "approval": self.paths.approvals_dir,
            "approvals": self.paths.approvals_dir,
            "outcome": self.paths.outcomes_dir,
            "outcomes": self.paths.outcomes_dir,
            "perception": self.paths.perceptions_dir,
            "perceptions": self.paths.perceptions_dir,
        }
        try:
            return mapping[kind]
        except KeyError as exc:
            raise ValueError(f"Unknown store record kind: {kind}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Local JSON store file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Local JSON store payload must be an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    if not isinstance(payload, dict):
        raise ValueError("Local JSON store payload must be a dict.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _safe_filename(record_id: str) -> str:
    allowed = []
    for char in record_id:
        if char.isalnum() or char in {"-", "_", "."}:
            allowed.append(char)
        else:
            allowed.append("_")
    safe = "".join(allowed).strip("._")
    if not safe:
        raise ValueError(f"record_id cannot be converted to a safe filename: {record_id!r}")
    return safe
