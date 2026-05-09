from __future__ import annotations

from pathlib import Path
from typing import Any

from spice.runtime.executor_discovery import detect_known_executor_clis
from spice.runtime.executor_runtime import (
    executor_runtime_specs,
    resolve_executor_runtime_from_config,
)
from spice.runtime.workspace import load_workspace_config


def build_executor_status(project_root: str | Path = ".") -> dict[str, Any]:
    config = load_workspace_config(project_root)
    configured = resolve_executor_runtime_from_config(config)
    detections = detect_known_executor_clis()
    executors: list[dict[str, Any]] = []
    for executor_id, spec in sorted(executor_runtime_specs().items()):
        detection = detections.get(executor_id)
        payload: dict[str, Any] = {
            "executor_id": executor_id,
            "configured": executor_id == configured.executor_id,
            "transport": spec.transport,
            "command_required": spec.command_required,
            "default_command": spec.default_command,
            "permission_enforcement": spec.permission_enforcement,
            "permission_commands": dict(spec.permission_commands),
            "real_executor": spec.real_executor,
            "sends_sdep_request": spec.sends_sdep_request,
            "status": "ready" if executor_id == "dry_run" else "not_configured",
            "detail": spec.metadata.get("description", ""),
        }
        if executor_id == "sdep_subprocess":
            payload["status"] = "configured" if configured.executor_id == executor_id else "needs_command"
        if detection is not None:
            payload["cli"] = detection.to_payload()
            payload["status"] = detection.status
            payload["detail"] = detection.detail
        if executor_id == configured.executor_id:
            payload["resolved_runtime"] = configured.to_payload()
            payload["status"] = configured.status
            payload["detail"] = configured.detail
        executors.append(payload)
    return {
        "configured_executor": configured.executor_id,
        "configured_runtime": configured.to_payload(),
        "executors": executors,
    }


def render_executor_list(status: dict[str, Any]) -> str:
    runtime = status.get("configured_runtime")
    lines = [
        "Spice Executors",
        f"configured: {status.get('configured_executor')}",
        f"permission: {runtime.get('permission_mode') if isinstance(runtime, dict) else '<unknown>'}",
        "",
    ]
    for item in status.get("executors", []):
        if not isinstance(item, dict):
            continue
        marker = "*" if item.get("configured") else "-"
        lines.append(
            f"{marker} {item.get('executor_id')} "
            f"[{item.get('status')}] "
            f"transport={item.get('transport')}"
        )
        detail = str(item.get("detail") or "").strip()
        if detail:
            lines.append(f"    {detail}")
    return "\n".join(lines)


def render_executor_doctor(status: dict[str, Any]) -> str:
    runtime = status.get("configured_runtime")
    lines = [
        "Spice Executor Doctor",
        f"configured: {status.get('configured_executor')}",
        "",
    ]
    if isinstance(runtime, dict):
        lines.extend(
            [
                "Resolved Runtime",
                f"- executor_id: {runtime.get('executor_id')}",
                f"- status: {runtime.get('status')}",
                f"- transport: {runtime.get('transport')}",
                f"- permission_mode: {runtime.get('permission_mode')}",
                f"- permission_enforcement: {runtime.get('permission_enforcement')}",
                f"- command: {runtime.get('command') or '<none>'}",
                f"- command_found: {str(bool(runtime.get('command_found'))).lower()}",
                f"- real_executor: {str(bool(runtime.get('real_executor'))).lower()}",
                f"- sends_sdep_request: {str(bool(runtime.get('sends_sdep_request'))).lower()}",
                f"- detail: {runtime.get('detail')}",
                "",
            ]
        )
    lines.append("CLI Discovery")
    for item in status.get("executors", []):
        if not isinstance(item, dict):
            continue
        cli = item.get("cli")
        if isinstance(cli, dict):
            detail = cli.get("detail") or cli.get("status")
            lines.append(f"- {item.get('executor_id')}: {cli.get('status')} - {detail}")
        else:
            lines.append(f"- {item.get('executor_id')}: {item.get('status')} - {item.get('detail')}")
    return "\n".join(lines)
