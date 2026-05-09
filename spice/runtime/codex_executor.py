from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

from spice.protocols.sdep import SDEPExecuteRequest, SDEPExecuteResponse


CODEX_EXECUTOR_ID = "codex"
CODEX_EXECUTOR_NAME = "Codex CLI Executor"
CODEX_EXECUTOR_IMPLEMENTATION = "spice.runtime.codex_executor"
DEFAULT_CODEX_COMMAND = "codex exec --skip-git-repo-check --sandbox workspace-write -"


@dataclass(frozen=True, slots=True)
class CodexCommandResult:
    status: str
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m spice.runtime.codex_executor",
        description="SDEP-compatible Codex subprocess executor.",
    )
    parser.add_argument(
        "--command",
        default=DEFAULT_CODEX_COMMAND,
        help="Codex command to run. The SDEP-derived task prompt is passed on stdin.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Timeout in seconds for the Codex command.",
    )
    args = parser.parse_args(argv)
    try:
        response = execute_codex_sdep_request(
            sys.stdin.read(),
            command=args.command,
            timeout_seconds=float(args.timeout),
        )
        sys.stdout.write(json.dumps(response, ensure_ascii=True, sort_keys=True))
        return 0
    except Exception as exc:
        sys.stderr.write(f"codex executor failed: {exc}\n")
        return 1


def execute_codex_sdep_request(
    request_json: str | dict[str, Any],
    *,
    command: str | list[str] = DEFAULT_CODEX_COMMAND,
    timeout_seconds: float = 600.0,
) -> dict[str, Any]:
    if isinstance(request_json, str):
        payload = json.loads(request_json)
    else:
        payload = dict(request_json)
    request = SDEPExecuteRequest.from_dict(payload)
    request_payload = request.to_dict()
    prompt = build_codex_prompt(request_payload)
    argv = _normalize_command(command)
    command_result = _invoke_codex(argv, prompt=prompt, timeout_seconds=timeout_seconds)
    response_payload = build_codex_sdep_response(
        request_payload=request_payload,
        command=argv,
        prompt=prompt,
        command_result=command_result,
    )
    return SDEPExecuteResponse.from_dict(response_payload).to_dict()


def build_codex_prompt(request_payload: dict[str, Any]) -> str:
    request = SDEPExecuteRequest.from_dict(request_payload)
    execution = request.execution.to_dict()
    execution_input = _dict(execution.get("input"))
    context_pack = _dict(execution_input.get("context_pack"))
    skill_invocation = _dict(execution_input.get("skill_invocation"))
    metadata = _dict(execution.get("metadata"))
    sections = [
        "You are the Codex executor for a Spice-approved decision.",
        "Follow the task and context exactly. Do not re-decide the candidate.",
        "",
        "SPICE ATTRIBUTION",
        f"- request_id: {request.request_id}",
        f"- execution_id: {request.traceability.get('execution_id')}",
        f"- decision_id: {request.traceability.get('spice_decision_id')}",
        f"- trace_ref: {request.traceability.get('trace_ref')}",
        f"- candidate_id: {request.traceability.get('candidate_id')}",
        f"- approval_id: {request.traceability.get('approval_id')}",
        "",
        "SKILL",
        f"- skill_id: {metadata.get('skill_id') or skill_invocation.get('skill_id')}",
        f"- action_type: {execution.get('action_type')}",
        f"- side_effect_class: {metadata.get('side_effect_class')}",
        "",
        "TASK",
        str(context_pack.get("task") or execution.get("action_type") or "Execute the approved task."),
        "",
        "WHY NOW",
        str(context_pack.get("why_now") or "Approved by Spice."),
        "",
        "CONSTRAINTS / DO NOT",
    ]
    for item in _list(context_pack.get("do_not")):
        sections.append(f"- {item}")
    if not _list(context_pack.get("do_not")):
        sections.append("- Preserve Spice attribution in any result summary.")
    sections.extend(
        [
            "",
            "EXPECTED OUTPUT",
            json.dumps(
                context_pack.get("expected_output") or context_pack.get("return_schema") or {},
                ensure_ascii=True,
                sort_keys=True,
            ),
            "",
            "COMPRESSED CONTEXT PACK",
            json.dumps(context_pack, ensure_ascii=True, sort_keys=True, indent=2),
            "",
            "Return a concise execution summary on stdout.",
        ]
    )
    return "\n".join(sections)


def build_codex_sdep_response(
    *,
    request_payload: dict[str, Any],
    command: list[str],
    prompt: str,
    command_result: CodexCommandResult,
) -> dict[str, Any]:
    request = SDEPExecuteRequest.from_dict(request_payload)
    traceability = dict(request.traceability)
    execution = request.execution.to_dict()
    execution_id = str(traceability.get("execution_id") or "")
    execution_report = _execution_report_from_stdout(command_result.stdout)
    task_status = _task_status_from_command_result(command_result, execution_report)
    protocol_status = "success" if task_status in {"success", "completed"} else task_status
    summary = _summary_from_command_result(command_result)
    if execution_report.get("output"):
        summary = str(execution_report.get("output"))[:500]
    return {
        "protocol": "sdep",
        "sdep_version": "0.1",
        "message_type": "execute.response",
        "message_id": f"sdep-msg.codex.{_hash([request.request_id, execution_id, command_result.status])[:16]}",
        "request_id": request.request_id,
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "responder": {
            "id": CODEX_EXECUTOR_ID,
            "name": CODEX_EXECUTOR_NAME,
            "version": "0.1",
            "vendor": "Spice",
            "implementation": CODEX_EXECUTOR_IMPLEMENTATION,
            "role": "executor",
        },
        "status": protocol_status,
        "outcome": {
            "execution_id": execution_id,
            "status": task_status,
            "outcome_type": "observation" if task_status in {"success", "completed"} else "error",
            "output": {
                "summary": summary,
                "stdout": command_result.stdout,
                "stderr": command_result.stderr,
                "exit_code": command_result.exit_code,
                "timed_out": command_result.timed_out,
                "executor_report": execution_report,
                "task": _dict(_dict(execution.get("input")).get("context_pack")).get("task")
                or execution.get("action_type"),
                "state_delta": {
                    "task_status": task_status,
                    "executor_provider": "codex",
                    **_dict(execution_report.get("state_delta")),
                },
            },
            "artifacts": [],
            "metrics": {},
            "metadata": {
                "executor_provider": "codex",
                "codex_command": _command_summary(command),
                "prompt_sha256": _hash(prompt),
                "real_executor": True,
                "transport": "local_subprocess",
            },
        },
        "traceability": {
            "execution_id": execution_id,
            "spice_decision_id": traceability.get("spice_decision_id"),
            "trace_ref": traceability.get("trace_ref"),
            "candidate_id": traceability.get("candidate_id"),
            "approval_id": traceability.get("approval_id"),
            "skill_id": traceability.get("skill_id"),
            "context_pack_id": traceability.get("context_pack_id"),
        },
        "metadata": {
            "executor_provider": "codex",
            "codex_command": _command_summary(command),
            "real_executor": True,
            "transport": "local_subprocess",
        },
    }


def _invoke_codex(
    command: list[str],
    *,
    prompt: str,
    timeout_seconds: float,
) -> CodexCommandResult:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")
    try:
        completed = subprocess.run(
            command,
            input=prompt,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        return CodexCommandResult(
            status="failed",
            stdout=exc.stdout or "",
            stderr=f"Codex command timed out after {timeout_seconds} seconds.",
            exit_code=None,
            timed_out=True,
        )
    except OSError as exc:
        return CodexCommandResult(
            status="failed",
            stdout="",
            stderr=str(exc),
            exit_code=None,
            timed_out=False,
        )
    status = "success" if completed.returncode == 0 else "failed"
    return CodexCommandResult(
        status=status,
        stdout=completed.stdout,
        stderr=completed.stderr,
        exit_code=completed.returncode,
        timed_out=False,
    )


def _summary_from_command_result(result: CodexCommandResult) -> str:
    if result.status == "success":
        text = result.stdout.strip()
        return text[:500] if text else "Codex command completed successfully."
    if result.timed_out:
        return result.stderr
    text = result.stderr.strip() or result.stdout.strip()
    return text[:500] if text else "Codex command failed."


def _execution_report_from_stdout(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _task_status_from_command_result(
    result: CodexCommandResult,
    report: dict[str, Any],
) -> str:
    reported = str(
        report.get("task_status")
        or _dict(report.get("state_delta")).get("task_status")
        or report.get("status")
        or ""
    ).strip().lower()
    if reported:
        if reported in {"ok", "completed"}:
            return "success"
        if reported in {"blocked", "cancelled", "canceled", "failed", "error", "partial"}:
            return reported
        if reported == "success":
            return "success"
    return "success" if result.status == "success" else "failed"


def _normalize_command(command: str | list[str]) -> list[str]:
    if isinstance(command, str):
        argv = shlex.split(command)
    else:
        argv = [str(item) for item in command]
    if not argv:
        raise ValueError("Codex command must be non-empty.")
    return argv


def _command_summary(command: list[str]) -> str:
    return " ".join(shlex.quote(item) for item in command)


def _hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


if __name__ == "__main__":
    raise SystemExit(main())
