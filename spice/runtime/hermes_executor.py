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


HERMES_EXECUTOR_ID = "hermes"
HERMES_EXECUTOR_NAME = "Hermes Executor"
HERMES_EXECUTOR_IMPLEMENTATION = "spice.runtime.hermes_executor"
DEFAULT_HERMES_COMMAND = "hermes chat -Q"


@dataclass(frozen=True, slots=True)
class HermesCommandResult:
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
        prog="python -m spice.runtime.hermes_executor",
        description="SDEP-compatible Hermes subprocess executor.",
    )
    parser.add_argument(
        "--command",
        default=DEFAULT_HERMES_COMMAND,
        help=(
            "Hermes command to run. For `hermes chat` commands, the SDEP-derived "
            "task prompt is passed as `-q QUERY`; custom commands receive the "
            "prompt on stdin."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Timeout in seconds for the Hermes command.",
    )
    args = parser.parse_args(argv)
    try:
        response = execute_hermes_sdep_request(
            sys.stdin.read(),
            command=args.command,
            timeout_seconds=float(args.timeout),
        )
        sys.stdout.write(json.dumps(response, ensure_ascii=True, sort_keys=True))
        return 0
    except Exception as exc:
        sys.stderr.write(f"hermes executor failed: {exc}\n")
        return 1


def execute_hermes_sdep_request(
    request_json: str | dict[str, Any],
    *,
    command: str | list[str] = DEFAULT_HERMES_COMMAND,
    timeout_seconds: float = 600.0,
) -> dict[str, Any]:
    if isinstance(request_json, str):
        payload = json.loads(request_json)
    else:
        payload = dict(request_json)
    request = SDEPExecuteRequest.from_dict(payload)
    request_payload = request.to_dict()
    prompt = build_hermes_prompt(request_payload)
    argv = _normalize_command(command)
    command_result = _invoke_hermes(argv, prompt=prompt, timeout_seconds=timeout_seconds)
    response_payload = build_hermes_sdep_response(
        request_payload=request_payload,
        command=argv,
        prompt=prompt,
        command_result=command_result,
    )
    return SDEPExecuteResponse.from_dict(response_payload).to_dict()


def build_hermes_prompt(request_payload: dict[str, Any]) -> str:
    request = SDEPExecuteRequest.from_dict(request_payload)
    execution = request.execution.to_dict()
    execution_input = _dict(execution.get("input"))
    context_pack = _dict(execution_input.get("context_pack"))
    skill_invocation = _dict(execution_input.get("skill_invocation"))
    metadata = _dict(execution.get("metadata"))
    do_not = _list(context_pack.get("do_not"))
    sections = [
        "You are Hermes executing a Spice-approved decision.",
        "Use the approved task/context handoff. Do not re-decide or replace Spice policy.",
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
        "DO NOT",
    ]
    if do_not:
        sections.extend(f"- {item}" for item in do_not)
    else:
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


def build_hermes_sdep_response(
    *,
    request_payload: dict[str, Any],
    command: list[str],
    prompt: str,
    command_result: HermesCommandResult,
) -> dict[str, Any]:
    request = SDEPExecuteRequest.from_dict(request_payload)
    traceability = dict(request.traceability)
    execution = request.execution.to_dict()
    execution_id = str(traceability.get("execution_id") or "")
    task_status = "success" if command_result.status == "success" else "failed"
    summary = _summary_from_command_result(command_result)
    return {
        "protocol": "sdep",
        "sdep_version": "0.1",
        "message_type": "execute.response",
        "message_id": f"sdep-msg.hermes.{_hash([request.request_id, execution_id, command_result.status])[:16]}",
        "request_id": request.request_id,
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "responder": {
            "id": HERMES_EXECUTOR_ID,
            "name": HERMES_EXECUTOR_NAME,
            "version": "0.1",
            "vendor": "Spice",
            "implementation": HERMES_EXECUTOR_IMPLEMENTATION,
            "role": "executor",
        },
        "status": "success",
        "outcome": {
            "execution_id": execution_id,
            "status": task_status,
            "outcome_type": "observation",
            "output": {
                "summary": summary,
                "stdout": command_result.stdout,
                "stderr": command_result.stderr,
                "exit_code": command_result.exit_code,
                "timed_out": command_result.timed_out,
                "task": _dict(_dict(execution.get("input")).get("context_pack")).get("task")
                or execution.get("action_type"),
                "state_delta": {
                    "task_status": task_status,
                    "executor_provider": "hermes",
                },
            },
            "artifacts": [],
            "metrics": {},
            "metadata": {
                "executor_provider": "hermes",
                "hermes_command": _command_summary(command),
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
            "executor_provider": "hermes",
            "hermes_command": _command_summary(command),
            "real_executor": True,
            "transport": "local_subprocess",
        },
    }


def _invoke_hermes(
    command: list[str],
    *,
    prompt: str,
    timeout_seconds: float,
) -> HermesCommandResult:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")
    run_command, stdin_text = _prepare_hermes_invocation(command, prompt)
    try:
        completed = subprocess.run(
            run_command,
            input=stdin_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        return HermesCommandResult(
            status="failed",
            stdout=exc.stdout or "",
            stderr=f"Hermes command timed out after {timeout_seconds} seconds.",
            exit_code=None,
            timed_out=True,
        )
    except OSError as exc:
        return HermesCommandResult(
            status="failed",
            stdout="",
            stderr=str(exc),
            exit_code=None,
            timed_out=False,
        )
    status = "success" if completed.returncode == 0 else "failed"
    return HermesCommandResult(
        status=status,
        stdout=completed.stdout,
        stderr=completed.stderr,
        exit_code=completed.returncode,
        timed_out=False,
    )


def _prepare_hermes_invocation(command: list[str], prompt: str) -> tuple[list[str], str | None]:
    if any(item == "{prompt}" for item in command):
        return [prompt if item == "{prompt}" else item for item in command], None
    if _is_hermes_chat_command(command) and not _has_query_argument(command):
        return [*command, "-q", prompt], None
    return list(command), prompt


def _is_hermes_chat_command(command: list[str]) -> bool:
    if len(command) < 2:
        return False
    executable = command[0].rsplit("/", 1)[-1]
    return executable == "hermes" and command[1] == "chat"


def _has_query_argument(command: list[str]) -> bool:
    return any(item == "-q" or item == "--query" or item.startswith("--query=") for item in command)


def _summary_from_command_result(result: HermesCommandResult) -> str:
    if result.status == "success":
        text = result.stdout.strip()
        return text[:500] if text else "Hermes command completed successfully."
    if result.timed_out:
        return result.stderr
    text = result.stderr.strip() or result.stdout.strip()
    return text[:500] if text else "Hermes command failed."


def _normalize_command(command: str | list[str]) -> list[str]:
    if isinstance(command, str):
        argv = shlex.split(command)
    else:
        argv = [str(item) for item in command]
    if not argv:
        raise ValueError("Hermes command must be non-empty.")
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
