from __future__ import annotations

import sys
import shlex
from datetime import datetime
from pathlib import Path

from spice.runtime.codex_executor import DEFAULT_CODEX_COMMAND
from spice.runtime.sdep_subprocess_executor import (
    SDEPSubprocessExecutionResult,
    execute_sdep_subprocess_approval,
)


CODEX_EXECUTOR_PROVIDER_ID = "codex"


def execute_codex_approval(
    approval_id: str,
    *,
    command: str | list[str] = DEFAULT_CODEX_COMMAND,
    project_root: str | Path = ".",
    timeout_seconds: int = 600,
    now: datetime | None = None,
) -> SDEPSubprocessExecutionResult:
    wrapper_command = [
        sys.executable,
        "-m",
        "spice.runtime.codex_executor",
        "--command",
        _command_string(command),
        "--timeout",
        str(timeout_seconds),
    ]
    return execute_sdep_subprocess_approval(
        approval_id,
        command=wrapper_command,
        project_root=project_root,
        timeout_seconds=timeout_seconds + 5,
        now=now,
        executor_provider_id=CODEX_EXECUTOR_PROVIDER_ID,
        real_executor_called=True,
    )


def _command_string(command: str | list[str]) -> str:
    if isinstance(command, str):
        return command
    return shlex.join([str(item) for item in command])
