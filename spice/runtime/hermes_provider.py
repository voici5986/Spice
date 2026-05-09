from __future__ import annotations

import shlex
import sys
from datetime import datetime
from pathlib import Path

from spice.runtime.hermes_executor import DEFAULT_HERMES_COMMAND
from spice.runtime.sdep_subprocess_executor import (
    SDEPSubprocessExecutionResult,
    execute_sdep_subprocess_approval,
)


HERMES_EXECUTOR_PROVIDER_ID = "hermes"


def execute_hermes_approval(
    approval_id: str,
    *,
    command: str | list[str] = DEFAULT_HERMES_COMMAND,
    project_root: str | Path = ".",
    timeout_seconds: int = 600,
    now: datetime | None = None,
) -> SDEPSubprocessExecutionResult:
    wrapper_command = [
        sys.executable,
        "-m",
        "spice.runtime.hermes_executor",
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
        executor_provider_id=HERMES_EXECUTOR_PROVIDER_ID,
        real_executor_called=True,
    )


def _command_string(command: str | list[str]) -> str:
    if isinstance(command, str):
        return command
    return shlex.join([str(item) for item in command])
