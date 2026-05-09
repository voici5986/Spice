from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

from spice.protocols.sdep import SDEPExecuteRequest, SDEPExecuteResponse


def main() -> int:
    try:
        request_payload = json.loads(sys.stdin.read())
        if not isinstance(request_payload, dict):
            raise ValueError("stdin must contain a JSON object.")
        request = SDEPExecuteRequest.from_dict(request_payload)
        traceability = dict(request.traceability)
        execution = request.execution.to_dict()
        execution_id = str(traceability.get("execution_id") or "")
        candidate_id = str(traceability.get("candidate_id") or "")
        context_pack = _dict(execution.get("input")).get("context_pack")
        task = str(_dict(context_pack).get("task") or execution.get("action_type") or "planned action")
        response = {
            "protocol": "sdep",
            "sdep_version": "0.1",
            "message_type": "execute.response",
            "message_id": f"sdep-msg.echo.{_hash([request.request_id, execution_id])[:16]}",
            "request_id": request.request_id,
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "responder": {
                "id": "spice.sdep_echo_executor",
                "name": "Spice SDEP Echo Executor",
                "version": "0.1",
                "vendor": "Spice",
                "implementation": "local-subprocess-fixture",
                "role": "executor",
            },
            "status": "success",
            "outcome": {
                "execution_id": execution_id,
                "status": "success",
                "outcome_type": "observation",
                "output": {
                    "summary": "SDEP echo executor received the request and returned a fixture response.",
                    "task": task,
                    "fixture": True,
                    "state_delta": {
                        "updated_refs": [candidate_id] if candidate_id else [],
                        "task_status": "success",
                        "fixture": True,
                    },
                },
                "artifacts": [],
                "metrics": {},
                "metadata": {
                    "executor_provider": "sdep_echo",
                    "real_executor": False,
                    "fixture": True,
                },
            },
            "traceability": {
                "execution_id": execution_id,
                "spice_decision_id": traceability.get("spice_decision_id"),
                "trace_ref": traceability.get("trace_ref"),
                "candidate_id": candidate_id,
                "approval_id": traceability.get("approval_id"),
                "skill_id": traceability.get("skill_id"),
                "context_pack_id": traceability.get("context_pack_id"),
            },
            "metadata": {
                "executor_provider": "sdep_echo",
                "real_executor": False,
                "fixture": True,
            },
        }
        sys.stdout.write(json.dumps(SDEPExecuteResponse.from_dict(response).to_dict(), sort_keys=True))
        return 0
    except Exception as exc:
        sys.stderr.write(f"sdep echo executor failed: {exc}\n")
        return 1


def _hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
