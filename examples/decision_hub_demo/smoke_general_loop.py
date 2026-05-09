from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.decision_hub_demo.general_loop import build_general_loop_artifact


REQUIRED_TOP_LEVEL_FIELDS = (
    "path_type",
    "generated_by",
    "loop_status",
    "read_only",
    "sdep_request_sent",
    "executor_called",
    "executed",
    "persisted",
    "update_mode",
    "state_snapshot_updated",
    "decision_id",
    "trace_ref",
    "selected_candidate_id",
    "approval_id",
    "execution_id",
    "request_id",
    "outcome_id",
    "skill_id",
    "executor_id",
    "context_pack_id",
    "compare_payload",
    "approval_artifact",
    "execution_artifact",
    "outcome_artifact",
    "state_feedback_artifact",
    "rendered_text",
)

TEXT_MARKERS = (
    "SPICE DECISION LOOP",
    "no executor called | no SDEP sent | no state persisted",
    "0. INPUT SIGNALS",
    "1. GENERAL STATE",
    "2. CANDIDATE DECISIONS",
    "3. SELECTED DECISION",
    "4. WHY NOT OTHERS",
    "5. APPROVAL CHECKPOINT",
    "6. EXECUTION HANDOFF",
    "7. EXECUTION BOUNDARY",
    "8. OUTCOME RETURN",
    "9. STATE FEEDBACK",
    "10. TRACE",
    "planned_executor:",
    "context_pack_id:",
    "sdep_request_sent: false",
    "executor_called: false",
    "persisted: false",
)


def validate_general_loop_smoke(artifact: dict[str, Any]) -> list[str]:
    """Return smoke-check failures for a full General loop artifact."""

    failures: list[str] = []
    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in artifact:
            failures.append(f"missing top-level field: {field}")

    expected_pairs = {
        "path_type": "read_only_general_full_loop",
        "loop_status": "completed_read_only",
        "read_only": True,
        "sdep_request_sent": False,
        "executor_called": False,
        "executed": False,
        "persisted": False,
        "update_mode": "read_only_snapshot",
        "state_snapshot_updated": True,
    }
    for key, expected in expected_pairs.items():
        if artifact.get(key) != expected:
            failures.append(f"{key} expected {expected!r}, got {artifact.get(key)!r}")

    rendered = artifact.get("rendered_text")
    if not isinstance(rendered, str) or not rendered.strip():
        failures.append("rendered_text must be a non-empty string")
    else:
        for marker in TEXT_MARKERS:
            if marker not in rendered:
                failures.append(f"rendered_text missing marker: {marker}")
        if "executed successfully" in rendered:
            failures.append("rendered_text must not imply real execution")

    execution = _dict(artifact.get("execution_artifact"))
    sdep_request = _dict(execution.get("sdep_request"))
    sdep_execution = _dict(sdep_request.get("execution"))
    sdep_input = _dict(sdep_execution.get("input"))
    skill_hint = _dict(sdep_input.get("skill_hint"))
    context_pack = _dict(sdep_input.get("context_pack"))

    if sdep_request.get("message_type") != "execute.request":
        failures.append("sdep_request.message_type must be execute.request")
    if sdep_request.get("request_id") != artifact.get("request_id"):
        failures.append("sdep_request.request_id must match artifact.request_id")
    if skill_hint.get("skill_id") != artifact.get("skill_id"):
        failures.append("skill_hint.skill_id must match artifact.skill_id")
    if context_pack.get("context_pack_id") != artifact.get("context_pack_id"):
        failures.append("context_pack.context_pack_id must match artifact.context_pack_id")
    if context_pack.get("execution_id") != artifact.get("execution_id"):
        failures.append("context_pack.execution_id must match artifact.execution_id")

    outcome = _dict(artifact.get("outcome_artifact"))
    state_feedback = _dict(artifact.get("state_feedback_artifact"))
    if outcome.get("outcome_id") != artifact.get("outcome_id"):
        failures.append("outcome_artifact.outcome_id must match artifact.outcome_id")
    if state_feedback.get("outcome_id") != artifact.get("outcome_id"):
        failures.append("state_feedback_artifact.outcome_id must match artifact.outcome_id")

    try:
        json.loads(json.dumps(artifact))
    except (TypeError, ValueError) as exc:
        failures.append(f"artifact must be JSON serializable: {exc}")

    return failures


def build_smoke_artifact(
    *,
    now: datetime | None = None,
    use_bars: bool = False,
) -> dict[str, Any]:
    return build_general_loop_artifact(now=now, use_bars=use_bars)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the read-only General Decision Loop smoke check.",
    )
    parser.add_argument(
        "--bars",
        action="store_true",
        help="Render score bars in the nested Decision Card when available.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the smoke result, not the human-readable loop output.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full JSON artifact after validation.",
    )
    args = parser.parse_args(argv)

    artifact = build_smoke_artifact(
        now=datetime.now(timezone.utc).replace(microsecond=0),
        use_bars=args.bars,
    )
    failures = validate_general_loop_smoke(artifact)
    if failures:
        print("SPICE GENERAL LOOP SMOKE: FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1

    if args.json:
        print(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if not args.quiet:
        print(str(artifact["rendered_text"]).rstrip())
        print()

    print("SPICE GENERAL LOOP SMOKE: OK")
    print(f"decision_id: {artifact['decision_id']}")
    print(f"trace_ref: {artifact['trace_ref']}")
    print(f"request_id: {artifact['request_id']}")
    print(f"outcome_id: {artifact['outcome_id']}")
    return 0


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
