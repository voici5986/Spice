from __future__ import annotations

import tempfile
import unittest

from spice.runtime import setup_workspace
from spice.runtime.execution_conversation import open_execution_approval_from_frame
from spice.runtime.store import LocalJsonStore


class RuntimeExecutionConversationTests(unittest.TestCase):
    def test_advisory_candidate_never_creates_execution_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            store = LocalJsonStore.from_project_root(tmp_dir)
            frame = _frame(
                _candidate(
                    "candidate.advisory",
                    title="Advisory recommendation",
                    affordance={
                        "candidate_execution_requested": False,
                        "candidate_executable": False,
                        "executor_available": True,
                        "executable": False,
                        "approval": {"required": False, "eligible_for_approval": False},
                    },
                )
            )

            with self.assertRaisesRegex(ValueError, "advisory-only"):
                open_execution_approval_from_frame(
                    store=store,
                    user_input="execute selected",
                    active_frame=frame,
                )

            self.assertEqual(store.list_record_ids("approvals"), [])

    def test_read_only_candidate_never_creates_execution_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            store = LocalJsonStore.from_project_root(tmp_dir)
            frame = _frame(
                _candidate(
                    "candidate.read_only",
                    title="Read repo evidence",
                    action="item.triage",
                    executor_task="read_file spice/runtime/run_once.py and return sources",
                    affordance={
                        "candidate_execution_requested": False,
                        "candidate_executable": False,
                        "executor_available": True,
                        "executable": False,
                        "permission": {
                            "required": "read_only",
                            "side_effect_class": "read_only",
                        },
                        "approval": {"required": False, "eligible_for_approval": False},
                    },
                    skill_resolution={"skill_id": "work_item.triage.read_only"},
                )
            )

            with self.assertRaisesRegex(ValueError, "read-only"):
                open_execution_approval_from_frame(
                    store=store,
                    user_input="execute selected",
                    active_frame=frame,
                )

            self.assertEqual(store.list_record_ids("approvals"), [])

    def test_noop_defer_candidate_never_creates_execution_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            store = LocalJsonStore.from_project_root(tmp_dir)
            frame = _frame(
                _candidate(
                    "candidate.defer",
                    title="Defer this decision",
                    action="time.defer",
                    recommended_action="Record this as an open loop and revisit later.",
                    affordance={
                        "candidate_execution_requested": False,
                        "candidate_executable": False,
                        "executor_available": True,
                        "executable": False,
                        "approval": {"required": False, "eligible_for_approval": False},
                    },
                )
            )

            with self.assertRaisesRegex(ValueError, "no-op/defer/record-only"):
                open_execution_approval_from_frame(
                    store=store,
                    user_input="execute selected",
                    active_frame=frame,
                )

            self.assertEqual(store.list_record_ids("approvals"), [])


def _frame(candidate: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "0.1",
        "frame_id": "frame.test",
        "run_id": "run.test",
        "decision_id": "decision.test",
        "trace_ref": "trace.test",
        "session_id": "session.default",
        "selected_candidate_id": candidate["candidate_id"],
        "selected": candidate,
        "candidates": [candidate],
        "approval_id": "",
    }


def _candidate(
    candidate_id: str,
    *,
    title: str,
    action: str = "item.triage",
    recommended_action: str = "",
    executor_task: str = "",
    affordance: dict[str, object],
    skill_resolution: dict[str, object] | None = None,
) -> dict[str, object]:
    execution_affordance = {
        "schema_version": "0.1",
        "generated_by": "spice.runtime.execution_affordance",
        "blocked": True,
        "blockers": [],
        "executor": {"executor_id": "dry_run", "status": "ready"},
        **affordance,
    }
    return {
        "label": "A",
        "candidate_id": candidate_id,
        "title": title,
        "action": action,
        "intent": title,
        "recommended_action": recommended_action or title,
        "expected_result": "",
        "executor_task": executor_task,
        "requires_confirmation": False,
        "is_selected": True,
        "execution_affordance": execution_affordance,
        "skill_resolution": dict(skill_resolution or {}),
    }


if __name__ == "__main__":
    unittest.main()
