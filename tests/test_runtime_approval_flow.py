from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone

from spice.decision.general.approval import Approval
from spice.runtime import (
    LocalJsonStore,
    approve_approval,
    list_approvals,
    load_approval,
    reject_approval,
    render_approval_details,
    render_approval_list,
    setup_workspace,
)


NOW = datetime(2026, 4, 29, 6, 0, tzinfo=timezone.utc)


class RuntimeApprovalFlowTests(unittest.TestCase):
    def test_approval_list_and_details_are_readable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = _setup_with_pending_approval(tmp_dir)

            approvals = list_approvals(store)
            pending = list_approvals(store, status="pending")
            details = render_approval_details(approvals[0])
            rendered_list = render_approval_list(approvals)

            self.assertEqual([item.approval_id for item in approvals], ["approval.test.1"])
            self.assertEqual(len(pending), 1)
            self.assertIn("SPICE APPROVALS", rendered_list)
            self.assertIn("approval.test.1", rendered_list)
            self.assertIn("SPICE APPROVAL", details)
            self.assertIn("no executor is called", details)

    def test_approve_updates_approval_run_decision_and_session_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = _setup_with_pending_approval(tmp_dir)

            result = approve_approval(store, "approval.test.1", reason="Looks safe.", now=NOW)
            approval = load_approval(store, "approval.test.1")
            run = store.load_run("run.test.1")
            decision = store.load_decision("decision.test.1")
            session = store.load_session("session.default")

            self.assertEqual(result.approval.status, "approved")
            self.assertTrue(result.approval.execution_allowed)
            self.assertFalse(result.to_payload()["executor_called"])
            self.assertFalse(result.to_payload()["sdep_request_sent"])
            self.assertFalse(result.to_payload()["executed"])
            self.assertEqual(approval.status, "approved")
            self.assertEqual(run["approval"]["status"], "approved")
            self.assertEqual(
                decision["checkpoint"]["approval"]["status"],
                "approved",
            )
            self.assertEqual(session["pending_approval_ids"], [])
            self.assertEqual(result.synced_runs, ["run.test.1"])
            self.assertEqual(result.synced_decisions, ["decision.test.1"])
            self.assertEqual(result.synced_sessions, ["session.default"])
            json.dumps(result.to_payload())

    def test_reject_updates_status_and_keeps_execution_disallowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = _setup_with_pending_approval(tmp_dir)

            result = reject_approval(store, "approval.test.1", reason="Too risky.", now=NOW)
            approval = load_approval(store, "approval.test.1")

            self.assertEqual(result.approval.status, "rejected")
            self.assertFalse(result.approval.execution_allowed)
            self.assertEqual(approval.status, "rejected")
            self.assertFalse(approval.execution_allowed)
            self.assertEqual(approval.reason, "Too risky.")

    def test_resolved_approval_cannot_be_approved_again(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = _setup_with_pending_approval(tmp_dir)
            approve_approval(store, "approval.test.1", now=NOW)

            with self.assertRaisesRegex(ValueError, "not pending"):
                approve_approval(store, "approval.test.1", now=NOW)


def _setup_with_pending_approval(tmp_dir: str) -> LocalJsonStore:
    setup_workspace(project_root=tmp_dir)
    store = LocalJsonStore.from_project_root(tmp_dir)
    approval = Approval(
        approval_id="approval.test.1",
        decision_id="decision.test.1",
        candidate_id="candidate.test.1",
        status="pending",
        mode="confirm_before_execution",
        requested_at="2026-04-29T06:00:00+00:00",
        prompt="Approve candidate.test.1?",
        execution_allowed=False,
    )
    approval_payload = approval.to_payload()
    store.save_approval(approval.approval_id, approval_payload)
    store.save_decision(
        approval.decision_id,
        {
            "checkpoint": {
                "decision_id": approval.decision_id,
                "approval": approval_payload,
            }
        },
    )
    store.save_run(
        "run.test.1",
        {
            "run_id": "run.test.1",
            "decision_id": approval.decision_id,
            "approval_id": approval.approval_id,
            "approval": approval_payload,
            "decision": {
                "checkpoint": {
                    "decision_id": approval.decision_id,
                    "approval": approval_payload,
                }
            },
            "session": {
                "session_id": "session.default",
                "pending_approval_ids": [approval.approval_id],
            },
        },
    )
    store.save_session(
        "session.default",
        {
            "session_id": "session.default",
            "created_at": "2026-04-29T06:00:00+00:00",
            "updated_at": "2026-04-29T06:00:00+00:00",
            "status": "active",
            "run_ids": ["run.test.1"],
            "decision_ids": [approval.decision_id],
            "approval_ids": [approval.approval_id],
            "active_state_ref": ".spice/state/state.json#after:test",
            "last_run_id": "run.test.1",
            "last_decision_id": approval.decision_id,
            "last_trace_ref": "trace.test.1",
            "pending_approval_ids": [approval.approval_id],
            "metadata": {},
        },
    )
    return store


if __name__ == "__main__":
    unittest.main()
