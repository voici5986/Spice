from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone

from spice.runtime import (
    DEFAULT_SESSION_ID,
    LocalJsonStore,
    SessionRecord,
    archive_session,
    build_session_timeline,
    delete_session,
    load_workspace_config,
    list_sessions,
    render_session_list,
    render_session_resume,
    search_sessions,
    session_stats,
    run_once,
    set_workspace_active_session,
    setup_workspace,
)


NOW = datetime(2026, 4, 29, 6, 0, tzinfo=timezone.utc)


class RuntimeSessionTests(unittest.TestCase):
    def test_run_once_creates_default_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)

            result = run_once("Review the project.", project_root=tmp_dir, now=NOW)
            store = LocalJsonStore.from_project_root(tmp_dir)
            session = SessionRecord.from_payload(store.load_session(DEFAULT_SESSION_ID))

            self.assertTrue(result.session_path.exists())
            self.assertEqual(result.artifact["session_id"], DEFAULT_SESSION_ID)
            self.assertEqual(session.session_id, DEFAULT_SESSION_ID)
            self.assertEqual(session.run_ids, [result.artifact["run_id"]])
            self.assertEqual(session.decision_ids, [result.artifact["decision_id"]])
            self.assertEqual(session.last_run_id, result.artifact["run_id"])
            self.assertEqual(session.last_decision_id, result.artifact["decision_id"])
            self.assertEqual(session.last_trace_ref, result.artifact["trace_ref"])
            self.assertEqual(session.active_state_ref, result.artifact["state_after_ref"])
            self.assertEqual(result.artifact["session"]["last_run_id"], result.artifact["run_id"])
            self.assertIn("session", result.artifact["store_paths"])
            json.dumps(session.to_payload())

    def test_repeated_runs_append_to_same_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)

            first = run_once("Review the project.", project_root=tmp_dir, now=NOW)
            second = run_once("Review the project again.", project_root=tmp_dir, now=NOW)
            store = LocalJsonStore.from_project_root(tmp_dir)
            session = SessionRecord.from_payload(store.load_session(DEFAULT_SESSION_ID))

            self.assertEqual(session.run_ids, [first.artifact["run_id"], second.artifact["run_id"]])
            self.assertEqual(
                session.decision_ids,
                [first.artifact["decision_id"], second.artifact["decision_id"]],
            )
            self.assertEqual(session.last_run_id, second.artifact["run_id"])
            self.assertEqual(session.last_decision_id, second.artifact["decision_id"])
            self.assertEqual(len(store.list_record_ids("sessions")), 1)

    def test_list_and_render_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            run_once("Review the project.", project_root=tmp_dir, now=NOW)
            store = LocalJsonStore.from_project_root(tmp_dir)

            sessions = list_sessions(store)
            rendered_list = render_session_list(sessions)
            rendered_resume = render_session_resume(sessions[0])

            self.assertEqual(len(sessions), 1)
            self.assertIn(DEFAULT_SESSION_ID, rendered_list)
            self.assertIn("SPICE SESSION RESUME", rendered_resume)
            self.assertIn("LAST DECISION", rendered_resume)
            self.assertIn("RESUME COMMANDS", rendered_resume)
            self.assertIn(f"spice run --session-id {DEFAULT_SESSION_ID}", rendered_resume)

    def test_active_session_can_be_switched_in_workspace_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            run_once("Review the project.", project_root=tmp_dir, now=NOW, session_id="session.alpha")

            config = set_workspace_active_session(tmp_dir, "session.alpha")
            loaded = load_workspace_config(tmp_dir)

            self.assertEqual(config.active_session_id, "session.alpha")
            self.assertEqual(loaded.active_session_id, "session.alpha")

    def test_archive_hides_session_from_default_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            run_once("Review the project.", project_root=tmp_dir, now=NOW, session_id="session.alpha")
            store = LocalJsonStore.from_project_root(tmp_dir)

            archived = archive_session(store, "session.alpha", now=NOW)

            self.assertEqual(archived.status, "archived")
            self.assertEqual(list_sessions(store), [])
            self.assertEqual([session.session_id for session in list_sessions(store, include_archived=True)], ["session.alpha"])

    def test_session_timeline_and_search_use_run_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            first = run_once("Review auth risk.", project_root=tmp_dir, now=NOW, session_id="session.alpha")
            second = run_once("Plan database migration.", project_root=tmp_dir, now=NOW, session_id="session.alpha")
            store = LocalJsonStore.from_project_root(tmp_dir)
            session = SessionRecord.from_payload(store.load_session("session.alpha"))

            timeline = build_session_timeline(store, session)
            matches = search_sessions(store, "database")

            self.assertEqual([entry.run_id for entry in timeline], [first.artifact["run_id"], second.artifact["run_id"]])
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0].run_id, second.artifact["run_id"])
            self.assertIn("database", matches[0].summary.lower())

    def test_session_stats_count_sessions_and_approvals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            run_once("Fix the failing test.", project_root=tmp_dir, now=NOW, session_id="session.alpha", run_intent_mode="act")
            store = LocalJsonStore.from_project_root(tmp_dir)

            stats = session_stats(store)

            self.assertEqual(stats["total_sessions"], 1)
            self.assertEqual(stats["total_runs"], 1)
            self.assertEqual(stats["total_decisions"], 1)
            self.assertEqual(stats["approval_counts"]["pending"], 1)
            self.assertEqual(stats["most_active_session"], "session.alpha")

    def test_delete_session_can_cascade_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            result = run_once("Fix the failing test.", project_root=tmp_dir, now=NOW, session_id="session.alpha", run_intent_mode="act")
            store = LocalJsonStore.from_project_root(tmp_dir)

            deleted = delete_session(store, "session.alpha", cascade=True)

            self.assertIn("session.alpha", deleted["deleted"]["sessions"])
            self.assertIn(result.artifact["run_id"], deleted["deleted"]["runs"])
            self.assertIn(result.artifact["decision_id"], deleted["deleted"]["decisions"])
            self.assertIn(result.artifact["approval_id"], deleted["deleted"]["approvals"])
            self.assertEqual(store.list_record_ids("sessions"), [])
            self.assertEqual(store.list_record_ids("runs"), [])


if __name__ == "__main__":
    unittest.main()
