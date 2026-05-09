from __future__ import annotations

import tempfile
import unittest

from spice.runtime import LocalJsonStore, setup_workspace


class LocalJsonStoreTests(unittest.TestCase):
    def test_store_loads_and_saves_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            store = LocalJsonStore.from_project_root(tmp_dir)

            state = store.load_state()
            self.assertEqual(state["schema_version"], "spice.workspace.state.v1")

            store.save_state({"schema_version": "custom", "value": 1})
            self.assertEqual(store.load_state(), {"schema_version": "custom", "value": 1})

    def test_store_round_trips_record_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            store = LocalJsonStore.from_project_root(tmp_dir)

            cases = [
                ("session", "session.1", {"session_id": "session.1"}),
                ("run", "run.1", {"run_id": "run.1"}),
                ("decision", "decision.1", {"decision_id": "decision.1"}),
                ("approval", "approval.1", {"approval_id": "approval.1"}),
                ("outcome", "outcome.1", {"outcome_id": "outcome.1"}),
                ("perception", "perception.1", {"perception_id": "perception.1"}),
            ]
            for kind, record_id, payload in cases:
                save = getattr(store, f"save_{kind}")
                load = getattr(store, f"load_{kind}")
                save(record_id, payload)
                self.assertEqual(load(record_id), payload)

            self.assertEqual(store.list_record_ids("sessions"), ["session.1"])
            self.assertEqual(store.list_record_ids("runs"), ["run.1"])
            self.assertEqual(store.list_record_ids("decisions"), ["decision.1"])
            self.assertEqual(store.list_record_ids("approvals"), ["approval.1"])
            self.assertEqual(store.list_record_ids("outcomes"), ["outcome.1"])
            self.assertEqual(store.list_record_ids("perceptions"), ["perception.1"])

    def test_store_sanitizes_record_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            store = LocalJsonStore.from_project_root(tmp_dir)

            path = store.save_run("run:with/slash", {"ok": True})

            self.assertEqual(path.name, "run_with_slash.json")
            self.assertEqual(store.load_run("run:with/slash"), {"ok": True})

    def test_store_rejects_empty_record_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            store = LocalJsonStore.from_project_root(tmp_dir)

            with self.assertRaises(ValueError):
                store.save_run("", {})


if __name__ == "__main__":
    unittest.main()
