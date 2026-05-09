from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone

from examples.decision_hub_demo.smoke_general_loop import (
    build_smoke_artifact,
    main as smoke_main,
    validate_general_loop_smoke,
)


NOW = datetime(2026, 4, 17, 6, 0, tzinfo=timezone.utc)


class DecisionHubGeneralSmokeTests(unittest.TestCase):
    def test_smoke_validator_accepts_full_loop_artifact(self) -> None:
        artifact = build_smoke_artifact(now=NOW, use_bars=False)

        self.assertEqual(validate_general_loop_smoke(artifact), [])

    def test_smoke_validator_rejects_missing_read_only_flag(self) -> None:
        artifact = build_smoke_artifact(now=NOW, use_bars=False)
        artifact["read_only"] = False

        failures = validate_general_loop_smoke(artifact)

        self.assertTrue(any("read_only expected True" in failure for failure in failures))

    def test_smoke_validator_rejects_missing_context_pack_link(self) -> None:
        artifact = build_smoke_artifact(now=NOW, use_bars=False)
        artifact["execution_artifact"]["sdep_request"]["execution"]["input"]["context_pack"][
            "context_pack_id"
        ] = "context_pack.other"

        failures = validate_general_loop_smoke(artifact)

        self.assertTrue(
            any("context_pack.context_pack_id must match" in failure for failure in failures)
        )

    def test_smoke_main_prints_human_readable_output_and_ok(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = smoke_main([])

        rendered = stdout.getvalue()

        self.assertEqual(exit_code, 0)
        self.assertIn("SPICE DECISION LOOP", rendered)
        self.assertIn("SPICE GENERAL LOOP SMOKE: OK", rendered)
        self.assertIn("no executor called | no SDEP sent | no state persisted", rendered)

    def test_smoke_main_quiet_prints_only_summary(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = smoke_main(["--quiet"])

        rendered = stdout.getvalue()

        self.assertEqual(exit_code, 0)
        self.assertIn("SPICE GENERAL LOOP SMOKE: OK", rendered)
        self.assertNotIn("0. INPUT SIGNALS", rendered)

    def test_smoke_main_json_outputs_parseable_artifact(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = smoke_main(["--json"])

        payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["path_type"], "read_only_general_full_loop")
        self.assertFalse(payload["executor_called"])
        self.assertFalse(payload["sdep_request_sent"])


if __name__ == "__main__":
    unittest.main()
