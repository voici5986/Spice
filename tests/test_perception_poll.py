from __future__ import annotations

import json
import sys
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from spice.perception.providers.poll import PollPerceptionProvider


NOW = datetime(2026, 4, 30, 8, 0, tzinfo=timezone.utc)


class PollPerceptionProviderTests(unittest.TestCase):
    def test_url_poll_emits_signal_observation_when_content_changes(self) -> None:
        response = _FakeResponse(b"ci: failing\n")
        with patch("spice.perception.providers.poll.urllib.request.urlopen", return_value=response):
            provider = PollPerceptionProvider(
                url="https://ci.example/status",
                now=NOW,
            )
            results = provider.poll_results()

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].changed)
        observation = results[0].observation
        self.assertIsNotNone(observation)
        payload = observation.to_payload() if observation is not None else {}
        self.assertEqual(payload["kind"], "signal")
        self.assertEqual(payload["source"]["provider"], "poll")
        self.assertEqual(payload["source"]["channel"], "url")
        self.assertEqual(payload["attributes"]["kind"], "url_changed")
        self.assertIn("ci: failing", payload["evidence"][0]["content"])
        json.dumps(results[0].to_payload())

    def test_poll_dedupes_unchanged_content_by_previous_hash(self) -> None:
        first = PollPerceptionProvider(
            command=[sys.executable, "-c", "print('same')"],
            allow_command_poll=True,
            now=NOW,
        ).poll_results()[0]
        second = PollPerceptionProvider(
            command=[sys.executable, "-c", "print('same')"],
            allow_command_poll=True,
            previous_hashes={f"{first.source_type}:{first.source_value}": first.content_hash},
            now=NOW,
        ).poll_results()[0]

        self.assertFalse(second.changed)
        self.assertIsNone(second.observation)

    def test_command_poll_requires_explicit_opt_in(self) -> None:
        provider = PollPerceptionProvider(command=[sys.executable, "-c", "print('x')"])

        with self.assertRaisesRegex(PermissionError, "Command poll is disabled"):
            provider.poll_results()

    def test_command_poll_uses_shell_false_and_emits_observation(self) -> None:
        with patch("spice.perception.providers.poll.subprocess.run") as run:
            run.return_value.stdout = "branch changed\n"
            run.return_value.stderr = ""
            run.return_value.returncode = 0
            provider = PollPerceptionProvider(
                command="git status --short",
                allow_command_poll=True,
                now=NOW,
            )
            result = provider.poll_results()[0]

        _, kwargs = run.call_args
        self.assertFalse(kwargs["shell"])
        self.assertTrue(result.changed)
        self.assertIsNotNone(result.observation)
        payload = result.observation.to_payload() if result.observation is not None else {}
        self.assertEqual(payload["attributes"]["kind"], "command_changed")
        self.assertEqual(payload["metadata"]["source_type"], "command")
        json.dumps(result.to_payload())

    def test_poll_requires_at_least_one_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a poll URL or poll command"):
            PollPerceptionProvider().poll_results()


class _FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, _limit: int) -> bytes:
        return self.content


if __name__ == "__main__":
    unittest.main()
