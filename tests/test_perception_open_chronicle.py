from __future__ import annotations

import json
import urllib.error
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from spice.perception.providers.open_chronicle import (
    OpenChronicleMCPClient,
    OpenChroniclePerceptionProvider,
)


NOW = datetime(2026, 5, 1, 8, 0, tzinfo=timezone.utc)


class OpenChroniclePerceptionProviderTests(unittest.TestCase):
    def test_open_chronicle_emits_signal_observations_for_changed_mcp_tools(self) -> None:
        provider = OpenChroniclePerceptionProvider(
            mcp_url="http://127.0.0.1:8742/mcp",
            now=NOW,
            client=_FakeOpenChronicleClient(
                {
                    "current_context": {"content": [{"type": "text", "text": "VS Code: spice/entry/cli.py"}]},
                    "recent_activity": {"content": [{"type": "text", "text": "Edited runtime perception code"}]},
                }
            ),
        )

        results = provider.poll_results()

        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.changed for result in results))
        observation = results[0].observation
        self.assertIsNotNone(observation)
        payload = observation.to_payload() if observation is not None else {}
        self.assertEqual(payload["kind"], "signal")
        self.assertEqual(payload["source"]["provider"], "open_chronicle")
        self.assertEqual(payload["source"]["channel"], "mcp")
        self.assertEqual(payload["attributes"]["source"], "open_chronicle")
        self.assertIn("VS Code", payload["evidence"][0]["content"])
        json.dumps(results[0].to_payload())

    def test_open_chronicle_dedupes_unchanged_tool_content(self) -> None:
        first = OpenChroniclePerceptionProvider(
            now=NOW,
            client=_FakeOpenChronicleClient(
                {
                    "current_context": {"content": [{"type": "text", "text": "same"}]},
                    "recent_activity": {"content": [{"type": "text", "text": "same activity"}]},
                }
            ),
        ).poll_results()[0]
        second = OpenChroniclePerceptionProvider(
            previous_hashes={f"{first.source_type}:{first.source_value}": first.content_hash},
            now=NOW,
            client=_FakeOpenChronicleClient(
                {
                    "current_context": {"content": [{"type": "text", "text": "same"}]},
                    "recent_activity": {"content": [{"type": "text", "text": "same activity"}]},
                }
            ),
        ).poll_results()[0]

        self.assertFalse(second.changed)
        self.assertIsNone(second.observation)

    def test_mcp_client_posts_tools_call_json_rpc(self) -> None:
        response = _FakeResponse(
            {
                "jsonrpc": "2.0",
                "id": "spice.open_chronicle.current_context",
                "result": {"content": [{"type": "text", "text": "context"}]},
            }
        )
        with patch(
            "spice.perception.providers.open_chronicle.urllib.request.urlopen",
            return_value=response,
        ) as urlopen:
            result = OpenChronicleMCPClient(
                mcp_url="http://127.0.0.1:8742/mcp",
            ).call_tool("current_context", {"limit": 3})

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["method"], "tools/call")
        self.assertEqual(payload["params"]["name"], "current_context")
        self.assertEqual(payload["params"]["arguments"]["limit"], 3)
        self.assertEqual(result["content"][0]["text"], "context")

    def test_mcp_client_unreachable_has_clear_error(self) -> None:
        with patch(
            "spice.perception.providers.open_chronicle.urllib.request.urlopen",
            side_effect=urllib.error.URLError("refused"),
        ):
            with self.assertRaisesRegex(ConnectionError, "Open Chronicle MCP endpoint not reachable"):
                OpenChronicleMCPClient().call_tool("current_context", {})


class _FakeOpenChronicleClient:
    def __init__(self, responses: dict[str, dict[str, object]]) -> None:
        self.responses = responses

    def call_tool(self, tool_name: str, arguments: dict[str, object] | None = None) -> dict[str, object]:
        return self.responses[tool_name]


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, _limit: int) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


if __name__ == "__main__":
    unittest.main()
