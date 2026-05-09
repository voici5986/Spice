from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

from spice.decision.general import (
    GenericObservation,
    ObservationConfidence,
    ObservationEvidence,
    ObservationKind,
    ObservationSource,
    ObservationSubject,
)
from spice.decision.general.types import payload_value
from spice.perception.provider import PerceptionProvider


MAX_OPEN_CHRONICLE_CONTENT_BYTES = 24 * 1024
DEFAULT_OPEN_CHRONICLE_MCP_URL = "http://127.0.0.1:8742/mcp"


@dataclass(frozen=True, slots=True)
class OpenChronicleResult:
    source_type: str
    source_value: str
    content: str
    content_hash: str
    changed: bool
    observation: GenericObservation | None = None
    previous_hash: str = ""
    metadata: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


class OpenChronicleMCPClient:
    def __init__(self, *, mcp_url: str = DEFAULT_OPEN_CHRONICLE_MCP_URL, timeout_seconds: int = 10) -> None:
        self.mcp_url = mcp_url.strip() or DEFAULT_OPEN_CHRONICLE_MCP_URL
        self.timeout_seconds = timeout_seconds

    def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        payload = {
            "jsonrpc": "2.0",
            "id": f"spice.open_chronicle.{tool_name}",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": dict(arguments or {}),
            },
        }
        request = urllib.request.Request(
            self.mcp_url,
            data=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                content = response.read(MAX_OPEN_CHRONICLE_CONTENT_BYTES + 1)
        except (OSError, urllib.error.URLError) as exc:
            raise ConnectionError(
                f"Open Chronicle MCP endpoint not reachable ({self.mcp_url}). "
                "Next: Start Open Chronicle with `openchronicle start`."
            ) from exc
        if len(content) > MAX_OPEN_CHRONICLE_CONTENT_BYTES:
            content = content[:MAX_OPEN_CHRONICLE_CONTENT_BYTES]
        try:
            response_payload = json.loads(content.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            raise ValueError("Open Chronicle MCP endpoint returned non-JSON response.") from exc
        if not isinstance(response_payload, dict):
            raise ValueError("Open Chronicle MCP response must be a JSON object.")
        error = response_payload.get("error")
        if error:
            raise ValueError(f"Open Chronicle MCP tool call failed: {error}")
        result = response_payload.get("result")
        if not isinstance(result, dict):
            raise ValueError("Open Chronicle MCP response is missing result object.")
        return result


class OpenChroniclePerceptionProvider(PerceptionProvider):
    provider_id = "open_chronicle"

    def __init__(
        self,
        *,
        mcp_url: str = DEFAULT_OPEN_CHRONICLE_MCP_URL,
        since_minutes: int = 15,
        context_limit: int = 5,
        timeout_seconds: int = 10,
        previous_hashes: dict[str, str] | None = None,
        now: datetime | None = None,
        client: OpenChronicleMCPClient | None = None,
    ) -> None:
        if since_minutes <= 0:
            raise ValueError("since_minutes must be positive.")
        if context_limit <= 0:
            raise ValueError("context_limit must be positive.")
        self.mcp_url = mcp_url.strip() or DEFAULT_OPEN_CHRONICLE_MCP_URL
        self.since_minutes = since_minutes
        self.context_limit = context_limit
        self.timeout_seconds = timeout_seconds
        self.previous_hashes = dict(previous_hashes or {})
        self.now = now or datetime.now(timezone.utc)
        self.client = client or OpenChronicleMCPClient(
            mcp_url=self.mcp_url,
            timeout_seconds=self.timeout_seconds,
        )

    def poll(self) -> list[GenericObservation]:
        return [
            result.observation
            for result in self.poll_results()
            if result.observation is not None
        ]

    def poll_results(self) -> list[OpenChronicleResult]:
        return [
            self._tool_result(
                tool_name="current_context",
                arguments={"limit": self.context_limit},
                signal_kind="open_chronicle_context_changed",
                evidence_kind="open_chronicle_current_context",
            ),
            self._tool_result(
                tool_name="recent_activity",
                arguments={
                    "since_minutes": self.since_minutes,
                    "limit": self.context_limit,
                },
                signal_kind="open_chronicle_activity_changed",
                evidence_kind="open_chronicle_recent_activity",
            ),
        ]

    def _tool_result(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        signal_kind: str,
        evidence_kind: str,
    ) -> OpenChronicleResult:
        payload = self.client.call_tool(tool_name, arguments)
        content = _content_from_mcp_result(payload)
        source_key = f"open_chronicle:{tool_name}"
        content_hash = _hash(content)
        previous_hash = self.previous_hashes.get(source_key, "")
        changed = content_hash != previous_hash
        observation = None
        metadata = {
            "provider": "open_chronicle",
            "mcp_url": self.mcp_url,
            "tool_name": tool_name,
            "arguments": arguments,
        }
        if changed:
            observation = _observation_from_open_chronicle(
                source_key=source_key,
                tool_name=tool_name,
                content=content,
                content_hash=content_hash,
                signal_kind=signal_kind,
                evidence_kind=evidence_kind,
                metadata=metadata,
                now=self.now,
            )
        return OpenChronicleResult(
            source_type="open_chronicle",
            source_value=tool_name,
            content=content,
            content_hash=content_hash,
            changed=changed,
            observation=observation,
            previous_hash=previous_hash,
            metadata=metadata,
        )


def _observation_from_open_chronicle(
    *,
    source_key: str,
    tool_name: str,
    content: str,
    content_hash: str,
    signal_kind: str,
    evidence_kind: str,
    metadata: dict[str, Any],
    now: datetime,
) -> GenericObservation:
    observed_at = _timestamp(now)
    slug = _hash([source_key, content_hash])[:16]
    subject_id = f"open_chronicle.{tool_name}"
    summary = f"Open Chronicle {tool_name} changed: {_preview(content)}"
    return GenericObservation(
        observation_id=f"obs.open_chronicle.{tool_name}.{slug}",
        kind=ObservationKind.SIGNAL,
        source=ObservationSource(
            provider="open_chronicle",
            channel="mcp",
            external_id=tool_name,
            received_at=observed_at,
            metadata={"source_key": source_key, "mcp_url": metadata.get("mcp_url", "")},
        ),
        subject=ObservationSubject(
            subject_id=subject_id,
            subject_type="desktop_context",
            title=f"Open Chronicle {tool_name}",
            refs=[source_key],
        ),
        summary=summary,
        attributes={
            "signal_id": f"signal.open_chronicle.{tool_name}.{slug}",
            "source": "open_chronicle",
            "kind": signal_kind,
            "observed_at": observed_at,
            "source_type": "mcp_tool",
            "source_value": tool_name,
            "source_key": source_key,
            "content_hash": content_hash,
            "output_preview": _preview(content, limit=500),
            "status": "active",
        },
        evidence=[
            ObservationEvidence(
                evidence_id=f"evidence.open_chronicle.{tool_name}.{slug}",
                kind=evidence_kind,
                summary=f"Open Chronicle MCP tool output: {tool_name}.",
                content=content,
                uri=str(metadata.get("mcp_url") or ""),
                metadata=dict(metadata),
            )
        ],
        confidence=ObservationConfidence(score=0.85, level="high"),
        refs=[source_key],
        metadata={
            "source": "open_chronicle",
            "source_type": "mcp_tool",
            "source_value": tool_name,
            "source_key": source_key,
            "content_hash": content_hash,
            **dict(metadata),
        },
    )


def _content_from_mcp_result(payload: dict[str, Any]) -> str:
    content = payload.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif "json" in item:
                parts.append(json.dumps(item["json"], ensure_ascii=True, sort_keys=True))
            elif "data" in item:
                parts.append(json.dumps(item["data"], ensure_ascii=True, sort_keys=True))
        if parts:
            return "\n".join(parts)
    if isinstance(payload.get("text"), str):
        return str(payload["text"])
    if isinstance(payload.get("data"), (dict, list)):
        return json.dumps(payload["data"], ensure_ascii=True, sort_keys=True)
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _preview(value: str, *, limit: int = 120) -> str:
    collapsed = " ".join(value.strip().split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(0, limit - 1)] + "..."


def _hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()
