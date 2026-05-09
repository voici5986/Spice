from __future__ import annotations

import json
import shlex
import subprocess
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


MAX_POLL_CONTENT_BYTES = 16 * 1024


@dataclass(frozen=True, slots=True)
class PollResult:
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


class PollPerceptionProvider(PerceptionProvider):
    provider_id = "poll"

    def __init__(
        self,
        *,
        url: str = "",
        command: str | list[str] = "",
        allow_command_poll: bool = False,
        timeout_seconds: int = 10,
        previous_hashes: dict[str, str] | None = None,
        now: datetime | None = None,
    ) -> None:
        self.url = url.strip()
        self.command = command
        self.allow_command_poll = allow_command_poll
        self.timeout_seconds = timeout_seconds
        self.previous_hashes = dict(previous_hashes or {})
        self.now = now or datetime.now(timezone.utc)

    def poll(self) -> list[GenericObservation]:
        return [
            result.observation
            for result in self.poll_results()
            if result.observation is not None
        ]

    def poll_results(self) -> list[PollResult]:
        results: list[PollResult] = []
        if self.url:
            results.append(self._poll_url(self.url))
        if self.command:
            if not self.allow_command_poll:
                raise PermissionError(
                    "Command poll is disabled. Pass --allow-command-poll or set perception_allow_command_poll=true."
                )
            results.append(self._poll_command(self.command))
        if not results:
            raise ValueError("Poll provider requires a poll URL or poll command.")
        return results

    def _poll_url(self, url: str) -> PollResult:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        with urllib.request.urlopen(url, timeout=self.timeout_seconds) as response:
            content_bytes = response.read(MAX_POLL_CONTENT_BYTES + 1)
        truncated = len(content_bytes) > MAX_POLL_CONTENT_BYTES
        if truncated:
            content_bytes = content_bytes[:MAX_POLL_CONTENT_BYTES]
        content = content_bytes.decode("utf-8", errors="replace")
        return self._result(
            source_type="url",
            source_value=url,
            content=content,
            signal_kind="url_changed",
            evidence_kind="poll_url_response",
            metadata={"truncated": truncated},
        )

    def _poll_command(self, command: str | list[str]) -> PollResult:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        argv = shlex.split(command) if isinstance(command, str) else [str(item) for item in command]
        if not argv:
            raise ValueError("Poll command must be non-empty.")
        try:
            completed = subprocess.run(
                argv,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_seconds,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"Poll command timed out after {self.timeout_seconds} seconds."
            ) from exc
        output = completed.stdout
        if completed.stderr:
            output = f"{output}\n[stderr]\n{completed.stderr}"
        truncated = len(output.encode("utf-8")) > MAX_POLL_CONTENT_BYTES
        if truncated:
            output = output.encode("utf-8")[:MAX_POLL_CONTENT_BYTES].decode(
                "utf-8",
                errors="replace",
            )
        return self._result(
            source_type="command",
            source_value=" ".join(shlex.quote(item) for item in argv),
            content=output,
            signal_kind="command_changed",
            evidence_kind="poll_command_output",
            metadata={
                "exit_code": completed.returncode,
                "truncated": truncated,
            },
        )

    def _result(
        self,
        *,
        source_type: str,
        source_value: str,
        content: str,
        signal_kind: str,
        evidence_kind: str,
        metadata: dict[str, Any],
    ) -> PollResult:
        source_key = f"{source_type}:{source_value}"
        content_hash = _hash(content)
        previous_hash = self.previous_hashes.get(source_key, "")
        changed = content_hash != previous_hash
        observation = None
        if changed:
            observation = _observation_from_poll(
                source_type=source_type,
                source_value=source_value,
                source_key=source_key,
                content=content,
                content_hash=content_hash,
                signal_kind=signal_kind,
                evidence_kind=evidence_kind,
                metadata=metadata,
                now=self.now,
            )
        return PollResult(
            source_type=source_type,
            source_value=source_value,
            content=content,
            content_hash=content_hash,
            changed=changed,
            observation=observation,
            previous_hash=previous_hash,
            metadata=metadata,
        )


def _observation_from_poll(
    *,
    source_type: str,
    source_value: str,
    source_key: str,
    content: str,
    content_hash: str,
    signal_kind: str,
    evidence_kind: str,
    metadata: dict[str, Any],
    now: datetime,
) -> GenericObservation:
    observed_at = _timestamp(now)
    slug = _hash([source_key, content_hash])[:16]
    subject_id = f"poll.{source_type}.{_safe_slug(source_value)[:48]}"
    summary = f"Poll {source_type} changed: {_preview(content)}"
    return GenericObservation(
        observation_id=f"obs.poll.{source_type}.{slug}",
        kind=ObservationKind.SIGNAL,
        source=ObservationSource(
            provider="poll",
            channel=source_type,
            external_id=source_value,
            received_at=observed_at,
            metadata={"source_key": source_key},
        ),
        subject=ObservationSubject(
            subject_id=subject_id,
            subject_type=f"poll_{source_type}",
            title=source_value[:120],
            refs=[source_key],
        ),
        summary=summary,
        attributes={
            "signal_id": f"signal.poll.{source_type}.{slug}",
            "source": "poll",
            "kind": signal_kind,
            "observed_at": observed_at,
            "source_type": source_type,
            "source_value": source_value,
            "source_key": source_key,
            "content_hash": content_hash,
            "output_preview": _preview(content, limit=500),
            "status": "active",
        },
        evidence=[
            ObservationEvidence(
                evidence_id=f"evidence.poll.{source_type}.{slug}",
                kind=evidence_kind,
                summary=f"Poll {source_type} output.",
                content=content,
                uri=source_value if source_type == "url" else "",
                metadata=dict(metadata),
            )
        ],
        confidence=ObservationConfidence(score=0.85, level="high"),
        refs=[source_key],
        metadata={
            "source": "poll",
            "source_type": source_type,
            "source_value": source_value,
            "source_key": source_key,
            "content_hash": content_hash,
            **dict(metadata),
        },
    )


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _preview(value: str, *, limit: int = 120) -> str:
    collapsed = " ".join(value.strip().split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(0, limit - 1)] + "..."


def _hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _safe_slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_") or "unknown"
