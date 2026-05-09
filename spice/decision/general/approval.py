from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from spice.decision.general.types import PayloadRecord


@dataclass(slots=True)
class Approval(PayloadRecord):
    approval_id: str
    decision_id: str
    candidate_id: str = ""
    status: str = "pending"
    mode: str = "confirm_before_execution"
    requested_at: str = ""
    resolved_at: str = ""
    actor: str = "user"
    prompt: str = ""
    response: str = ""
    reason: str = ""
    execution_allowed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
