from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from spice.decision.general.types import PayloadRecord


PROPOSAL_RISK_LEVELS = frozenset({"low", "medium", "high", "unknown"})

LLM_DECISION_PROPOSAL_FIELDS = (
    "title",
    "recommendation",
    "why_now",
    "expected_result",
    "downside",
    "success_signal",
    "confidence",
    "risk_level",
    "explicit_option_index",
    "execution_requested",
    "handoff_task",
)

RUNTIME_CANDIDATE_FIELDS = frozenset(
    {
        "candidate_id",
        "action_type",
        "candidate_kind",
        "decision_mode",
        "execution_intent",
        "side_effect_class",
        "requires_confirmation",
        "required_permission_hint",
        "estimated_cost",
        "expected_state_delta",
        "target_refs",
        "required_capability",
        "execution_boundary",
        "skill_resolution",
        "execution_affordance",
    }
)


@dataclass(slots=True)
class LLMDecisionProposal(PayloadRecord):
    """A lightweight semantic proposal from an LLM.

    This is intentionally not a GenericCandidate. The LLM supplies only the
    decision content that benefits from semantic reasoning; Spice runtime fills
    candidate ids, action types, permissions, approval, skills, and execution
    affordance deterministically in later pipeline stages.
    """

    title: str
    recommendation: str
    why_now: list[str] = field(default_factory=list)
    expected_result: str = ""
    downside: str = ""
    success_signal: str = ""
    confidence: float | None = None
    risk_level: str = "unknown"
    explicit_option_index: int | None = None
    execution_requested: bool = False
    handoff_task: str = ""

    @classmethod
    def from_payload(cls, payload: Any) -> "LLMDecisionProposal":
        if not isinstance(payload, dict):
            raise ValueError("LLMDecisionProposal payload must be a dict.")
        title = _string(payload.get("title"))
        recommendation = _string(payload.get("recommendation"))
        if not title:
            raise ValueError("LLMDecisionProposal.title is required.")
        if not recommendation:
            raise ValueError("LLMDecisionProposal.recommendation is required.")
        return cls(
            title=title,
            recommendation=recommendation,
            why_now=_string_list(payload.get("why_now")),
            expected_result=_string(payload.get("expected_result")),
            downside=_string(payload.get("downside")),
            success_signal=_string(payload.get("success_signal")),
            confidence=_confidence(payload.get("confidence")),
            risk_level=_risk_level(payload.get("risk_level")),
            explicit_option_index=_positive_int_or_none(
                payload.get("explicit_option_index")
            ),
            execution_requested=_bool(payload.get("execution_requested")),
            handoff_task=_string(payload.get("handoff_task")),
        )

    @classmethod
    def response_schema(cls) -> dict[str, Any]:
        return {
            "decisions": [
                {
                    "title": "specific user-facing decision option",
                    "recommendation": "concrete recommendation for this option",
                    "why_now": ["why this option matters now"],
                    "expected_result": "likely outcome if chosen",
                    "downside": "main trade-off or downside",
                    "success_signal": "observable sign that this worked",
                    "confidence": 0.7,
                    "risk_level": "low|medium|high|unknown",
                    "explicit_option_index": "integer when the user provided explicit options",
                    "execution_requested": False,
                    "handoff_task": "only when the user explicitly requested execution",
                }
            ]
        }


def _string(value: Any) -> str:
    return str(value or "").strip()


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [item for item in (_string(item) for item in value) if item]
    text = _string(value)
    return [text] if text else []


def _confidence(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, parsed))


def _risk_level(value: Any) -> str:
    normalized = _string(value).lower()
    return normalized if normalized in PROPOSAL_RISK_LEVELS else "unknown"


def _positive_int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)
