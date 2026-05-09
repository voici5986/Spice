from __future__ import annotations

import json
import re
from hashlib import sha256
from typing import Any

from spice.decision.general.candidates import (
    EstimatedCost,
    ExecutionBoundary,
    ExpectedStateDelta,
    GenericCandidate,
    GenericExecutionIntent,
    RiskProfile,
)
from spice.llm.decision_proposal import LLMDecisionProposal


def normalize_decision_proposal(
    payload: Any,
    *,
    index: int,
    decision_mode: str,
    execution_requested_default: bool = False,
) -> GenericCandidate:
    """Convert a lightweight LLM proposal into a runtime-owned candidate.

    LLMs provide semantic proposal content only. This normalizer owns the stable
    runtime fields: candidate id, action type, execution intent, side effects,
    approval boundary, and candidate metadata.
    """

    proposal = (
        payload
        if isinstance(payload, LLMDecisionProposal)
        else LLMDecisionProposal.from_payload(payload)
    )
    execution_requested = bool(proposal.execution_requested)
    if execution_requested_default and proposal.handoff_task:
        execution_requested = True

    action_type = "intent.execute" if execution_requested else "item.triage"
    handoff_task = proposal.handoff_task.strip()
    if execution_requested and not handoff_task:
        handoff_task = proposal.recommendation or proposal.title

    side_effect_class = "external_effect" if execution_requested else "read_only"
    execution_intent = GenericExecutionIntent(
        intent_class="execution_requested" if execution_requested else "advisory",
        requested=execution_requested,
        handoff_task=handoff_task if execution_requested else "",
        reason=(
            "The user explicitly requested executor handoff."
            if execution_requested
            else "Advisory decision proposal; no executor handoff requested."
        ),
        required_permission_hint="workspace_write" if execution_requested else "read_only",
        side_effect_class=side_effect_class,
        metadata={
            "source": "llm_decision_proposal",
            "decision_mode": decision_mode,
        },
    )
    target_refs: list[str] = []
    if proposal.explicit_option_index is not None:
        target_refs.append(f"explicit_option.{proposal.explicit_option_index}")
    candidate_id = _candidate_id(
        action_type=action_type,
        intent=proposal.title,
        target_refs=target_refs,
        index=index,
    )
    confidence_note = (
        f"confidence={proposal.confidence:.2f}"
        if proposal.confidence is not None
        else ""
    )
    why_available = [
        "LLM proposed this semantic decision from the current intent and compiled context."
    ]
    if proposal.explicit_option_index is not None:
        why_available.append(
            f"It corresponds to explicit option {proposal.explicit_option_index}."
        )

    metadata = {
        "source": "llm_candidate_expander",
        "candidate_kind": "decision",
        "candidate_source": "llm_generator",
        "decision_mode": decision_mode,
        "llm_generated": True,
        "llm_payload_mode": "decision_proposal",
        "execution_intent": execution_intent.to_payload(),
        "user_facing_title": proposal.title,
        "recommended_action": proposal.recommendation,
        "why_now": proposal.why_now,
        "expected_result": proposal.expected_result,
        "downside": proposal.downside,
        "success_signal": proposal.success_signal,
        "confidence": proposal.confidence,
        "risk_level": proposal.risk_level,
        "executor_task": handoff_task,
    }
    if proposal.explicit_option_index is not None:
        metadata["explicit_option_index"] = proposal.explicit_option_index

    return GenericCandidate(
        candidate_id=candidate_id,
        action_type=action_type,
        intent=proposal.title,
        candidate_kind="decision",
        target_refs=target_refs,
        execution_intent=execution_intent,
        estimated_cost=EstimatedCost(attention="unknown"),
        risk_profile=RiskProfile(
            level=proposal.risk_level,
            summary=proposal.downside,
            uncertainty=confidence_note or "unknown",
        ),
        reversibility="unknown",
        requires_confirmation=execution_requested,
        expected_state_delta=ExpectedStateDelta(summary=proposal.expected_result),
        execution_boundary=ExecutionBoundary(
            mode="execution_intent" if execution_requested else "none",
            target=handoff_task if execution_requested else "",
            protocol="sdep" if execution_requested else "",
            requires_confirmation=execution_requested,
            side_effect_class=side_effect_class,
            metadata={"source": "llm_decision_proposal"},
        ),
        why_available=why_available,
        side_effect_class=side_effect_class,
        availability_status="needs_confirmation" if execution_requested else "available",
        metadata=metadata,
    )


def _candidate_id(
    *,
    action_type: str,
    intent: str,
    target_refs: list[str],
    index: int,
) -> str:
    seed = json.dumps(
        {
            "action_type": action_type,
            "intent": intent,
            "target_refs": target_refs,
            "index": index,
        },
        ensure_ascii=True,
        sort_keys=True,
    )
    digest = sha256(seed.encode("utf-8")).hexdigest()[:12]
    return f"candidate.llm.{_slug(action_type)}.{digest}"


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return cleaned or "candidate"
