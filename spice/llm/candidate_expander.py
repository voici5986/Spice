from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from spice.decision.general.candidates import (
    EstimatedCost,
    ExecutionBoundary,
    ExpectedStateDelta,
    GENERIC_ACTION_TYPES,
    GenericCandidate,
    GenericExecutionIntent,
    RiskProfile,
)
from spice.decision.general.state import GeneralDecisionState
from spice.decision.general.types import payload_value
from spice.llm.core import LLMClient, LLMModelConfig, LLMRequest, LLMRouter, LLMTaskHook, ProviderRegistry
from spice.llm.providers import (
    AnthropicLLMProvider,
    DeepSeekLLMProvider,
    DeterministicLLMProvider,
    MiMoLLMProvider,
    OpenAILLMProvider,
    OpenRouterLLMProvider,
    SubprocessLLMProvider,
)
from spice.llm.decision_proposal import LLMDecisionProposal, RUNTIME_CANDIDATE_FIELDS
from spice.llm.proposal_normalizer import normalize_decision_proposal
from spice.llm.util import (
    extract_first_json_array,
    extract_first_json_object,
    strip_markdown_fences,
)
from spice.language import detect_display_language, language_instruction


DECISION_KEYWORDS = (
    "choose",
    "pick",
    "compare",
    "decide",
    "decision",
    "prioritize",
    "priority",
    "rank",
    "which",
    "should i",
    "should we",
    "选",
    "选择",
    "比较",
    "决定",
    "优先",
    "哪个",
)

EXPLICIT_CHOICE_BLOCKED_ACTION_TYPES = frozenset(
    {
        "approval.request",
        "context.prepare",
        "state.observe_more",
        "time.defer",
        "user.clarify",
        "state.record",
    }
)


@dataclass(slots=True)
class LLMCandidateExpansionResult:
    enabled: bool
    status: str
    candidates: list[GenericCandidate] = field(default_factory=list)
    proposed_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    skipped_duplicate_count: int = 0
    model_provider: str = ""
    model_id: str = ""
    request_id: str = ""
    context_ref: str = ""
    context_type: str = ""
    error: str = ""
    raw_output: str = ""
    rejected: list[dict[str, Any]] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "status": self.status,
            "proposed_count": self.proposed_count,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "skipped_duplicate_count": self.skipped_duplicate_count,
            "model_provider": self.model_provider,
            "model_id": self.model_id,
            "request_id": self.request_id,
            "context_ref": self.context_ref,
            "context_type": self.context_type,
            "error": self.error,
            "raw_output": self.raw_output,
            "rejected": [dict(item) for item in self.rejected],
            "candidates": [candidate.to_payload() for candidate in self.candidates],
        }


def expand_candidates_from_runtime_config(
    *,
    config: dict[str, Any],
    state: GeneralDecisionState,
    intent_text: str,
    rule_candidates: list[GenericCandidate],
    display_language: str | None = None,
    explicit_options: list[str] | None = None,
    decision_context: Any | None = None,
) -> LLMCandidateExpansionResult:
    context_payload = _compiled_context_payload(decision_context)
    if not _truthy(config.get("llm_candidate_expand")):
        return LLMCandidateExpansionResult(
            enabled=False,
            status="disabled",
            context_ref=_context_ref(context_payload),
            context_type=_context_type(context_payload),
        )

    provider_id = str(config.get("llm_provider") or "deterministic").strip()
    model_id = _runtime_model_id(config)
    if not model_id:
        return LLMCandidateExpansionResult(
            enabled=True,
            status="fallback",
            model_provider=provider_id,
            context_ref=_context_ref(context_payload),
            context_type=_context_type(context_payload),
            error="llm_model is required when llm_candidate_expand=true.",
        )

    client = build_candidate_expander_client(provider_id=provider_id, model_id=model_id)
    try:
        return expand_candidates_with_llm(
            client=client,
            state=state,
            intent_text=intent_text,
            rule_candidates=rule_candidates,
            display_language=display_language or detect_display_language(intent_text),
            explicit_options=explicit_options,
            decision_context=decision_context,
            model_provider=provider_id,
            model_id=model_id,
        )
    except Exception as exc:
        return LLMCandidateExpansionResult(
            enabled=True,
            status="fallback",
            model_provider=provider_id,
            model_id=model_id,
            context_ref=_context_ref(context_payload),
            context_type=_context_type(context_payload),
            error=str(exc),
        )


def build_candidate_expander_client(*, provider_id: str, model_id: str) -> LLMClient:
    model = LLMModelConfig(
        provider_id=provider_id,
        model_id=model_id,
        temperature=0.2,
        max_tokens=1800,
        timeout_sec=45.0,
        response_format_hint="json_object",
    )
    registry = (
        ProviderRegistry.empty()
        .register(AnthropicLLMProvider())
        .register(DeepSeekLLMProvider())
        .register(DeterministicLLMProvider())
        .register(MiMoLLMProvider())
        .register(OpenAILLMProvider())
        .register(OpenRouterLLMProvider())
        .register(SubprocessLLMProvider())
    )
    router = LLMRouter(
        global_default=model,
        hook_defaults={LLMTaskHook.DECISION_PROPOSE: model},
    )
    return LLMClient(registry=registry, router=router)


def expand_candidates_with_llm(
    *,
    client: LLMClient,
    state: GeneralDecisionState,
    intent_text: str,
    rule_candidates: list[GenericCandidate],
    display_language: str = "en",
    explicit_options: list[str] | None = None,
    decision_context: Any | None = None,
    model_provider: str = "",
    model_id: str = "",
) -> LLMCandidateExpansionResult:
    resolved_explicit_options = explicit_options
    if resolved_explicit_options is None:
        resolved_explicit_options = extract_explicit_options(intent_text)
    response = client.generate(
        LLMRequest(
            task_hook=LLMTaskHook.DECISION_PROPOSE,
            input_text=_build_expansion_prompt(
                state=state,
                intent_text=intent_text,
                rule_candidates=rule_candidates,
                display_language=display_language,
                explicit_options=resolved_explicit_options,
                decision_context=decision_context,
            ),
            system_text=_system_prompt(display_language),
            response_format_hint="json_object",
            temperature=0.2,
            max_tokens=1800,
            metadata={
                "purpose": "generic_candidate_generation",
                "context_ref": _context_ref(_compiled_context_payload(decision_context)),
            },
        )
    )
    try:
        payload = _parse_expansion_payload(response.output_text)
    except Exception as exc:
        return LLMCandidateExpansionResult(
            enabled=True,
            status="fallback",
            model_provider=model_provider or response.provider_id,
            model_id=model_id or response.model_id,
            request_id=response.request_id,
            context_ref=_context_ref(_compiled_context_payload(decision_context)),
            context_type=_context_type(_compiled_context_payload(decision_context)),
            error=str(exc),
            raw_output=response.output_text,
        )
    raw_proposals = payload.get("decisions")
    payload_mode = "proposal"
    if not isinstance(raw_proposals, list):
        raw_proposals = payload.get("candidates")
        payload_mode = "legacy_candidate"
    if not isinstance(raw_proposals, list):
        return LLMCandidateExpansionResult(
            enabled=True,
            status="fallback",
            model_provider=model_provider or response.provider_id,
            model_id=model_id or response.model_id,
            request_id=response.request_id,
            context_ref=_context_ref(_compiled_context_payload(decision_context)),
            context_type=_context_type(_compiled_context_payload(decision_context)),
            error="missing decisions list",
            raw_output=response.output_text,
        )

    existing_ids = {candidate.candidate_id for candidate in rule_candidates}
    accepted: list[GenericCandidate] = []
    rejected: list[dict[str, Any]] = []
    skipped_duplicate_count = 0
    for index, raw_candidate in enumerate(raw_proposals):
        try:
            decision_mode = _decision_mode_for_intent(
                intent_text,
                explicit_options=resolved_explicit_options,
            )
            if payload_mode == "proposal":
                candidate = normalize_decision_proposal(
                    raw_candidate,
                    index=index,
                    decision_mode=decision_mode,
                    execution_requested_default=decision_mode == "execution_request",
                )
            else:
                candidate = _candidate_from_llm_payload(
                    raw_candidate,
                    index=index,
                    decision_mode=decision_mode,
                )
        except Exception as exc:
            rejected.append({"index": index, "reason": str(exc)})
            continue
        if resolved_explicit_options:
            explicit_choice_error = _explicit_choice_candidate_error(
                candidate,
                resolved_explicit_options,
            )
        else:
            explicit_choice_error = ""
        if explicit_choice_error:
            rejected.append(
                {
                    "index": index,
                    "candidate_id": candidate.candidate_id,
                    "reason": explicit_choice_error,
                }
            )
            continue
        if candidate.candidate_id in existing_ids:
            skipped_duplicate_count += 1
            continue
        existing_ids.add(candidate.candidate_id)
        accepted.append(candidate)

    return LLMCandidateExpansionResult(
        enabled=True,
        status="expanded" if accepted else "no_valid_candidates",
        candidates=accepted,
        proposed_count=len(raw_proposals),
        accepted_count=len(accepted),
        rejected_count=len(rejected),
        skipped_duplicate_count=skipped_duplicate_count,
        model_provider=model_provider or response.provider_id,
        model_id=model_id or response.model_id,
        request_id=response.request_id,
        context_ref=_context_ref(_compiled_context_payload(decision_context)),
        context_type=_context_type(_compiled_context_payload(decision_context)),
        rejected=rejected,
    )


def merge_expanded_candidates(
    rule_candidates: list[GenericCandidate],
    expansion: LLMCandidateExpansionResult,
) -> list[GenericCandidate]:
    merged = list(expansion.candidates)
    seen = {candidate.candidate_id for candidate in merged}
    for candidate in rule_candidates:
        if candidate.candidate_id in seen:
            continue
        seen.add(candidate.candidate_id)
        merged.append(candidate)
    return merged


def extract_explicit_options(intent_text: str) -> list[str]:
    """Extract short user-provided options from choice-shaped intents.

    This intentionally uses conservative heuristics rather than general NLP. The
    goal is to catch common "pick one of these" inputs without turning arbitrary
    long prose into candidates.
    """

    text = " ".join(str(intent_text or "").strip().split())
    if not text:
        return []

    lower = text.lower()
    has_decision_keyword = any(keyword in lower for keyword in DECISION_KEYWORDS)
    option_segment = _explicit_option_segment(text, has_decision_keyword=has_decision_keyword)
    raw_options = _split_option_segment(option_segment)
    options = _clean_options(raw_options)
    if not (2 <= len(options) <= 6):
        return []
    if has_decision_keyword:
        return options
    if ":" in text and len(options) >= 3 and all(_is_short_option(option) for option in options):
        return options
    return []


def _decision_mode_for_intent(intent_text: str, *, explicit_options: list[str]) -> str:
    if explicit_options:
        return "explicit_choice"
    text = str(intent_text or "").strip().lower()
    execution_markers = (
        "/act",
        "execute",
        "run ",
        "create ",
        "write ",
        "add ",
        "fix ",
        "implement ",
        "执行",
        "创建",
        "写入",
        "添加",
        "修复",
        "实现",
    )
    if any(marker in text for marker in execution_markers):
        return "execution_request"
    return "open_problem"


def build_explicit_option_candidates(
    *,
    intent_text: str,
    state: GeneralDecisionState,
    explicit_options: list[str] | None = None,
) -> list[GenericCandidate]:
    options = explicit_options if explicit_options is not None else extract_explicit_options(intent_text)
    if not options:
        return []

    target_refs = _active_intent_refs(state)
    candidates: list[GenericCandidate] = []
    for index, option in enumerate(options):
        execution_intent = GenericExecutionIntent(
            intent_class="advisory",
            requested=False,
            reason="The user asked Spice to compare an explicit option, not execute it.",
            side_effect_class="read_only",
        )
        candidate_id = _candidate_id(
            action_type="item.triage",
            intent=f"Choose explicit option: {option}",
            target_refs=[*target_refs, f"explicit_option.{index + 1}"],
            index=index,
        ).replace("candidate.llm.", "candidate.explicit_option.", 1)
        candidates.append(
            GenericCandidate(
                candidate_id=candidate_id,
                action_type="item.triage",
                intent=f"Choose explicit option: {option}",
                candidate_kind="decision",
                target_refs=target_refs,
                execution_intent=execution_intent,
                estimated_cost=EstimatedCost(time_minutes=10, attention="medium"),
                risk_profile=RiskProfile(
                    level="low",
                    summary="Low risk decision candidate derived from a user-provided option.",
                    uncertainty="medium",
                ),
                reversibility="high",
                requires_confirmation=False,
                expected_state_delta=ExpectedStateDelta(
                    summary=f"Selects '{option}' as the next decision candidate from the explicit option set.",
                ),
                execution_boundary=ExecutionBoundary(
                    mode="none",
                    target="",
                    protocol="",
                    requires_confirmation=False,
                    side_effect_class="read_only",
                    metadata={"source": "explicit_options"},
                ),
                why_available=[
                    "The user provided this as an explicit option to compare.",
                    "Explicit choice mode compares the provided options directly instead of inventing meta-actions.",
                ],
                side_effect_class="read_only",
                availability_status="available",
                metadata={
                    "source": "explicit_options",
                    "candidate_kind": "decision",
                    "candidate_source": "explicit_options",
                    "decision_mode": "explicit_choice",
                    "execution_intent": execution_intent.to_payload(),
                    "explicit_choice_option": True,
                    "explicit_option_index": index + 1,
                    "user_facing_title": option,
                    "recommended_action": f"Choose {option} as the first step.",
                    "why_now": [
                        "It is one of the concrete options the user asked Spice to compare."
                    ],
                    "expected_result": f"Spice selects whether '{option}' should be prioritized first.",
                    "executor_task": f"Use '{option}' as the selected next step if this decision is approved for execution.",
                },
            )
        )
    return candidates


def _candidate_from_llm_payload(
    payload: Any,
    *,
    index: int,
    decision_mode: str,
) -> GenericCandidate:
    if not isinstance(payload, dict):
        raise ValueError("candidate must be an object")
    action_type = str(payload.get("action_type") or "").strip()
    if action_type not in GENERIC_ACTION_TYPES:
        raise ValueError(f"unsupported action_type: {action_type!r}")
    intent = str(payload.get("intent") or payload.get("summary") or "").strip()
    if not intent:
        raise ValueError("intent must be non-empty")
    target_refs = _string_list(payload.get("target_refs"))
    required_capability = str(payload.get("required_capability") or "").strip()
    side_effect_class = _normalize_side_effect(payload.get("side_effect_class"), action_type)
    requires_confirmation = _requires_confirmation(payload, side_effect_class, action_type)
    why_available = _string_list(payload.get("why_available")) or [
        "LLM proposed this candidate from the current state."
    ]

    candidate_id = str(payload.get("candidate_id") or "").strip()
    if not candidate_id:
        candidate_id = _candidate_id(
            action_type=action_type,
            intent=intent,
            target_refs=target_refs,
            index=index,
        )
    if not candidate_id.startswith("candidate.llm."):
        candidate_id = f"candidate.llm.{_slug(candidate_id)}"
    metadata = _dict(payload.get("metadata"))
    explicit_option_index = _optional_int(
        payload.get("explicit_option_index", metadata.get("explicit_option_index"))
    )
    if explicit_option_index is not None:
        metadata["explicit_option_index"] = explicit_option_index

    candidate_kind = _candidate_kind(
        metadata.get("candidate_kind") or payload.get("candidate_kind") or "decision"
    )
    execution_intent = _execution_intent(
        payload.get("execution_intent", metadata.get("execution_intent"))
    )

    return GenericCandidate(
        candidate_id=candidate_id,
        action_type=action_type,
        intent=intent,
        candidate_kind=candidate_kind,
        target_refs=target_refs,
        required_capability=required_capability,
        execution_intent=execution_intent,
        estimated_cost=_estimated_cost(payload.get("estimated_cost")),
        risk_profile=_risk_profile(payload.get("risk_profile")),
        reversibility=str(payload.get("reversibility") or "unknown"),
        requires_confirmation=requires_confirmation,
        expected_state_delta=_expected_state_delta(payload.get("expected_state_delta")),
        execution_boundary=ExecutionBoundary(
            mode="llm_proposed",
            target=str(payload.get("execution_target") or ""),
            protocol="",
            required_capability=required_capability,
            requires_confirmation=requires_confirmation,
            side_effect_class=side_effect_class,
            metadata={"source": "llm_candidate_expander"},
        ),
        constraints_triggered=_dict_list(payload.get("constraints_triggered")),
        why_available=why_available,
        why_blocked=_string_list(payload.get("why_blocked")),
        side_effect_class=side_effect_class,
        availability_status=str(payload.get("availability_status") or "needs_confirmation")
        if requires_confirmation
        else str(payload.get("availability_status") or "available"),
        metadata={
            **metadata,
            "source": "llm_candidate_expander",
            "candidate_kind": candidate_kind,
            "candidate_source": "llm_generator",
            "decision_mode": str(metadata.get("decision_mode") or decision_mode),
            "llm_generated": True,
            "execution_intent": execution_intent.to_payload(),
            "user_facing_title": _first_string(
                payload.get("user_facing_title"),
                payload.get("title"),
                metadata.get("user_facing_title"),
            ),
            "recommended_action": _first_string(
                payload.get("recommended_action"),
                payload.get("recommendation"),
                metadata.get("recommended_action"),
                payload.get("intent"),
            ),
            "why_now": _string_list(payload.get("why_now")) or _string_list(metadata.get("why_now")),
            "expected_result": _first_string(
                payload.get("expected_result"),
                payload.get("expected_outcome"),
                metadata.get("expected_result"),
            ),
            "executor_task": _first_string(
                payload.get("executor_task"),
                payload.get("execution_objective"),
                metadata.get("executor_task"),
            ),
        },
    )


def _build_expansion_prompt(
    *,
    state: GeneralDecisionState,
    intent_text: str,
    rule_candidates: list[GenericCandidate],
    display_language: str,
    explicit_options: list[str] | None = None,
    decision_context: Any | None = None,
) -> str:
    options = list(explicit_options or [])
    decision_mode = _decision_mode_for_intent(intent_text, explicit_options=options)
    compiled_context = _compact_decision_context_for_prompt(
        _compiled_context_payload(decision_context)
    )
    indexed_options = [
        {"index": index + 1, "text": option}
        for index, option in enumerate(options)
    ]
    execution_requested_default = decision_mode == "execution_request"
    payload = {
        "user_intent": intent_text,
        "decision_mode": decision_mode,
        "display_language": display_language,
        "language_instruction": language_instruction(display_language),
        "explicit_choice_options": indexed_options,
        "context_usage": {
            "primary_context": "compiled_context",
            "legacy_fallback": "state_summary",
            "instruction": (
                "Use compiled_context as the source of truth for decision-relevant state. "
                "It contains current_intent, active_decision_frame, recent decisions, "
                "recent approvals, executor affordance, workspace context, and retrieved "
                "memory. Use state_summary only when compiled_context is empty or missing "
                "a specific field."
            ),
        },
        "decision_proposal_policy": {
            "output_contract": "lightweight_decision_proposals",
            "top_level_field": "decisions",
            "runtime_normalizes": True,
            "execution_requested_default": execution_requested_default,
            "instruction": (
                "Generate lightweight semantic decision proposals only. Spice runtime "
                "will convert each proposal into a GenericCandidate, assign action_type, "
                "candidate_kind, execution_intent, side effect class, approval boundary, "
                "permission, estimated cost, ids, and state deltas. Do not output those "
                "runtime fields. Set execution_requested=true only when the user explicitly "
                "asks Spice to execute, implement, create, write, modify files, run "
                "commands, or otherwise cross the executor boundary. For open_problem and "
                "explicit_choice, default to advisory."
            ),
            "forbidden_runtime_fields": sorted(RUNTIME_CANDIDATE_FIELDS),
        },
        "compiled_context": compiled_context,
        "explicit_choice_instruction": (
            "The user provided explicit options. Create proposals for these options only. "
            "Each returned proposal must include explicit_option_index matching one of "
            "explicit_choice_options. You may paraphrase option wording, but preserve option "
            "identity through explicit_option_index. "
            "Do not create meta-actions such as asking for preference, clarifying priorities, "
            "triaging the choice itself, preparing context, deferring, or recording state."
            if options
            else ""
        ),
        "state_summary": {
            "role": "legacy_fallback_only",
            "intents": [_compact_record(item.to_payload(), ("intent_id", "summary", "urgency", "target_refs")) for item in state.intents[:8]],
            "work_items": [_compact_record(item.to_payload(), ("work_item_id", "title", "urgency", "status", "blocker_refs")) for item in state.work_items[:8]],
            "capabilities": [_compact_record(item.to_payload(), ("capability_id", "provider", "scope", "requires_confirmation", "side_effects")) for item in state.capabilities[:8]],
            "constraints": [_compact_record(item.to_payload(), ("constraint_id", "kind", "description", "severity", "applies_to_refs")) for item in state.constraints[:8]],
            "risks": [_compact_record(item.to_payload(), ("risk_id", "kind", "description", "level", "applies_to_refs")) for item in state.risks[:8]],
        },
        "runtime_guardrails": [
            _compact_record(
                candidate.to_payload(),
                (
                    "candidate_id",
                    "action_type",
                    "intent",
                    "target_refs",
                    "required_capability",
                    "metadata",
                ),
            )
            for candidate in rule_candidates[:12]
        ],
        "proposal_schema": LLMDecisionProposal.response_schema(),
    }
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)


def _compiled_context_payload(context: Any | None) -> dict[str, Any]:
    if context is None:
        return {}
    payload = payload_value(context)
    return payload if isinstance(payload, dict) else {"value": payload}


def _compact_decision_context_for_prompt(context_payload: dict[str, Any]) -> dict[str, Any]:
    if not context_payload:
        return {}
    return {
        "id": context_payload.get("id", ""),
        "context_type": context_payload.get("context_type", ""),
        "schema_version": context_payload.get("schema_version", ""),
        "world_state_id": context_payload.get("world_state_id", ""),
        "domain": context_payload.get("domain", ""),
        "current_intent": _dict(context_payload.get("current_intent")),
        "active_decision_frame": _compact_active_decision_frame(
            _dict(context_payload.get("active_decision_frame"))
        ),
        "objectives": _take_dicts(context_payload.get("objectives"), 6),
        "constraints": _take_dicts(context_payload.get("constraints"), 8),
        "signals": _take_dicts(context_payload.get("signals"), 8),
        "risks": _take_dicts(context_payload.get("risks"), 8),
        "active_intents": _take_dicts(context_payload.get("active_intents"), 8),
        "recent_decisions": _take_dicts(context_payload.get("recent_decisions"), 6),
        "recent_approvals": _take_dicts(context_payload.get("recent_approvals"), 6),
        "recent_outcomes": _take_dicts(context_payload.get("recent_outcomes"), 6),
        "executor_affordance": _dict(context_payload.get("executor_affordance")),
        "session_summary": _dict(context_payload.get("session_summary")),
        "workspace_context": _dict(context_payload.get("workspace_context")),
        "retrieved_memory": _take_dicts(context_payload.get("retrieved_memory"), 8),
        "warnings": _string_list(context_payload.get("warnings")),
        "metadata": _dict(context_payload.get("metadata")),
    }


def _compact_active_decision_frame(frame: dict[str, Any]) -> dict[str, Any]:
    if not frame:
        return {}
    compact = {
        "decision_id": frame.get("decision_id", ""),
        "run_id": frame.get("run_id", ""),
        "status": frame.get("status", ""),
        "selected_candidate_id": frame.get("selected_candidate_id", ""),
        "approval_id": frame.get("approval_id", ""),
        "run_intent_mode": frame.get("run_intent_mode", ""),
        "display_language": frame.get("display_language", ""),
        "selection_pool": _dict(frame.get("selection_pool")),
        "handoff_blocked": frame.get("handoff_blocked", False),
        "handoff_blockers": payload_value(frame.get("handoff_blockers", [])),
    }
    candidates = frame.get("candidates")
    if isinstance(candidates, list):
        compact["candidates"] = [
            _compact_record(
                item,
                (
                    "label",
                    "candidate_id",
                    "title",
                    "action",
                    "recommended_action",
                    "expected_result",
                    "execution_affordance",
                ),
            )
            for item in candidates
            if isinstance(item, dict)
        ][:6]
    selected = frame.get("selected")
    if isinstance(selected, dict):
        compact["selected"] = _compact_record(
            selected,
            (
                "label",
                "candidate_id",
                "title",
                "action",
                "recommended_action",
                "expected_result",
                "execution_affordance",
            ),
        )
    return compact


def _take_dicts(value: Any, limit: int) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [payload_value(item) for item in value if isinstance(item, dict)][:limit]


def _context_ref(context_payload: dict[str, Any]) -> str:
    return str(context_payload.get("id") or "")


def _context_type(context_payload: dict[str, Any]) -> str:
    return str(context_payload.get("context_type") or "")


def _system_prompt(display_language: str = "en") -> str:
    return (
        "You are Spice's decision proposal generator. Return only a JSON object "
        "with top-level key 'decisions'. "
        "Do not select a winner. Do not execute. "
        "Use compiled_context as the primary decision-relevant context. Consult "
        "current_intent, active_decision_frame, recent_decisions, recent_approvals, "
        "executor_affordance, workspace_context, and retrieved_memory before using "
        "legacy state_summary. "
        "Generate 2-3 concrete decision proposals that directly address the user intent "
        "and current state. A proposal is something the user could choose as the next "
        "direction, not a generic runtime step. "
        "Return only lightweight proposal fields: title, recommendation, why_now, "
        "expected_result, downside, success_signal, confidence, risk_level, "
        "explicit_option_index, execution_requested, and handoff_task. "
        "Do not output runtime fields such as candidate_id, action_type, candidate_kind, "
        "execution_intent, side_effect_class, requires_confirmation, required_permission_hint, "
        "estimated_cost, expected_state_delta, target_refs, or required_capability. "
        "Spice runtime will normalize proposals into executable or advisory candidates. "
        "Set execution_requested=true only when the user explicitly asks to execute, "
        "implement, create, write, modify files, run commands, or cross an executor boundary. "
        "Questions, comparisons, prioritization, strategy, and planning should be advisory. "
        "If explicit_choice_options is non-empty, create proposals for those user-provided "
        "options only and include explicit_option_index on every proposal. Do not create "
        "meta-actions such as clarify, ask preference, triage "
        "the choice itself, prepare context, defer, or record state unless every explicit "
        "option is unsafe or impossible. "
        "If decision_mode is open_problem, infer the decision frame and propose concrete "
        "strategic or tactical advisory options. "
        "If decision_mode is execution_request, propose options that can become bounded "
        "executor tasks and mark only those proposals as execution_requested. "
        "Treat runtime_guardrails as fallback runtime actions, not as the primary proposal "
        "space to extend. "
        "Avoid meta-actions such as gather more information, prepare context, ask for "
        "clarification, defer, or record state unless the user intent is impossible to act "
        "on without missing critical information. "
        "Write title, recommendation, why_now, expected_result, downside, and "
        "success_signal in concrete user language. "
        + language_instruction(display_language)
    )


def _parse_expansion_payload(text: str) -> dict[str, Any]:
    stripped = strip_markdown_fences(str(text or ""))
    payload = _load_json_payload(stripped)
    if isinstance(payload, list):
        payload = {"decisions": payload}
    if not isinstance(payload, dict):
        raise ValueError("LLM candidate expansion response must be a JSON object or list.")
    repaired = _repair_expansion_payload(payload)
    if not isinstance(repaired.get("decisions"), list) and not isinstance(
        repaired.get("candidates"), list
    ):
        raise ValueError("missing decisions list")
    return repaired


def _load_json_payload(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    extracted_object = extract_first_json_object(text)
    if extracted_object:
        return json.loads(extracted_object)
    extracted_array = extract_first_json_array(text)
    if extracted_array:
        return json.loads(extracted_array)
    raise ValueError("LLM candidate expansion response did not contain valid JSON.")


def _repair_expansion_payload(payload: dict[str, Any]) -> dict[str, Any]:
    repaired = dict(payload)
    list_key = _first_list_key(
        repaired,
        ("decisions", "candidates", "options", "recommendations", "items"),
    )
    if not list_key:
        return repaired
    items = repaired.get(list_key)
    if not isinstance(items, list):
        return repaired
    normalized_items = [
        _repair_decision_item(item, index=index)
        for index, item in enumerate(items)
    ]
    if list_key == "candidates" and _looks_like_legacy_candidates(normalized_items):
        repaired["candidates"] = normalized_items
        return repaired
    repaired["decisions"] = normalized_items
    if list_key != "decisions":
        repaired.pop(list_key, None)
    return repaired


def _first_list_key(payload: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if isinstance(payload.get(key), list):
            return key
    return ""


def _looks_like_legacy_candidates(items: list[Any]) -> bool:
    return any(
        isinstance(item, dict)
        and (
            "action_type" in item
            or "intent" in item
            or "execution_intent" in item
            or "candidate_kind" in item
        )
        for item in items
    )


def _repair_decision_item(item: Any, *, index: int) -> dict[str, Any]:
    if isinstance(item, str):
        text = item.strip()
        return {
            "title": text or f"Option {index + 1}",
            "recommendation": text or f"Consider option {index + 1}.",
            "why_now": [],
            "expected_result": "",
            "downside": "",
            "success_signal": "",
            "risk_level": "unknown",
            "execution_requested": False,
        }
    if not isinstance(item, dict):
        return {
            "title": f"Option {index + 1}",
            "recommendation": str(item or "").strip() or f"Consider option {index + 1}.",
            "why_now": [],
            "expected_result": "",
            "downside": "",
            "success_signal": "",
            "risk_level": "unknown",
            "execution_requested": False,
        }
    repaired = dict(item)
    _copy_alias(repaired, "why", "why_now")
    _copy_alias(repaired, "expected_outcome", "expected_result")
    _copy_alias(repaired, "next_step", "recommendation")
    if "risk_level" not in repaired:
        if "risk" in repaired:
            repaired["risk_level"] = _repair_risk_level(repaired.get("risk"))
        elif "risks" in repaired:
            repaired["risk_level"] = _repair_risk_level(repaired.get("risks"))
    repaired["title"] = _first_string(
        repaired.get("title"),
        repaired.get("name"),
        repaired.get("option"),
        repaired.get("summary"),
        repaired.get("recommendation"),
        f"Option {index + 1}",
    )
    repaired["recommendation"] = _first_string(
        repaired.get("recommendation"),
        repaired.get("next_step"),
        repaired.get("action"),
        repaired.get("title"),
        f"Consider {repaired['title']}.",
    )
    repaired["why_now"] = _repair_string_list(repaired.get("why_now"))
    repaired["expected_result"] = str(repaired.get("expected_result") or "").strip()
    repaired["downside"] = str(
        repaired.get("downside")
        or repaired.get("tradeoff")
        or repaired.get("trade_off")
        or ""
    ).strip()
    repaired["success_signal"] = str(
        repaired.get("success_signal") or repaired.get("success_metric") or ""
    ).strip()
    repaired["risk_level"] = _repair_risk_level(repaired.get("risk_level"))
    repaired["execution_requested"] = _truthy(repaired.get("execution_requested"))
    repaired["handoff_task"] = str(repaired.get("handoff_task") or "").strip()
    if "explicit_option_index" in repaired:
        explicit_option_index = _optional_int(repaired.get("explicit_option_index"))
        if explicit_option_index is None:
            repaired.pop("explicit_option_index", None)
        else:
            repaired["explicit_option_index"] = explicit_option_index
    return repaired


def _copy_alias(payload: dict[str, Any], source: str, target: str) -> None:
    if target not in payload and source in payload:
        payload[target] = payload[source]


def _repair_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _repair_risk_level(value: Any) -> str:
    if isinstance(value, list):
        text = " ".join(str(item or "") for item in value).lower()
    else:
        text = str(value or "").strip().lower()
    if text in {"low", "medium", "high", "unknown"}:
        return text
    for level in ("high", "medium", "low"):
        if level in text:
            return level
    return "unknown"


def _explicit_option_segment(text: str, *, has_decision_keyword: bool) -> str:
    if ":" in text and has_decision_keyword:
        segment = text.split(":", 1)[1].strip()
    else:
        segment = text
    for marker in (
        ". Which ",
        ". which ",
        ". Pick ",
        ". pick ",
        ". Choose ",
        ". choose ",
        "?",
    ):
        if marker in segment:
            segment = segment.split(marker, 1)[0].strip()
    return segment


def _split_option_segment(segment: str) -> list[str]:
    normalized = re.sub(r"\s+(?:or|或者|还是)\s+", ", ", segment, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+(?:and|以及|和)\s+", ", ", normalized, flags=re.IGNORECASE)
    return re.split(r"[,;\n]+", normalized)


def _clean_options(raw_options: list[str]) -> list[str]:
    options: list[str] = []
    seen: set[str] = set()
    for raw_option in raw_options:
        option = re.sub(r"^\s*(?:[-*]|\d+[.)]|[A-Z][.)])\s*", "", raw_option).strip()
        option = option.strip(" .,:;")
        option = re.sub(r"\s+", " ", option)
        if not _looks_like_explicit_option(option):
            continue
        key = option.lower()
        if key in seen:
            continue
        seen.add(key)
        options.append(option)
    return options


def _looks_like_explicit_option(option: str) -> bool:
    if not option:
        return False
    if len(option) > 96:
        return False
    if not _is_short_option(option):
        return False
    lower = option.lower()
    if lower.startswith(("compare ", "choose ", "pick ", "which ", "should ")):
        return False
    if lower in {"time", "risk", "cost", "impact", "urgency", "priority"}:
        return False
    return any(char.isalpha() for char in option)


def _is_short_option(option: str) -> bool:
    return len(option.split()) < 15


def _active_intent_refs(state: GeneralDecisionState) -> list[str]:
    active = [item.intent_id for item in state.intents if item.status == "active"]
    if active:
        return [active[-1]]
    if state.intents:
        return [state.intents[-1].intent_id]
    return []


def _explicit_choice_candidate_error(
    candidate: GenericCandidate,
    explicit_options: list[str],
) -> str:
    if candidate.action_type in EXPLICIT_CHOICE_BLOCKED_ACTION_TYPES:
        return (
            "explicit-choice mode rejected meta action_type "
            f"{candidate.action_type!r}; expected a candidate tied to a user-provided option"
        )
    explicit_option_index = _optional_int(candidate.metadata.get("explicit_option_index"))
    if explicit_option_index is None:
        return "explicit-choice candidate missing explicit_option_index"
    if explicit_option_index < 1 or explicit_option_index > len(explicit_options):
        return (
            "explicit-choice candidate explicit_option_index out of range "
            f"(got {explicit_option_index}, expected 1..{len(explicit_options)})"
        )
    return ""


def _runtime_model_id(config: dict[str, Any]) -> str:
    configured = str(config.get("llm_model") or "").strip()
    if configured:
        return configured
    provider_id = str(config.get("llm_provider") or "deterministic").strip()
    if provider_id == "deterministic":
        return "deterministic.v1"
    return ""


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _candidate_kind(value: Any) -> str:
    token = str(value or "decision").strip()
    if token in {"decision", "runtime_action", "execution_handoff"}:
        return token
    return "decision"


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _estimated_cost(payload: Any) -> EstimatedCost:
    data = _dict(payload)
    minutes = data.get("time_minutes")
    return EstimatedCost(
        time_minutes=minutes if isinstance(minutes, int) else None,
        attention=str(data.get("attention") or "unknown"),
        money=data.get("money") if isinstance(data.get("money"), (int, float)) else None,
        notes=_string_list(data.get("notes")),
    )


def _risk_profile(payload: Any) -> RiskProfile:
    data = _dict(payload)
    level = str(data.get("level") or "unknown")
    if level not in {"low", "medium", "high", "unknown", "none"}:
        level = "unknown"
    return RiskProfile(
        level=level,
        risk_refs=_string_list(data.get("risk_refs")),
        summary=str(data.get("summary") or ""),
        uncertainty=str(data.get("uncertainty") or "unknown"),
    )


def _expected_state_delta(payload: Any) -> ExpectedStateDelta:
    data = _dict(payload)
    return ExpectedStateDelta(
        creates_refs=_string_list(data.get("creates_refs")),
        updates_refs=_string_list(data.get("updates_refs")),
        closes_refs=_string_list(data.get("closes_refs")),
        summary=str(data.get("summary") or ""),
    )


def _execution_intent(payload: Any) -> GenericExecutionIntent:
    data = _dict(payload)
    if not data:
        return GenericExecutionIntent()
    known_keys = {
        "intent_class",
        "execution_mode",
        "intent_type",
        "mode",
        "requested",
        "needs_execution",
        "handoff_task",
        "executor_task",
        "execution_objective",
        "requested_action",
        "reason",
        "required_permission_hint",
        "side_effect_class",
        "side_effect",
    }
    requested = _truthy(data.get("requested", data.get("needs_execution", False)))
    intent_class = _execution_intent_class(data, requested=requested)
    return GenericExecutionIntent(
        intent_class=intent_class,
        requested=requested,
        handoff_task=_first_string(
            data.get("handoff_task"),
            data.get("executor_task"),
            data.get("execution_objective"),
            data.get("requested_action"),
        ),
        reason=str(data.get("reason") or ""),
        required_permission_hint=str(data.get("required_permission_hint") or "unknown"),
        side_effect_class=str(data.get("side_effect_class") or data.get("side_effect") or "none"),
        metadata={key: value for key, value in data.items() if key not in known_keys},
    )


def _execution_intent_class(data: dict[str, Any], *, requested: bool) -> str:
    value = str(
        data.get("intent_class")
        or data.get("execution_mode")
        or data.get("intent_type")
        or data.get("mode")
        or ""
    ).strip()
    if value in {"advisory", "execution_requested"}:
        return value
    return "execution_requested" if requested else "advisory"


def _requires_confirmation(payload: dict[str, Any], side_effect_class: str, action_type: str) -> bool:
    if side_effect_class in {"state_change", "external_effect"}:
        return True
    if action_type in {"intent.execute", "capability.use", "artifact.draft", "approval.request"}:
        return True
    value = payload.get("requires_confirmation")
    if isinstance(value, bool):
        return value
    return False


def _normalize_side_effect(value: Any, action_type: str) -> str:
    token = str(value or "").strip().lower()
    aliases = {
        "external": "external_effect",
        "execute": "external_effect",
        "write": "external_effect",
        "send": "external_effect",
        "draft": "state_change",
        "low": "state_change",
        "read_or_prepare": "read_only",
        "none": "none",
        "": "",
    }
    token = aliases.get(token, token)
    if token in {"none", "read_only", "state_change", "external_effect"}:
        return token
    if action_type in {"intent.execute", "capability.use"}:
        return "external_effect"
    if action_type in {"artifact.draft", "approval.request", "state.record", "task.split", "item.triage"}:
        return "state_change"
    return "read_only"


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


def _compact_record(payload: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: payload_value(payload.get(key)) for key in keys if payload.get(key) not in (None, "", [])}


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _first_string(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return cleaned or "candidate"
