from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from spice.decision.general.candidates import GenericCandidate
from spice.decision.general.state import GeneralDecisionState
from spice.decision.general.types import payload_value
from spice.llm.candidate_expander import build_candidate_expander_client
from spice.llm.core import LLMClient, LLMRequest, LLMTaskHook
from spice.llm.util import extract_first_json_object
from spice.language import detect_display_language, language_instruction


SIMULATION_OUTPUT_MALFORMED_ERROR = "simulation output appears truncated or malformed"


@dataclass(slots=True)
class LLMSimulationResult:
    enabled: bool
    status: str
    candidates: list[GenericCandidate] = field(default_factory=list)
    proposed_count: int = 0
    applied_count: int = 0
    rejected_count: int = 0
    model_provider: str = ""
    model_id: str = ""
    request_id: str = ""
    context_ref: str = ""
    context_type: str = ""
    error: str = ""
    raw_output: str = ""
    rejected: list[dict[str, Any]] = field(default_factory=list)
    simulation_target_ids: list[str] = field(default_factory=list)
    simulation_target_count: int = 0
    matched_simulation_ids: list[str] = field(default_factory=list)
    unmatched_simulation_ids: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "status": self.status,
            "proposed_count": self.proposed_count,
            "applied_count": self.applied_count,
            "rejected_count": self.rejected_count,
            "model_provider": self.model_provider,
            "model_id": self.model_id,
            "request_id": self.request_id,
            "context_ref": self.context_ref,
            "context_type": self.context_type,
            "error": self.error,
            "raw_output": self.raw_output,
            "rejected": [dict(item) for item in self.rejected],
            "simulation_target_ids": list(self.simulation_target_ids),
            "simulation_target_count": self.simulation_target_count,
            "matched_simulation_ids": list(self.matched_simulation_ids),
            "unmatched_simulation_ids": list(self.unmatched_simulation_ids),
            "simulations": [
                dict(candidate.metadata.get("llm_simulation"))
                for candidate in self.candidates
                if isinstance(candidate.metadata.get("llm_simulation"), dict)
            ],
        }


def simulate_candidates_from_runtime_config(
    *,
    config: dict[str, Any],
    state: GeneralDecisionState,
    intent_text: str,
    candidates: list[GenericCandidate],
    display_language: str | None = None,
    simulation_context: Any | None = None,
) -> LLMSimulationResult:
    context_payload = _compiled_context_payload(simulation_context)
    if not _truthy(config.get("llm_simulation")):
        return LLMSimulationResult(
            enabled=False,
            status="disabled",
            candidates=list(candidates),
            context_ref=_context_ref(context_payload),
            context_type=_context_type(context_payload),
            simulation_target_ids=_candidate_ids(candidates),
            simulation_target_count=len(candidates),
        )

    provider_id = str(config.get("llm_provider") or "deterministic").strip()
    model_id = _runtime_model_id(config)
    if not model_id:
        return LLMSimulationResult(
            enabled=True,
            status="fallback",
            candidates=list(candidates),
            model_provider=provider_id,
            context_ref=_context_ref(context_payload),
            context_type=_context_type(context_payload),
            error="llm_model is required when llm_simulation=true.",
            simulation_target_ids=_candidate_ids(candidates),
            simulation_target_count=len(candidates),
        )

    client = build_candidate_expander_client(provider_id=provider_id, model_id=model_id)
    try:
        return simulate_candidates_with_llm(
            client=client,
            state=state,
            intent_text=intent_text,
            candidates=candidates,
            display_language=display_language or detect_display_language(intent_text),
            simulation_context=simulation_context,
            model_provider=provider_id,
            model_id=model_id,
        )
    except Exception as exc:
        return LLMSimulationResult(
            enabled=True,
            status="fallback",
            candidates=list(candidates),
            model_provider=provider_id,
            model_id=model_id,
            context_ref=_context_ref(context_payload),
            context_type=_context_type(context_payload),
            error=str(exc),
            simulation_target_ids=_candidate_ids(candidates),
            simulation_target_count=len(candidates),
        )


def simulate_candidates_with_llm(
    *,
    client: LLMClient,
    state: GeneralDecisionState,
    intent_text: str,
    candidates: list[GenericCandidate],
    display_language: str = "en",
    simulation_context: Any | None = None,
    model_provider: str = "",
    model_id: str = "",
) -> LLMSimulationResult:
    context_payload = _compiled_context_payload(simulation_context)
    response = client.generate(
        LLMRequest(
            task_hook=LLMTaskHook.SIMULATION_ADVISE,
            input_text=_build_simulation_prompt(
                state=state,
                intent_text=intent_text,
                candidates=candidates,
                display_language=display_language,
                simulation_context=simulation_context,
            ),
            system_text=_system_prompt(display_language),
            response_format_hint="json_object",
            temperature=0.2,
            max_tokens=_simulation_max_tokens(len(candidates)),
            metadata={
                "purpose": "generic_candidate_simulation",
                "context_ref": _context_ref(context_payload),
            },
        )
    )
    try:
        payload = _parse_simulation_payload(response.output_text)
    except Exception as exc:
        return LLMSimulationResult(
            enabled=True,
            status="fallback",
            candidates=list(candidates),
            model_provider=model_provider or response.provider_id,
            model_id=model_id or response.model_id,
            request_id=response.request_id,
            context_ref=_context_ref(context_payload),
            context_type=_context_type(context_payload),
            error=str(exc),
            raw_output=response.output_text,
            simulation_target_ids=_candidate_ids(candidates),
            simulation_target_count=len(candidates),
        )
    raw_simulations = payload.get("simulations")
    if not isinstance(raw_simulations, list):
        return LLMSimulationResult(
            enabled=True,
            status="fallback",
            candidates=list(candidates),
            model_provider=model_provider or response.provider_id,
            model_id=model_id or response.model_id,
            request_id=response.request_id,
            context_ref=_context_ref(context_payload),
            context_type=_context_type(context_payload),
            error="missing simulations list",
            raw_output=response.output_text,
            simulation_target_ids=_candidate_ids(candidates),
            simulation_target_count=len(candidates),
        )

    by_candidate_id = {candidate.candidate_id: candidate for candidate in candidates}
    simulations_by_candidate_id: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    unmatched_simulation_ids: list[str] = []
    for index, raw_simulation in enumerate(raw_simulations):
        try:
            simulation = _normalize_simulation(raw_simulation)
        except Exception as exc:
            rejected.append({"index": index, "reason": str(exc)})
            continue
        candidate_id = simulation["candidate_id"]
        if candidate_id not in by_candidate_id:
            unmatched_simulation_ids.append(candidate_id)
            rejected.append({"index": index, "candidate_id": candidate_id, "reason": "unknown candidate_id"})
            continue
        simulations_by_candidate_id[candidate_id] = simulation

    simulated_candidates = [
        _clone_with_simulation(candidate, simulations_by_candidate_id.get(candidate.candidate_id))
        for candidate in candidates
    ]
    applied_count = len(simulations_by_candidate_id)
    return LLMSimulationResult(
        enabled=True,
        status="simulated" if applied_count else "no_valid_simulations",
        candidates=simulated_candidates,
        proposed_count=len(raw_simulations),
        applied_count=applied_count,
        rejected_count=len(rejected),
        model_provider=model_provider or response.provider_id,
        model_id=model_id or response.model_id,
        request_id=response.request_id,
        context_ref=_context_ref(context_payload),
        context_type=_context_type(context_payload),
        rejected=rejected,
        simulation_target_ids=_candidate_ids(candidates),
        simulation_target_count=len(candidates),
        matched_simulation_ids=list(simulations_by_candidate_id.keys()),
        unmatched_simulation_ids=unmatched_simulation_ids,
    )


def _candidate_ids(candidates: list[GenericCandidate]) -> list[str]:
    return [candidate.candidate_id for candidate in candidates]


def _clone_with_simulation(
    candidate: GenericCandidate,
    simulation: dict[str, Any] | None,
) -> GenericCandidate:
    payload = candidate.to_payload()
    if simulation is not None:
        metadata = dict(payload.get("metadata")) if isinstance(payload.get("metadata"), dict) else {}
        metadata["llm_simulation"] = dict(simulation)
        payload["metadata"] = metadata
    return GenericCandidate.from_payload(payload)


def _normalize_simulation(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("simulation must be an object")
    candidate_id = str(payload.get("candidate_id") or "").strip()
    if not candidate_id:
        raise ValueError("candidate_id is required")
    expected_outcome = str(
        payload.get("expected_outcome")
        or payload.get("simulated_outcome")
        or payload.get("summary")
        or ""
    ).strip()
    if not expected_outcome:
        raise ValueError("expected_outcome is required")
    downside = str(payload.get("downside") or "").strip()
    if not downside:
        downside = _join_list(_string_list(payload.get("likely_risks")))
    success_signal = str(payload.get("success_signal") or "").strip()
    time_fit = _normalize_time_fit(payload.get("time_fit"))
    confidence = _float_between_0_and_1(payload.get("confidence"), default=0.0)
    estimated_time = payload.get("estimated_time_minutes")
    return {
        "candidate_id": candidate_id,
        "expected_outcome": expected_outcome,
        "downside": downside,
        "success_signal": success_signal,
        "time_fit": time_fit,
        "simulated_outcome": expected_outcome,
        "likely_benefits": _string_list(payload.get("likely_benefits")),
        "likely_risks": _string_list(payload.get("likely_risks")),
        "estimated_time_minutes": estimated_time if isinstance(estimated_time, int) else None,
        "failure_modes": _string_list(payload.get("failure_modes")),
        "confidence": confidence,
        "source": "llm_simulation_runner",
    }


def _build_simulation_prompt(
    *,
    state: GeneralDecisionState,
    intent_text: str,
    candidates: list[GenericCandidate],
    display_language: str,
    simulation_context: Any | None = None,
) -> str:
    compiled_context = _compact_simulation_context_for_prompt(
        _compiled_context_payload(simulation_context)
    )
    payload = {
        "user_intent": intent_text,
        "display_language": display_language,
        "language_instruction": language_instruction(display_language),
        "context_usage": {
            "primary_context": "compiled_context",
            "legacy_fallback": "state_summary",
            "instruction": (
                "Use compiled_context as the source of truth for simulation. "
                "It contains current_intent, candidate_decisions, active_decision_frame, "
                "recent decisions, executor affordance, assumptions, evaluation axes, "
                "historical analogs, and retrieved memory. Use state_summary only when "
                "compiled_context is empty or missing a specific field."
            ),
        },
        "compiled_context": compiled_context,
        "state_summary": {
            "role": "legacy_fallback_only",
            "intents": [_compact_record(item.to_payload(), ("intent_id", "summary", "urgency", "target_refs")) for item in state.intents[:8]],
            "work_items": [_compact_record(item.to_payload(), ("work_item_id", "title", "urgency", "status", "blocker_refs")) for item in state.work_items[:8]],
            "constraints": [_compact_record(item.to_payload(), ("constraint_id", "kind", "description", "severity", "applies_to_refs")) for item in state.constraints[:8]],
            "risks": [_compact_record(item.to_payload(), ("risk_id", "kind", "description", "level", "applies_to_refs")) for item in state.risks[:8]],
        },
        "candidates": [
            _compact_record(
                candidate.to_payload(),
                (
                    "candidate_id",
                    "action_type",
                    "intent",
                    "target_refs",
                    "risk_profile",
                    "metadata",
                ),
            )
            for candidate in candidates
        ],
        "response_schema": {
            "simulations": [
                {
                    "candidate_id": "must match an input candidate_id",
                    "expected_outcome": "concise likely result if this candidate is chosen",
                    "downside": "main downside or trade-off to watch",
                    "success_signal": "observable signal that this candidate worked",
                    "confidence": 0.7,
                }
            ]
        },
    }
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)


def _simulation_max_tokens(candidate_count: int) -> int:
    if candidate_count <= 1:
        return 700
    if candidate_count == 2:
        return 900
    if candidate_count == 3:
        return 1100
    return 1200


def _compiled_context_payload(context: Any | None) -> dict[str, Any]:
    if context is None:
        return {}
    payload = payload_value(context)
    return payload if isinstance(payload, dict) else {"value": payload}


def _compact_simulation_context_for_prompt(context_payload: dict[str, Any]) -> dict[str, Any]:
    if not context_payload:
        return {}
    return {
        "id": context_payload.get("id", ""),
        "context_type": context_payload.get("context_type", ""),
        "schema_version": context_payload.get("schema_version", ""),
        "world_state_id": context_payload.get("world_state_id", ""),
        "domain": context_payload.get("domain", ""),
        "decision_context_ref": context_payload.get("decision_context_ref", ""),
        "current_intent": _dict(context_payload.get("current_intent")),
        "active_decision_frame": _compact_active_decision_frame(
            _dict(context_payload.get("active_decision_frame"))
        ),
        "candidate_decisions": _take_dicts(context_payload.get("candidate_decisions"), 12),
        "candidate_intents": _take_dicts(context_payload.get("candidate_intents"), 12),
        "recent_decisions": _take_dicts(context_payload.get("recent_decisions"), 6),
        "recent_approvals": _take_dicts(context_payload.get("recent_approvals"), 6),
        "executor_affordance": _dict(context_payload.get("executor_affordance")),
        "session_summary": _dict(context_payload.get("session_summary")),
        "workspace_context": _dict(context_payload.get("workspace_context")),
        "assumptions": _take_dicts(context_payload.get("assumptions"), 8),
        "evaluation_axes": _take_dicts(context_payload.get("evaluation_axes"), 8),
        "historical_analogs": _take_dicts(context_payload.get("historical_analogs"), 8),
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
        "You simulate Spice generic decision candidates before execution. "
        "Return only a JSON object. Do not select a winner. Do not execute. "
        "Use compiled_context as the primary simulation context. Consult "
        "current_intent, candidate_decisions, active_decision_frame, recent_decisions, "
        "executor_affordance, assumptions, evaluation_axes, historical_analogs, and "
        "retrieved_memory before using legacy state_summary. "
        "Do not change candidate IDs. Provide at most one simulation per candidate. "
        "For each candidate, return only these fields: candidate_id, expected_outcome, "
        "downside, success_signal, and confidence. Compress likely risks, trade-offs, "
        "and costs into the downside string. Do not include extra fields, arrays, "
        "markdown, comments, or explanations. Keep each field concise. "
        + language_instruction(display_language)
    )


def _parse_simulation_payload(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        extracted = extract_first_json_object(stripped)
        if not extracted:
            raise ValueError(SIMULATION_OUTPUT_MALFORMED_ERROR) from exc
        try:
            payload = json.loads(extracted)
        except json.JSONDecodeError as nested_exc:
            raise ValueError(SIMULATION_OUTPUT_MALFORMED_ERROR) from nested_exc
        if '"simulations"' in stripped and not (
            isinstance(payload, dict) and isinstance(payload.get("simulations"), list)
        ):
            raise ValueError(SIMULATION_OUTPUT_MALFORMED_ERROR) from exc
    if not isinstance(payload, dict):
        raise ValueError("LLM simulation response must be a JSON object.")
    return payload


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


def _float_between_0_and_1(value: Any, *, default: float) -> float:
    if not isinstance(value, (int, float)):
        return default
    return min(1.0, max(0.0, float(value)))


def _compact_record(payload: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: payload_value(payload.get(key)) for key in keys if payload.get(key) not in (None, "", [])}


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _join_list(value: list[str]) -> str:
    return "; ".join(item for item in value if item)


def _normalize_time_fit(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "fit": "fits",
        "fits": "fits",
        "fits_window": "fits",
        "within_window": "fits",
        "ok": "fits",
        "tight": "tight",
        "risky": "tight",
        "too_slow": "too_long",
        "too_long": "too_long",
        "too_much": "too_long",
        "over_window": "too_long",
        "unknown": "unknown",
        "unclear": "unknown",
        "": "unknown",
    }
    return aliases.get(normalized, normalized if normalized else "unknown")
