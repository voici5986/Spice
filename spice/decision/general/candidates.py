from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from spice.decision.general.state import GeneralDecisionState
from spice.decision.general.types import (
    Capability,
    Constraint,
    Intent,
    PayloadRecord,
    WorkItem,
    safe_dataclass_from_payload,
)
from spice.language import detect_display_language


GENERIC_ACTION_TYPES = (
    "intent.execute",
    "capability.use",
    "item.triage",
    "context.prepare",
    "state.observe_more",
    "artifact.draft",
    "approval.request",
    "user.clarify",
    "time.defer",
    "state.record",
    "item.ignore",
    "task.split",
)

CANDIDATE_KINDS = frozenset({"decision", "runtime_action", "execution_handoff"})
PLANNING_ACTION_TYPES = frozenset(
    {"task.split", "artifact.draft", "item.triage", "context.prepare", "user.clarify"}
)


@dataclass(slots=True)
class EstimatedCost(PayloadRecord):
    time_minutes: int | None = None
    attention: str = "unknown"
    money: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RiskProfile(PayloadRecord):
    level: str = "unknown"
    risk_refs: list[str] = field(default_factory=list)
    summary: str = ""
    uncertainty: str = "unknown"


@dataclass(slots=True)
class ExpectedStateDelta(PayloadRecord):
    creates_refs: list[str] = field(default_factory=list)
    updates_refs: list[str] = field(default_factory=list)
    closes_refs: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass(slots=True)
class ExecutionBoundary(PayloadRecord):
    mode: str = "none"
    target: str = ""
    protocol: str = ""
    required_capability: str = ""
    requires_confirmation: bool = True
    side_effect_class: str = "none"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenericExecutionIntent(PayloadRecord):
    intent_class: str = "advisory"
    requested: bool = False
    handoff_task: str = ""
    reason: str = ""
    required_permission_hint: str = "unknown"
    side_effect_class: str = "none"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenericCandidate(PayloadRecord):
    candidate_id: str
    action_type: str
    intent: str
    candidate_kind: str = "runtime_action"
    target_refs: list[str] = field(default_factory=list)
    required_capability: str = ""
    execution_intent: GenericExecutionIntent = field(default_factory=GenericExecutionIntent)
    estimated_cost: EstimatedCost = field(default_factory=EstimatedCost)
    risk_profile: RiskProfile = field(default_factory=RiskProfile)
    reversibility: str = "unknown"
    requires_confirmation: bool = True
    expected_state_delta: ExpectedStateDelta = field(default_factory=ExpectedStateDelta)
    execution_boundary: ExecutionBoundary = field(default_factory=ExecutionBoundary)
    constraints_triggered: list[dict[str, Any]] = field(default_factory=list)
    why_available: list[str] = field(default_factory=list)
    why_blocked: list[str] = field(default_factory=list)
    side_effect_class: str = "none"
    availability_status: str = "available"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "GenericCandidate":
        candidate = safe_dataclass_from_payload(cls, payload)
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        candidate.candidate_kind = _candidate_kind(
            payload.get("candidate_kind") or metadata.get("candidate_kind")
        )
        candidate.execution_intent = _load_execution_intent(
            payload.get("execution_intent", metadata.get("execution_intent"))
        )
        candidate.estimated_cost = _load_nested(
            EstimatedCost,
            payload.get("estimated_cost"),
            EstimatedCost(),
        )
        candidate.risk_profile = _load_nested(
            RiskProfile,
            payload.get("risk_profile"),
            RiskProfile(),
        )
        candidate.expected_state_delta = _load_nested(
            ExpectedStateDelta,
            payload.get("expected_state_delta"),
            ExpectedStateDelta(),
        )
        candidate.execution_boundary = _load_nested(
            ExecutionBoundary,
            payload.get("execution_boundary"),
            ExecutionBoundary(),
        )
        return candidate


def is_approval_eligible_executable_candidate(candidate: GenericCandidate) -> bool:
    """Return true when a candidate can cross the executor boundary after approval."""

    if candidate.availability_status == "blocked":
        return False
    if not _execution_intent_requests_handoff(candidate):
        return False
    if _normalized_candidate_side_effect(candidate) != "external_effect":
        return False
    if not candidate.requires_confirmation:
        return False
    boundary = candidate.execution_boundary
    if boundary is not None and boundary.requires_confirmation is False:
        return False
    if not _has_concrete_handoff_anchor(candidate):
        return False
    if candidate.action_type == "capability.use" and not candidate.required_capability:
        return False
    return True


def _execution_intent_requests_handoff(candidate: GenericCandidate) -> bool:
    execution_intent = getattr(candidate, "execution_intent", GenericExecutionIntent())
    return (
        execution_intent.intent_class == "execution_requested"
        and bool(execution_intent.requested)
    )


def _normalized_candidate_side_effect(candidate: GenericCandidate) -> str:
    boundary = candidate.execution_boundary
    execution_intent = getattr(candidate, "execution_intent", GenericExecutionIntent())
    values = [
        candidate.side_effect_class,
        getattr(execution_intent, "side_effect_class", ""),
        boundary.side_effect_class if boundary is not None else "",
        boundary.mode if boundary is not None else "",
    ]
    normalized: list[str] = []
    for value in values:
        token = str(value or "").strip().lower()
        if token in {"external", "external_effect", "execute", "write", "send", "capability"}:
            normalized.append("external_effect")
        elif token in {"draft", "state_change"}:
            normalized.append(token)
        elif token in {"none", "read_only", "read-only", "read_or_prepare", "read-or-prepare"}:
            normalized.append("read_only")
    if "external_effect" in normalized:
        return "external_effect"
    if "state_change" in normalized:
        return "state_change"
    if "draft" in normalized:
        return "draft"
    if "read_only" in normalized:
        return "read_only"
    return ""


def _candidate_kind(value: Any) -> str:
    kind = str(value or "runtime_action").strip()
    return kind if kind in CANDIDATE_KINDS else "runtime_action"


def _load_execution_intent(payload: Any) -> GenericExecutionIntent:
    if not isinstance(payload, dict):
        return GenericExecutionIntent()
    data = dict(payload)
    if "requested" not in data and "needs_execution" in data:
        data["requested"] = bool(data.get("needs_execution"))
    if "intent_class" not in data:
        alias = str(
            data.get("execution_mode")
            or data.get("intent_type")
            or data.get("mode")
            or ""
        ).strip()
        if alias in {"advisory", "execution_requested"}:
            data["intent_class"] = alias
        else:
            data["intent_class"] = (
                "execution_requested" if bool(data.get("requested")) else "advisory"
            )
    if "handoff_task" not in data:
        data["handoff_task"] = str(
            data.get("executor_task")
            or data.get("execution_objective")
            or data.get("requested_action")
            or ""
        )
    if "side_effect_class" not in data and data.get("side_effect"):
        data["side_effect_class"] = str(data.get("side_effect") or "none")
    return safe_dataclass_from_payload(GenericExecutionIntent, data)


def _has_concrete_handoff_anchor(candidate: GenericCandidate) -> bool:
    execution_intent = getattr(candidate, "execution_intent", GenericExecutionIntent())
    if execution_intent.handoff_task:
        return True
    boundary = candidate.execution_boundary
    if candidate.target_refs:
        return True
    if boundary is None:
        return False
    if boundary.target:
        return True
    if boundary.protocol:
        return True
    return boundary.mode in {"execution_intent", "capability", "sdep"}


def generate_generic_candidates(state: GeneralDecisionState) -> list[GenericCandidate]:
    candidates: list[GenericCandidate] = []

    active_intents = [item for item in state.intents if item.status == "active"]
    open_work_items = [item for item in state.work_items if item.status in {"open", "active"}]
    active_constraints = [item for item in state.constraints if item.status == "active"]
    available_capabilities = [
        item for item in state.capabilities if item.status == "available"
    ]

    for intent in active_intents:
        candidates.append(_execute_intent_candidate(intent, active_constraints))
        candidates.append(_prepare_context_candidate(intent))
        candidates.append(_defer_until_candidate(intent))
        candidates.append(_record_only_candidate("intent", intent.intent_id, intent.summary))

        for capability in available_capabilities:
            candidates.append(_delegate_candidate(intent, capability, active_constraints))
            if capability.requires_confirmation:
                candidates.append(_request_permission_candidate(intent, capability))

    for work_item in open_work_items:
        candidates.append(_quick_triage_candidate(work_item, active_constraints))
        candidates.append(_create_draft_candidate(work_item))
        candidates.append(_ignore_temporarily_candidate(work_item, active_constraints))
        if _should_split(work_item):
            candidates.append(_split_task_candidate(work_item))

    for observation in state.observations:
        confidence = observation.confidence
        has_uncertainty = bool(
            confidence.uncertain_fields
            or confidence.missing_fields
            or confidence.level in {"low", "unknown"}
        )
        subject_ref = observation.subject.subject_id if observation.subject else ""
        summary = observation.summary or subject_ref or observation.observation_id
        if has_uncertainty:
            candidates.append(
                _ask_user_candidate(
                    observation.observation_id,
                    subject_ref,
                    confidence.missing_fields,
                    confidence.uncertain_fields,
                )
            )
            candidates.append(_observe_more_candidate(observation.observation_id, subject_ref))
        candidates.append(
            _record_only_candidate("observation", observation.observation_id, summary)
        )

    for signal in state.signals:
        candidates.append(_record_only_candidate("signal", signal.signal_id, signal.summary))

    return _dedupe_candidates(candidates)


def _execute_intent_candidate(
    intent: Intent,
    constraints: list[Constraint],
) -> GenericCandidate:
    target_refs = _target_refs(intent.intent_id, intent.target_refs)
    return _candidate(
        action_type="intent.execute",
        intent=f"Act on intent: {intent.summary}",
        target_refs=target_refs,
        execution_intent=GenericExecutionIntent(
            intent_class="execution_requested",
            requested=True,
            handoff_task=f"Act on intent: {intent.summary}",
            reason="Rule candidate generated from an active intent with an execution boundary.",
            side_effect_class="external_effect",
        ),
        estimated_cost=EstimatedCost(attention="medium"),
        expected_state_delta=ExpectedStateDelta(
            updates_refs=target_refs,
            summary="The intent may move from active to handled or partially handled.",
        ),
        execution_boundary=ExecutionBoundary(
            mode="execution_intent",
            target="external_execution_boundary",
            requires_confirmation=True,
            side_effect_class="external",
        ),
        constraints=constraints,
        why_available=["Active intent is present."],
        side_effect_class="external",
    )


def _delegate_candidate(
    intent: Intent,
    capability: Capability,
    constraints: list[Constraint],
) -> GenericCandidate:
    target_refs = _target_refs(intent.intent_id, intent.target_refs)
    side_effect_class = _side_effect_class(capability)
    return _candidate(
        action_type="capability.use",
        intent=f"Delegate intent to available capability: {intent.summary}",
        target_refs=target_refs,
        required_capability=capability.capability_id,
        execution_intent=GenericExecutionIntent(
            intent_class="execution_requested",
            requested=True,
            handoff_task=f"Delegate intent to available capability: {intent.summary}",
            reason="Rule candidate generated from an active intent and available capability.",
            side_effect_class="external_effect" if side_effect_class != "none" else "none",
        ),
        estimated_cost=EstimatedCost(attention="low", notes=["External capability required."]),
        risk_profile=RiskProfile(
            level="medium" if side_effect_class != "none" else "low",
            summary="Delegation risk depends on the external capability and side effects.",
        ),
        expected_state_delta=ExpectedStateDelta(
            updates_refs=target_refs,
            summary="Execution outcome should return as an outcome record.",
        ),
        execution_boundary=ExecutionBoundary(
            mode="capability",
            required_capability=capability.capability_id,
            requires_confirmation=capability.requires_confirmation,
            side_effect_class=side_effect_class,
        ),
        constraints=constraints,
        why_available=[f"Capability {capability.capability_id} is available."],
        side_effect_class=side_effect_class,
        requires_confirmation=capability.requires_confirmation,
    )


def _quick_triage_candidate(
    work_item: WorkItem,
    constraints: list[Constraint],
) -> GenericCandidate:
    target_refs = [work_item.work_item_id]
    return _candidate(
        action_type="item.triage",
        intent=f"Reduce uncertainty for work item: {work_item.title}",
        target_refs=target_refs,
        estimated_cost=EstimatedCost(time_minutes=5, attention="low"),
        risk_profile=RiskProfile(level="low", summary="Small bounded action."),
        reversibility="high",
        expected_state_delta=ExpectedStateDelta(
            updates_refs=target_refs,
            summary="Work item gains status, summary, or follow-up context.",
        ),
        constraints=constraints,
        why_available=["Open work item can be triaged without full execution."],
        side_effect_class="low",
    )


def _prepare_context_candidate(intent: Intent) -> GenericCandidate:
    target_refs = _target_refs(intent.intent_id, intent.target_refs)
    return _candidate(
        action_type="context.prepare",
        intent=f"Prepare context before acting: {intent.summary}",
        target_refs=target_refs,
        estimated_cost=EstimatedCost(attention="low"),
        risk_profile=RiskProfile(level="low", summary="Context preparation is reversible."),
        reversibility="high",
        expected_state_delta=ExpectedStateDelta(
            updates_refs=target_refs,
            summary="More context becomes available before an execution decision.",
        ),
        why_available=["Active intent has enough structure for context preparation."],
        side_effect_class="none",
        requires_confirmation=False,
    )


def _observe_more_candidate(observation_id: str, subject_ref: str) -> GenericCandidate:
    target_refs = [ref for ref in [subject_ref, observation_id] if ref]
    return _candidate(
        action_type="state.observe_more",
        intent="Gather more evidence before choosing an action.",
        target_refs=target_refs,
        estimated_cost=EstimatedCost(attention="low"),
        risk_profile=RiskProfile(level="low", summary="More observation reduces uncertainty."),
        reversibility="high",
        expected_state_delta=ExpectedStateDelta(
            updates_refs=target_refs,
            summary="Additional evidence may clarify the state.",
        ),
        why_available=["Observation has missing or uncertain fields."],
        side_effect_class="none",
        requires_confirmation=False,
    )


def _create_draft_candidate(work_item: WorkItem) -> GenericCandidate:
    return _candidate(
        action_type="artifact.draft",
        intent=f"Create a draft for work item: {work_item.title}",
        target_refs=[work_item.work_item_id],
        estimated_cost=EstimatedCost(attention="medium"),
        risk_profile=RiskProfile(level="low", summary="Drafting avoids direct commitment."),
        reversibility="high",
        expected_state_delta=ExpectedStateDelta(
            updates_refs=[work_item.work_item_id],
            summary="A draft artifact or prepared response may be created.",
        ),
        execution_boundary=ExecutionBoundary(
            mode="draft",
            target="execution_boundary",
            requires_confirmation=True,
            side_effect_class="draft",
        ),
        why_available=["Open work item can be advanced without final execution."],
        side_effect_class="draft",
    )


def _request_permission_candidate(intent: Intent, capability: Capability) -> GenericCandidate:
    target_refs = _target_refs(intent.intent_id, intent.target_refs)
    return _candidate(
        action_type="approval.request",
        intent=f"Request approval before using capability: {capability.capability_id}",
        target_refs=target_refs,
        required_capability=capability.capability_id,
        estimated_cost=EstimatedCost(attention="low"),
        risk_profile=RiskProfile(level="low", summary="Permission request gates execution."),
        reversibility="high",
        expected_state_delta=ExpectedStateDelta(
            creates_refs=[f"approval.{_slug(intent.intent_id)}.{_slug(capability.capability_id)}"],
            summary="An approval checkpoint may be created.",
        ),
        execution_boundary=ExecutionBoundary(
            mode="approval",
            required_capability=capability.capability_id,
            requires_confirmation=True,
            side_effect_class="none",
        ),
        why_available=["Capability requires confirmation before execution."],
        side_effect_class="none",
    )


def _ask_user_candidate(
    observation_id: str,
    subject_ref: str,
    missing_fields: list[str],
    uncertain_fields: list[str],
) -> GenericCandidate:
    target_refs = [ref for ref in [subject_ref, observation_id] if ref]
    return _candidate(
        action_type="user.clarify",
        intent="Ask the user to clarify missing or uncertain state.",
        target_refs=target_refs,
        estimated_cost=EstimatedCost(attention="low"),
        risk_profile=RiskProfile(
            level="low",
            summary="Clarification reduces uncertainty before execution.",
            uncertainty="high",
        ),
        reversibility="high",
        expected_state_delta=ExpectedStateDelta(
            updates_refs=target_refs,
            summary="Missing or uncertain fields may become explicit.",
        ),
        why_available=_compact_reasons(
            [
                _field_reason("Missing fields", missing_fields),
                _field_reason("Uncertain fields", uncertain_fields),
            ]
        ),
        side_effect_class="none",
        requires_confirmation=False,
    )


def _defer_until_candidate(intent: Intent) -> GenericCandidate:
    target_refs = _target_refs(intent.intent_id, intent.target_refs)
    return _candidate(
        action_type="time.defer",
        intent=f"Defer intent until a clearer decision point: {intent.summary}",
        target_refs=target_refs,
        estimated_cost=EstimatedCost(time_minutes=0, attention="low"),
        risk_profile=RiskProfile(level="medium", summary="Deferral can preserve focus but delay progress."),
        reversibility="medium",
        expected_state_delta=ExpectedStateDelta(
            updates_refs=target_refs,
            summary="The intent remains open with a future follow-up point.",
        ),
        why_available=["Active intent can remain open for a later decision checkpoint."],
        side_effect_class="none",
        requires_confirmation=False,
    )


def _record_only_candidate(kind: str, item_ref: str, summary: str) -> GenericCandidate:
    return _candidate(
        action_type="state.record",
        intent=f"Record {kind} without taking action: {summary}",
        target_refs=[item_ref],
        estimated_cost=EstimatedCost(time_minutes=0, attention="low"),
        risk_profile=RiskProfile(level="low", summary="No execution side effect."),
        reversibility="high",
        expected_state_delta=ExpectedStateDelta(
            updates_refs=[item_ref],
            summary="The item remains in state for future decisions.",
        ),
        why_available=[f"{kind} can be retained without crossing an execution boundary."],
        side_effect_class="none",
        requires_confirmation=False,
    )


def _ignore_temporarily_candidate(
    work_item: WorkItem,
    constraints: list[Constraint],
) -> GenericCandidate:
    target_refs = [work_item.work_item_id]
    return _candidate(
        action_type="item.ignore",
        intent=f"Leave work item unchanged for now: {work_item.title}",
        target_refs=target_refs,
        estimated_cost=EstimatedCost(time_minutes=0, attention="low"),
        risk_profile=RiskProfile(level="medium", summary="Ignoring may allow unresolved risk to grow."),
        reversibility="medium",
        expected_state_delta=ExpectedStateDelta(
            updates_refs=target_refs,
            summary="The work item remains open and may need later follow-up.",
        ),
        constraints=constraints,
        why_available=["Open work item can be left unchanged if no constraint blocks it."],
        side_effect_class="none",
        requires_confirmation=False,
    )


def _split_task_candidate(work_item: WorkItem) -> GenericCandidate:
    return _candidate(
        action_type="task.split",
        intent=f"Split a large work item into smaller parts: {work_item.title}",
        target_refs=[work_item.work_item_id],
        estimated_cost=EstimatedCost(attention="medium"),
        risk_profile=RiskProfile(level="low", summary="Splitting reduces execution uncertainty."),
        reversibility="high",
        expected_state_delta=ExpectedStateDelta(
            creates_refs=[f"open_loop.{_slug(work_item.work_item_id)}.parts"],
            updates_refs=[work_item.work_item_id],
            summary="The work item may become smaller actionable parts.",
        ),
        why_available=["Work item is large or has blockers that can be decomposed."],
        side_effect_class="none",
    )


def _candidate(
    *,
    action_type: str,
    intent: str,
    target_refs: list[str],
    candidate_kind: str = "runtime_action",
    required_capability: str = "",
    execution_intent: GenericExecutionIntent | None = None,
    estimated_cost: EstimatedCost | None = None,
    risk_profile: RiskProfile | None = None,
    reversibility: str = "unknown",
    requires_confirmation: bool = True,
    expected_state_delta: ExpectedStateDelta | None = None,
    execution_boundary: ExecutionBoundary | None = None,
    constraints: list[Constraint] | None = None,
    why_available: list[str] | None = None,
    side_effect_class: str = "none",
    metadata: dict[str, Any] | None = None,
) -> GenericCandidate:
    constraints_triggered = _constraints_for_targets(target_refs, constraints or [])
    why_blocked = [
        str(item.get("description", "Constraint blocks this candidate."))
        for item in constraints_triggered
        if item.get("severity") == "veto"
    ]
    availability_status = _availability_status(
        why_blocked=why_blocked,
        requires_confirmation=requires_confirmation,
        why_available=why_available or [],
    )
    product_metadata = _user_facing_metadata(
        action_type=action_type,
        intent=intent,
        expected_state_delta=expected_state_delta or ExpectedStateDelta(),
        required_capability=required_capability,
        side_effect_class=side_effect_class,
    )
    resolved_execution_intent = execution_intent or GenericExecutionIntent()
    resolved_candidate_kind = _candidate_kind(candidate_kind)
    merged_metadata = {
        "candidate_kind": resolved_candidate_kind,
        "candidate_source": "rule",
        "execution_intent": resolved_execution_intent.to_payload(),
        **product_metadata,
        **dict(metadata or {}),
    }
    return GenericCandidate(
        candidate_id=_candidate_id(action_type, target_refs, required_capability),
        action_type=action_type,
        intent=intent,
        candidate_kind=resolved_candidate_kind,
        target_refs=list(target_refs),
        required_capability=required_capability,
        execution_intent=resolved_execution_intent,
        estimated_cost=estimated_cost or EstimatedCost(),
        risk_profile=risk_profile or RiskProfile(),
        reversibility=reversibility,
        requires_confirmation=requires_confirmation,
        expected_state_delta=expected_state_delta or ExpectedStateDelta(),
        execution_boundary=execution_boundary or ExecutionBoundary(
            requires_confirmation=requires_confirmation,
            side_effect_class=side_effect_class,
        ),
        constraints_triggered=constraints_triggered,
        why_available=list(why_available or []),
        why_blocked=why_blocked,
        side_effect_class=side_effect_class,
        availability_status=availability_status,
        metadata=merged_metadata,
    )


def _user_facing_metadata(
    *,
    action_type: str,
    intent: str,
    expected_state_delta: ExpectedStateDelta,
    required_capability: str,
    side_effect_class: str,
) -> dict[str, Any]:
    display_language = detect_display_language(intent)
    title = _user_facing_title(action_type, intent, display_language=display_language)
    recommended_action = _recommended_action(
        action_type=action_type,
        intent=intent,
        required_capability=required_capability,
        display_language=display_language,
    )
    expected_result = (
        _localize_expected_summary(expected_state_delta.summary, display_language)
        or _expected_result(action_type, display_language=display_language)
    )
    why_now = _why_now(action_type, side_effect_class, display_language=display_language)
    metadata = {
        "user_facing_title": title,
        "recommended_action": recommended_action,
        "expected_result": expected_result,
        "why_now": why_now,
    }
    executor_task = _executor_task(action_type, intent, display_language=display_language)
    if executor_task:
        metadata["executor_task"] = executor_task
    return metadata


def _user_facing_title(action_type: str, intent: str, *, display_language: str = "en") -> str:
    subject = _subject_from_intent(intent)
    if display_language == "zh":
        titles = {
            "intent.execute": f"执行：{subject}" if subject else "执行用户请求的任务",
            "capability.use": f"交给已配置的 executor：{subject}" if subject else "使用已配置的 executor",
            "item.triage": f"明确下一步：{subject}" if subject else "明确下一步",
            "context.prepare": f"先准备上下文：{subject}" if subject else "先准备上下文",
            "state.observe_more": "先收集更多证据",
            "artifact.draft": f"先起草：{subject}" if subject else "先起草结果",
            "approval.request": "执行前请求批准",
            "user.clarify": "向用户确认细节",
            "time.defer": f"暂缓：{subject}" if subject else "暂缓这个决策",
            "state.record": f"记录：{subject}" if subject else "记录这条信息",
            "item.ignore": f"暂时不处理：{subject}" if subject else "暂时不处理",
            "task.split": f"拆分成更小步骤：{subject}" if subject else "拆分成更小步骤",
        }
        return titles.get(action_type, intent or action_type)
    titles = {
        "intent.execute": f"Act on {subject}" if subject else "Execute the requested task",
        "capability.use": f"Hand {subject} to the configured executor" if subject else "Use the configured executor",
        "item.triage": f"Clarify the next step for {subject}" if subject else "Clarify the next step",
        "context.prepare": f"Prepare context for {subject}" if subject else "Prepare context before acting",
        "state.observe_more": "Gather more evidence first",
        "artifact.draft": f"Draft the artifact for {subject}" if subject else "Draft the artifact first",
        "approval.request": "Ask for approval before execution",
        "user.clarify": "Ask for clarification",
        "time.defer": f"Defer {subject}" if subject else "Defer this decision",
        "state.record": f"Record {subject}" if subject else "Record this information",
        "item.ignore": f"Leave {subject} unchanged for now" if subject else "Leave it unchanged for now",
        "task.split": f"Break {subject} into smaller steps" if subject else "Break the task into smaller steps",
    }
    return titles.get(action_type, intent or action_type)


def _recommended_action(
    *,
    action_type: str,
    intent: str,
    required_capability: str,
    display_language: str = "en",
) -> str:
    if display_language == "zh":
        user_intent = _subject_from_intent(intent) or intent
        if action_type == "intent.execute":
            return f"批准这个决策，并交给当前配置的 executor 执行：{user_intent}"
        if action_type == "capability.use":
            capability = f"（{required_capability}）" if required_capability else ""
            return f"批准这个决策，并把任务交给已配置的能力{capability}：{user_intent}"
        if action_type == "item.triage":
            return f"让 executor 检查这项工作、识别阻塞点，并返回最安全的下一步：{user_intent}"
        if action_type == "artifact.draft":
            return f"先准备一份草稿供检查，再决定是否执行最终变更：{user_intent}"
        if action_type == "task.split":
            return f"先把任务拆成小而可验证的步骤，再进入执行：{user_intent}"
        if action_type == "context.prepare":
            return f"先收集相关上下文，再判断是否执行：{user_intent}"
        if action_type == "time.defer":
            return f"保留这个事项，并在更清晰的决策点再处理：{user_intent}"
        if action_type == "user.clarify":
            return f"先向用户确认缺失细节，再选择执行路径：{user_intent}"
        if action_type == "state.observe_more":
            return f"先收集更多证据，再选择行动：{user_intent}"
        if action_type == "state.record":
            return f"只把这条信息记录到 state，不跨越执行边界：{user_intent}"
        if action_type == "item.ignore":
            return f"暂时保持不变，把注意力留给更高优先级事项：{user_intent}"
        return user_intent
    if action_type == "intent.execute":
        return f"Approve this decision to send the task to the configured executor: {intent}"
    if action_type == "capability.use":
        capability = f" via {required_capability}" if required_capability else ""
        return f"Approve this decision to delegate the task{capability}: {intent}"
    if action_type == "item.triage":
        return f"Have the executor inspect the work item, identify the blocker, and return the safest next action: {intent}"
    if action_type == "artifact.draft":
        return f"Prepare a draft artifact for review before committing to a final change: {intent}"
    if action_type == "task.split":
        return f"Split the task into small, verifiable steps before execution: {intent}"
    if action_type == "context.prepare":
        return f"Gather the relevant context before deciding whether to execute: {intent}"
    if action_type == "time.defer":
        return f"Keep this open and revisit it at a clearer decision point: {intent}"
    if action_type == "user.clarify":
        return f"Ask the user for the missing detail before choosing an execution path: {intent}"
    if action_type == "state.observe_more":
        return f"Collect more evidence before selecting an action: {intent}"
    if action_type == "state.record":
        return f"Record this information in state without crossing an execution boundary: {intent}"
    if action_type == "item.ignore":
        return f"Leave this item unchanged for now and preserve attention for higher-priority work: {intent}"
    return intent


def _localize_expected_summary(summary: str, display_language: str) -> str:
    if display_language != "zh":
        return summary
    known = {
        "The intent may move from active to handled or partially handled.": "该 intent 可能从 active 变为已处理或部分处理。",
        "Execution outcome should return as an outcome record.": "执行结果应作为 outcome record 返回。",
        "Work item gains status, summary, or follow-up context.": "该工作项会获得更清晰的状态、摘要或后续上下文。",
        "More context becomes available before an execution decision.": "执行决策前会获得更多相关上下文。",
        "Additional evidence may clarify the state.": "更多证据可能让当前状态更明确。",
        "A draft artifact or prepared response may be created.": "会生成一份草稿或预备回复。",
        "An approval checkpoint may be created.": "可能会创建一个 approval checkpoint。",
        "Missing or uncertain fields may become explicit.": "缺失或不确定字段可能会被明确。",
        "The intent remains open with a future follow-up point.": "该 intent 会保持 open，并等待后续处理点。",
        "The item remains in state for future decisions.": "该事项会保留在 state 中，供未来决策使用。",
        "The work item remains open and may need later follow-up.": "该工作项会保持 open，之后可能还需要跟进。",
        "The work item may become smaller actionable parts.": "该工作项可能会被拆成更小、可执行的部分。",
    }
    return known.get(summary, summary)


def _expected_result(action_type: str, *, display_language: str = "en") -> str:
    if display_language == "zh":
        values = {
            "intent.execute": "executor 会返回这个已批准任务的执行结果。",
            "capability.use": "选中的能力会返回这个已批准任务的执行结果。",
            "item.triage": "工作项会获得更清晰的状态、风险和下一步。",
            "artifact.draft": "会得到一份可检查的草稿。",
            "task.split": "任务会被拆解成更小的后续步骤。",
            "context.prepare": "执行前会获得更充分的上下文。",
            "state.record": "这条信息会被保留下来，供未来决策使用。",
        }
        return values.get(action_type, "决策状态会根据选中的动作更新。")
    values = {
        "intent.execute": "The executor returns an outcome for the approved task.",
        "capability.use": "The selected capability returns an outcome for the approved task.",
        "item.triage": "The work item gains a clearer status, risk, and next action.",
        "artifact.draft": "A draft artifact is available for review.",
        "task.split": "The task is decomposed into smaller follow-up steps.",
        "context.prepare": "The decision has better context before execution.",
        "state.record": "The information is available for future decisions.",
    }
    return values.get(action_type, "The decision state is updated with the selected action.")


def _why_now(action_type: str, side_effect_class: str, *, display_language: str = "en") -> list[str]:
    if display_language == "zh":
        reasons = {
            "intent.execute": ["用户明确要求执行动作，而不是只要建议。", "执行仍然需要通过审批。"],
            "capability.use": ["当前已有可用的 executor 能力。", "执行仍然需要通过审批。"],
            "item.triage": ["该事项需要一个边界清晰的下一步，再进入更深执行。"],
            "artifact.draft": ["先起草可以降低直接做最终变更的风险。"],
            "task.split": ["拆分任务可以降低执行不确定性和 review 风险。"],
            "context.prepare": ["更多上下文可以降低执行风险。"],
            "time.defer": ["当下不必立刻行动时，暂缓可以保留注意力。"],
            "state.record": ["记录下来可以保留信号，同时不触发执行。"],
        }
        result = list(reasons.get(action_type, []))
        if side_effect_class not in {"none", "read_only", ""}:
            result.append("这会跨越 state 或 execution 边界，因此需要审批。")
        return result
    reasons = {
        "intent.execute": ["The user asked for an action, not only advice.", "Execution stays approval-gated."],
        "capability.use": ["A configured executor capability is available.", "Execution stays approval-gated."],
        "item.triage": ["The item needs a bounded next step before deeper execution."],
        "artifact.draft": ["Drafting reduces risk before making a final change."],
        "task.split": ["Splitting lowers execution uncertainty and review risk."],
        "context.prepare": ["More context can reduce execution risk."],
        "time.defer": ["Deferral preserves focus when immediate action is not required."],
        "state.record": ["Recording keeps the signal available without execution."],
    }
    result = list(reasons.get(action_type, []))
    if side_effect_class not in {"none", "read_only", ""}:
        result.append("This crosses a state or execution boundary and requires approval.")
    return result


def _executor_task(action_type: str, intent: str, *, display_language: str = "en") -> str:
    if display_language == "zh":
        if action_type in {"intent.execute", "capability.use"}:
            return intent
        if action_type == "item.triage":
            return f"梳理这项工作，并返回最安全的下一步：{intent}"
        if action_type == "artifact.draft":
            return f"起草请求的内容，不应用无关变更：{intent}"
        if action_type == "task.split":
            return f"把这个任务拆成小而可验证的步骤：{intent}"
        return ""
    if action_type in {"intent.execute", "capability.use"}:
        return intent
    if action_type == "item.triage":
        return f"Triage this work and return the safest next action: {intent}"
    if action_type == "artifact.draft":
        return f"Draft the requested artifact without applying unrelated changes: {intent}"
    if action_type == "task.split":
        return f"Split this task into small, verifiable steps: {intent}"
    return ""


def _subject_from_intent(intent: str) -> str:
    patterns = (
        "Act on intent:",
        "Delegate intent to available capability:",
        "Reduce uncertainty for work item:",
        "Create a draft for work item:",
        "Split a large work item into smaller parts:",
        "Leave work item unchanged for now:",
        "Prepare context before acting:",
        "Defer intent until a clearer decision point:",
        "Record intent without taking action:",
        "Record observation without taking action:",
        "Record signal without taking action:",
    )
    for pattern in patterns:
        if intent.startswith(pattern):
            return intent[len(pattern) :].strip()
    return intent.strip()


def _constraints_for_targets(
    target_refs: list[str],
    constraints: list[Constraint],
) -> list[dict[str, Any]]:
    target_set = set(target_refs)
    triggered: list[dict[str, Any]] = []
    for constraint in constraints:
        applies_to = set(constraint.applies_to_refs)
        if applies_to and target_set.isdisjoint(applies_to):
            continue
        triggered.append(
            {
                "constraint_id": constraint.constraint_id,
                "kind": constraint.kind,
                "severity": constraint.severity,
                "description": constraint.description,
                "applies_to_refs": list(constraint.applies_to_refs),
            }
        )
    return triggered


def _target_refs(fallback_ref: str, target_refs: list[str]) -> list[str]:
    return list(target_refs) if target_refs else [fallback_ref]


def _candidate_id(
    action_type: str,
    target_refs: list[str],
    required_capability: str = "",
) -> str:
    parts = ["candidate", action_type]
    parts.extend(_slug(ref) for ref in target_refs[:2] if ref)
    if required_capability:
        parts.append(_slug(required_capability))
    return ".".join(part for part in parts if part)


def _side_effect_class(capability: Capability) -> str:
    if not capability.side_effects:
        return "none"
    if any("write" in item or "send" in item or "execute" in item for item in capability.side_effects):
        return "external"
    return "read_or_prepare"


def _should_split(work_item: WorkItem) -> bool:
    if work_item.blocker_refs:
        return True
    return bool(work_item.estimate_minutes and work_item.estimate_minutes >= 60)


def _dedupe_candidates(candidates: list[GenericCandidate]) -> list[GenericCandidate]:
    deduped: dict[str, GenericCandidate] = {}
    for candidate in candidates:
        if candidate.candidate_id not in deduped:
            deduped[candidate.candidate_id] = candidate
    return list(deduped.values())


def _availability_status(
    *,
    why_blocked: list[str],
    requires_confirmation: bool,
    why_available: list[str],
) -> str:
    if why_blocked:
        return "blocked"
    if not why_available:
        return "insufficient_context"
    if requires_confirmation:
        return "needs_confirmation"
    return "available"


def _load_nested(cls: type[Any], payload: Any, default: Any) -> Any:
    if not isinstance(payload, dict):
        return default
    return safe_dataclass_from_payload(cls, payload)


def _compact_reasons(reasons: list[str]) -> list[str]:
    return [reason for reason in reasons if reason]


def _field_reason(label: str, fields: list[str]) -> str:
    if not fields:
        return ""
    return f"{label}: {', '.join(fields)}."


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return cleaned or "unknown"
