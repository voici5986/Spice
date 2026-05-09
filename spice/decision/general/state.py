from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from spice.decision.general.approval import Approval
from spice.decision.general.observations import GenericObservation
from spice.decision.general.trace import DecisionCheckpoint
from spice.decision.general.types import (
    Capability,
    Commitment,
    Constraint,
    Intent,
    OpenLoop,
    OutcomeRecord,
    PayloadRecord,
    Resource,
    Risk,
    Signal,
    WorkItem,
    safe_dataclass_from_payload,
)
from spice.protocols.world_state import WorldState

GENERAL_STATE_KEY = "general_decision"


@dataclass(slots=True)
class GeneralDecisionState(PayloadRecord):
    state_id: str
    schema_version: str = "0.1"
    signals: list[Signal] = field(default_factory=list)
    observations: list[GenericObservation] = field(default_factory=list)
    intents: list[Intent] = field(default_factory=list)
    commitments: list[Commitment] = field(default_factory=list)
    work_items: list[WorkItem] = field(default_factory=list)
    resources: list[Resource] = field(default_factory=list)
    capabilities: list[Capability] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    risks: list[Risk] = field(default_factory=list)
    open_loops: list[OpenLoop] = field(default_factory=list)
    approvals: list[Approval] = field(default_factory=list)
    decision_checkpoints: list[DecisionCheckpoint] = field(default_factory=list)
    outcomes: list[OutcomeRecord] = field(default_factory=list)
    trace_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def ensure_general_state(world_state: WorldState) -> dict[str, Any]:
    if not isinstance(world_state.domain_state, dict):
        world_state.domain_state = {}
    general = world_state.domain_state.get(GENERAL_STATE_KEY)
    if not isinstance(general, dict):
        general = {}
        world_state.domain_state[GENERAL_STATE_KEY] = general
    general.setdefault("schema_version", "0.1")
    general.setdefault("signals", [])
    general.setdefault("observations", [])
    general.setdefault("intents", [])
    general.setdefault("commitments", [])
    general.setdefault("work_items", [])
    general.setdefault("resources", [])
    general.setdefault("capabilities", [])
    general.setdefault("constraints", [])
    general.setdefault("risks", [])
    general.setdefault("open_loops", [])
    general.setdefault("approvals", [])
    general.setdefault("decision_checkpoints", [])
    general.setdefault("outcomes", [])
    general.setdefault("trace_refs", [])
    general.setdefault("metadata", {})
    return general


def load_general_state(world_state: WorldState) -> GeneralDecisionState:
    general = ensure_general_state(world_state)
    return GeneralDecisionState(
        state_id=world_state.id,
        schema_version=str(general.get("schema_version", "0.1")),
        signals=[
            _load_record(Signal, item)
            for item in _list(general.get("signals"))
            if isinstance(item, dict)
        ],
        observations=[
            GenericObservation.from_payload(item)
            for item in _list(general.get("observations"))
            if isinstance(item, dict)
        ],
        intents=[
            _load_record(Intent, item)
            for item in _list(general.get("intents"))
            if isinstance(item, dict)
        ],
        commitments=[
            _load_record(Commitment, item)
            for item in _list(general.get("commitments"))
            if isinstance(item, dict)
        ],
        work_items=[
            _load_record(WorkItem, item)
            for item in _list(general.get("work_items"))
            if isinstance(item, dict)
        ],
        resources=[
            _load_record(Resource, item)
            for item in _list(general.get("resources"))
            if isinstance(item, dict)
        ],
        capabilities=[
            _load_record(Capability, item)
            for item in _list(general.get("capabilities"))
            if isinstance(item, dict)
        ],
        constraints=[
            _load_record(Constraint, item)
            for item in _list(general.get("constraints"))
            if isinstance(item, dict)
        ],
        risks=[
            _load_record(Risk, item)
            for item in _list(general.get("risks"))
            if isinstance(item, dict)
        ],
        open_loops=[
            _load_record(OpenLoop, item)
            for item in _list(general.get("open_loops"))
            if isinstance(item, dict)
        ],
        approvals=[
            _load_record(Approval, item)
            for item in _list(general.get("approvals"))
            if isinstance(item, dict)
        ],
        decision_checkpoints=[
            DecisionCheckpoint.from_payload(item)
            for item in _list(general.get("decision_checkpoints"))
            if isinstance(item, dict)
        ],
        outcomes=[
            _load_record(OutcomeRecord, item)
            for item in _list(general.get("outcomes"))
            if isinstance(item, dict)
        ],
        trace_refs=[str(item) for item in _list(general.get("trace_refs"))],
        metadata=dict(general.get("metadata", {})),
    )


def store_general_state(world_state: WorldState, state: GeneralDecisionState) -> None:
    if not isinstance(world_state.domain_state, dict):
        world_state.domain_state = {}
    payload = state.to_payload()
    payload.pop("state_id", None)
    world_state.domain_state[GENERAL_STATE_KEY] = payload


def _load_record(cls: type[Any], payload: Any) -> Any:
    return safe_dataclass_from_payload(cls, payload)


def _list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []
