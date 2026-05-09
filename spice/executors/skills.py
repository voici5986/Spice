from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any


SIDE_EFFECT_CLASSES = (
    "read_only",
    "state_change",
    "external_effect",
)

SKILL_SOURCES = (
    "builtin",
    "executor",
    "user",
    "project",
)


def payload_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {str(key): payload_value(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): payload_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [payload_value(item) for item in value]
    if isinstance(value, tuple):
        return [payload_value(item) for item in value]
    return value


def safe_dataclass_from_payload(cls: type[Any], payload: Any) -> Any:
    if not isinstance(payload, dict):
        raise ValueError(f"{cls.__name__} payload must be a dict")
    allowed = {item.name for item in fields(cls)}
    return cls(**{key: value for key, value in payload.items() if key in allowed})


@dataclass(slots=True)
class SkillPayloadRecord:
    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(slots=True)
class CapabilityDescriptor(SkillPayloadRecord):
    capability_id: str
    display_name: str = ""
    description: str = ""
    status: str = "available"
    side_effect_classes: list[str] = field(default_factory=list)
    max_duration_seconds: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require_non_empty(self.capability_id, "capability_id", "capability")
        for value in self.side_effect_classes:
            _validate_side_effect_class(value)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "CapabilityDescriptor":
        item = safe_dataclass_from_payload(cls, payload)
        item.side_effect_classes = _string_list(payload.get("side_effect_classes"))
        item.metadata = _dict(payload.get("metadata"))
        item.validate()
        return item


@dataclass(slots=True)
class SkillDescriptor(SkillPayloadRecord):
    skill_id: str
    display_name: str = ""
    description: str = ""
    source: str = "executor"
    supported_action_types: list[str] = field(default_factory=list)
    required_capabilities: list[str] = field(default_factory=list)
    side_effect_class: str = "read_only"
    requires_confirmation: bool = True
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    instructions: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require_non_empty(self.skill_id, "skill_id", "skill")
        _require_non_empty_list(
            self.supported_action_types,
            "supported_action_types",
            "skill",
        )
        _validate_side_effect_class(self.side_effect_class)
        if self.source not in SKILL_SOURCES:
            allowed = ", ".join(SKILL_SOURCES)
            raise ValueError(f"skill.source must be one of [{allowed}], got {self.source!r}")
        if not isinstance(self.requires_confirmation, bool):
            raise ValueError("skill.requires_confirmation must be a boolean")

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SkillDescriptor":
        item = safe_dataclass_from_payload(cls, payload)
        item.supported_action_types = _string_list(payload.get("supported_action_types"))
        item.required_capabilities = _string_list(payload.get("required_capabilities"))
        item.input_schema = _dict(payload.get("input_schema"))
        item.output_schema = _dict(payload.get("output_schema"))
        item.instructions = _string_list(payload.get("instructions"))
        item.tags = _string_list(payload.get("tags"))
        item.metadata = _dict(payload.get("metadata"))
        item.validate()
        return item


@dataclass(slots=True)
class ExecutorDescriptor(SkillPayloadRecord):
    executor_id: str
    display_name: str = ""
    description: str = ""
    status: str = "available"
    adapter_type: str = ""
    priority: int = 100
    capabilities: list[CapabilityDescriptor] = field(default_factory=list)
    skills: list[SkillDescriptor] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require_non_empty(self.executor_id, "executor_id", "executor")
        for capability in self.capabilities:
            if not isinstance(capability, CapabilityDescriptor):
                raise ValueError("executor.capabilities must contain CapabilityDescriptor items")
            capability.validate()
        for skill in self.skills:
            if not isinstance(skill, SkillDescriptor):
                raise ValueError("executor.skills must contain SkillDescriptor items")
            skill.validate()

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ExecutorDescriptor":
        item = safe_dataclass_from_payload(cls, payload)
        item.capabilities = [
            CapabilityDescriptor.from_payload(value)
            for value in _dict_list(payload.get("capabilities"))
        ]
        item.skills = [
            SkillDescriptor.from_payload(value)
            for value in _dict_list(payload.get("skills"))
        ]
        item.metadata = _dict(payload.get("metadata"))
        item.validate()
        return item

    def capability_ids(self) -> list[str]:
        return [item.capability_id for item in self.capabilities]

    def skill_ids(self) -> list[str]:
        return [item.skill_id for item in self.skills]


@dataclass(slots=True)
class SkillCatalog(SkillPayloadRecord):
    executors: list[ExecutorDescriptor] = field(default_factory=list)
    builtin_skills: list[SkillDescriptor] = field(default_factory=list)
    user_skills: list[SkillDescriptor] = field(default_factory=list)
    project_skills: list[SkillDescriptor] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        for executor in self.executors:
            if not isinstance(executor, ExecutorDescriptor):
                raise ValueError("catalog.executors must contain ExecutorDescriptor items")
            executor.validate()
        for skill in self.builtin_skills + self.user_skills + self.project_skills:
            if not isinstance(skill, SkillDescriptor):
                raise ValueError("catalog skills must contain SkillDescriptor items")
            skill.validate()

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SkillCatalog":
        catalog = cls(
            executors=[
                ExecutorDescriptor.from_payload(value)
                for value in _dict_list(payload.get("executors"))
            ],
            builtin_skills=[
                SkillDescriptor.from_payload(value)
                for value in _dict_list(payload.get("builtin_skills"))
            ],
            user_skills=[
                SkillDescriptor.from_payload(value)
                for value in _dict_list(payload.get("user_skills"))
            ],
            project_skills=[
                SkillDescriptor.from_payload(value)
                for value in _dict_list(payload.get("project_skills"))
            ],
            metadata=_dict(payload.get("metadata")),
        )
        catalog.validate()
        return catalog

    def all_skills(self) -> list[SkillDescriptor]:
        skills: list[SkillDescriptor] = []
        skills.extend(self.user_skills)
        skills.extend(self.project_skills)
        for executor in self.executors:
            skills.extend(executor.skills)
        skills.extend(self.builtin_skills)
        return skills

    def find_executor(self, executor_id: str) -> ExecutorDescriptor | None:
        for executor in self.executors:
            if executor.executor_id == executor_id:
                return executor
        return None

    def find_skills_for_action(self, action_type: str) -> list[SkillDescriptor]:
        return [
            skill for skill in self.all_skills()
            if action_type in skill.supported_action_types
        ]


@dataclass(slots=True)
class ResolvedSkill(SkillPayloadRecord):
    executor_id: str
    skill_id: str
    action_type: str
    capability_id: str = ""
    side_effect_class: str = "read_only"
    requires_confirmation: bool = True
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    instructions: list[str] = field(default_factory=list)
    resolution_reason: str = ""
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require_non_empty(self.executor_id, "executor_id", "resolved_skill")
        _require_non_empty(self.skill_id, "skill_id", "resolved_skill")
        _require_non_empty(self.action_type, "action_type", "resolved_skill")
        _validate_side_effect_class(self.side_effect_class)
        if not isinstance(self.requires_confirmation, bool):
            raise ValueError("resolved_skill.requires_confirmation must be a boolean")

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ResolvedSkill":
        item = safe_dataclass_from_payload(cls, payload)
        item.input_schema = _dict(payload.get("input_schema"))
        item.output_schema = _dict(payload.get("output_schema"))
        item.instructions = _string_list(payload.get("instructions"))
        item.metadata = _dict(payload.get("metadata"))
        item.validate()
        return item


def builtin_fallback_skill_catalog() -> SkillCatalog:
    catalog = SkillCatalog(
        builtin_skills=[
            SkillDescriptor(
                skill_id="state.record.builtin",
                display_name="Record state",
                source="builtin",
                supported_action_types=["state.record"],
                side_effect_class="state_change",
                requires_confirmation=False,
                output_schema={"type": "state_record"},
            ),
            SkillDescriptor(
                skill_id="user.clarify.builtin",
                display_name="Ask user for clarification",
                source="builtin",
                supported_action_types=["user.clarify"],
                side_effect_class="read_only",
                requires_confirmation=False,
                output_schema={"type": "clarification_request"},
            ),
            SkillDescriptor(
                skill_id="work_item.triage.read_only",
                display_name="Work item triage",
                source="builtin",
                supported_action_types=["item.triage", "context.prepare"],
                required_capabilities=["work_item_triage"],
                side_effect_class="read_only",
                requires_confirmation=False,
                output_schema={"type": "triage_report.v1"},
            ),
            SkillDescriptor(
                skill_id="intent.execute.generic",
                display_name="Execute intent",
                source="builtin",
                supported_action_types=["intent.execute", "capability.use"],
                side_effect_class="external_effect",
                requires_confirmation=True,
                output_schema={"type": "execution_report.v1"},
            ),
        ]
    )
    catalog.validate()
    return catalog


def _require_non_empty(value: Any, field_name: str, context: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{field_name} is required")


def _require_non_empty_list(value: Any, field_name: str, context: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context}.{field_name} must contain at least one item")
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{context}.{field_name}[{index}] is required")


def _validate_side_effect_class(value: str) -> None:
    if value not in SIDE_EFFECT_CLASSES:
        allowed = ", ".join(SIDE_EFFECT_CLASSES)
        raise ValueError(f"side_effect_class must be one of [{allowed}], got {value!r}")


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item.strip()]
