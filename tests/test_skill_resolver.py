from __future__ import annotations

import copy
import json
import unittest

from spice.decision.general.candidates import (
    EstimatedCost,
    ExecutionBoundary,
    ExpectedStateDelta,
    GenericCandidate,
    RiskProfile,
)
from spice.executors import (
    CapabilityDescriptor,
    ExecutorDescriptor,
    SkillCatalog,
    SkillDescriptor,
    SkillResolutionResult,
    candidate_required_side_effect_class,
    resolve_skill_for_candidate,
    resolve_skills_for_candidates,
)


class SkillResolverTests(unittest.TestCase):
    def test_resolves_executor_skill_by_action_and_capability(self) -> None:
        candidate = _candidate(
            action_type="item.triage",
            required_capability="work_item_triage",
            side_effect_class="none",
            requires_confirmation=False,
        )
        catalog = SkillCatalog(
            executors=[
                ExecutorDescriptor(
                    executor_id="codex",
                    priority=10,
                    capabilities=[
                        CapabilityDescriptor(
                            capability_id="work_item_triage",
                            side_effect_classes=["read_only"],
                        )
                    ],
                    skills=[
                        SkillDescriptor(
                            skill_id="work_item.triage.codex",
                            source="executor",
                            supported_action_types=["item.triage"],
                            required_capabilities=["work_item_triage"],
                            side_effect_class="read_only",
                            requires_confirmation=False,
                        )
                    ],
                )
            ]
        )

        result = resolve_skill_for_candidate(candidate, catalog)

        self.assertEqual(result.status, "resolved")
        self.assertIsNotNone(result.resolved_skill)
        self.assertEqual(result.resolved_skill.executor_id, "codex")  # type: ignore[union-attr]
        self.assertEqual(result.resolved_skill.skill_id, "work_item.triage.codex")  # type: ignore[union-attr]
        self.assertEqual(result.resolved_skill.capability_id, "work_item_triage")  # type: ignore[union-attr]
        self.assertFalse(result.resolved_skill.requires_confirmation)  # type: ignore[union-attr]

    def test_user_skill_has_priority_over_executor_skill(self) -> None:
        candidate = _candidate(action_type="context.prepare", requires_confirmation=False)
        catalog = SkillCatalog(
            executors=[
                ExecutorDescriptor(
                    executor_id="codex",
                    priority=1,
                    skills=[
                        SkillDescriptor(
                            skill_id="context.prepare.executor",
                            source="executor",
                            supported_action_types=["context.prepare"],
                            side_effect_class="read_only",
                            requires_confirmation=False,
                        )
                    ],
                )
            ],
            user_skills=[
                SkillDescriptor(
                    skill_id="context.prepare.user",
                    source="user",
                    supported_action_types=["context.prepare"],
                    side_effect_class="read_only",
                    requires_confirmation=False,
                )
            ],
        )

        result = resolve_skill_for_candidate(candidate, catalog)

        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.resolved_skill.skill_id, "context.prepare.user")  # type: ignore[union-attr]
        self.assertEqual(result.resolved_skill.executor_id, "spice.user")  # type: ignore[union-attr]

    def test_project_skill_has_priority_over_executor_skill_but_not_user_skill(self) -> None:
        candidate = _candidate(action_type="context.prepare", requires_confirmation=False)
        catalog = SkillCatalog(
            executors=[
                ExecutorDescriptor(
                    executor_id="codex",
                    priority=1,
                    skills=[
                        SkillDescriptor(
                            skill_id="context.prepare.executor",
                            source="executor",
                            supported_action_types=["context.prepare"],
                            side_effect_class="read_only",
                            requires_confirmation=False,
                        )
                    ],
                )
            ],
            project_skills=[
                SkillDescriptor(
                    skill_id="context.prepare.project",
                    source="project",
                    supported_action_types=["context.prepare"],
                    side_effect_class="read_only",
                    requires_confirmation=False,
                )
            ],
        )

        result = resolve_skill_for_candidate(candidate, catalog)

        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.resolved_skill.skill_id, "context.prepare.project")  # type: ignore[union-attr]

    def test_required_capability_must_exist_on_executor(self) -> None:
        candidate = _candidate(
            action_type="capability.use",
            required_capability="repo_write",
            side_effect_class="external",
        )
        catalog = SkillCatalog(
            executors=[
                ExecutorDescriptor(
                    executor_id="codex",
                    capabilities=[
                        CapabilityDescriptor(
                            capability_id="repo_read",
                            side_effect_classes=["read_only"],
                        )
                    ],
                    skills=[
                        SkillDescriptor(
                            skill_id="intent.execute.codex",
                            source="executor",
                            supported_action_types=["capability.use"],
                            side_effect_class="external_effect",
                        )
                    ],
                )
            ]
        )

        result = resolve_skill_for_candidate(candidate, catalog)

        self.assertEqual(result.status, "unresolved")
        self.assertIn("lacks repo_write", " ".join(result.unresolved_reasons))

    def test_user_skill_with_multiple_capabilities_selects_compatible_executor(self) -> None:
        candidate = _candidate(action_type="artifact.draft", requires_confirmation=True)
        catalog = SkillCatalog(
            executors=[
                ExecutorDescriptor(
                    executor_id="codex",
                    priority=10,
                    capabilities=[
                        CapabilityDescriptor(capability_id="repo_read"),
                        CapabilityDescriptor(capability_id="repo_write"),
                    ],
                )
            ],
            user_skills=[
                SkillDescriptor(
                    skill_id="artifact.draft.user",
                    source="user",
                    supported_action_types=["artifact.draft"],
                    required_capabilities=["repo_read", "repo_write"],
                    side_effect_class="state_change",
                    requires_confirmation=True,
                )
            ],
        )

        result = resolve_skill_for_candidate(candidate, catalog)

        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.resolved_skill.executor_id, "codex")  # type: ignore[union-attr]
        self.assertEqual(result.resolved_skill.skill_id, "artifact.draft.user")  # type: ignore[union-attr]

    def test_side_effect_matching_requires_exact_match_for_read_only_candidate(self) -> None:
        candidate = _candidate(action_type="context.prepare", requires_confirmation=False)
        catalog = SkillCatalog(
            executors=[
                ExecutorDescriptor(
                    executor_id="codex",
                    skills=[
                        SkillDescriptor(
                            skill_id="context.prepare.write",
                            source="executor",
                            supported_action_types=["context.prepare"],
                            side_effect_class="external_effect",
                        )
                    ],
                )
            ]
        )

        result = resolve_skill_for_candidate(candidate, catalog)

        self.assertEqual(result.status, "unresolved")
        self.assertIn("side_effect_mismatch external_effect != read_only", " ".join(result.unresolved_reasons))

    def test_read_only_skill_cannot_resolve_state_change_candidate(self) -> None:
        candidate = _candidate(action_type="state.record", side_effect_class="none", requires_confirmation=False)
        catalog = SkillCatalog(
            project_skills=[
                SkillDescriptor(
                    skill_id="state.record.read_only",
                    source="project",
                    supported_action_types=["state.record"],
                    side_effect_class="read_only",
                    requires_confirmation=False,
                )
            ]
        )

        result = resolve_skill_for_candidate(candidate, catalog)

        self.assertEqual(result.status, "unresolved")
        self.assertIn("side_effect_mismatch read_only != state_change", " ".join(result.unresolved_reasons))

    def test_read_only_skill_cannot_resolve_external_effect_candidate(self) -> None:
        candidate = _candidate(action_type="intent.execute", side_effect_class="external")
        catalog = SkillCatalog(
            project_skills=[
                SkillDescriptor(
                    skill_id="intent.execute.read_only",
                    source="project",
                    supported_action_types=["intent.execute"],
                    side_effect_class="read_only",
                )
            ]
        )

        result = resolve_skill_for_candidate(candidate, catalog)

        self.assertEqual(result.status, "unresolved")
        self.assertIn("side_effect_mismatch read_only != external_effect", " ".join(result.unresolved_reasons))

    def test_state_change_skill_cannot_resolve_external_effect_candidate(self) -> None:
        candidate = _candidate(action_type="intent.execute", side_effect_class="external")
        catalog = SkillCatalog(
            project_skills=[
                SkillDescriptor(
                    skill_id="intent.execute.state_change",
                    source="project",
                    supported_action_types=["intent.execute"],
                    side_effect_class="state_change",
                )
            ]
        )

        result = resolve_skill_for_candidate(candidate, catalog)

        self.assertEqual(result.status, "unresolved")
        self.assertIn("side_effect_mismatch state_change != external_effect", " ".join(result.unresolved_reasons))

    def test_preferred_skill_cannot_bypass_side_effect_mismatch(self) -> None:
        candidate = _candidate(action_type="intent.execute", side_effect_class="external")
        catalog = SkillCatalog(
            project_skills=[
                SkillDescriptor(
                    skill_id="intent.execute.safe",
                    source="project",
                    supported_action_types=["intent.execute"],
                    side_effect_class="read_only",
                )
            ]
        )

        result = resolve_skill_for_candidate(
            candidate,
            catalog,
            preferred_skill_ids=("intent.execute.safe",),
        )

        self.assertEqual(result.status, "unresolved")
        self.assertIn("side_effect_mismatch read_only != external_effect", " ".join(result.unresolved_reasons))

    def test_preferred_executor_cannot_bypass_side_effect_mismatch(self) -> None:
        candidate = _candidate(action_type="intent.execute", side_effect_class="external")
        catalog = SkillCatalog(
            executors=[
                ExecutorDescriptor(
                    executor_id="preferred",
                    skills=[
                        SkillDescriptor(
                            skill_id="intent.execute.read_only",
                            source="executor",
                            supported_action_types=["intent.execute"],
                            side_effect_class="read_only",
                        )
                    ],
                )
            ]
        )

        result = resolve_skill_for_candidate(
            candidate,
            catalog,
            preferred_executor_ids=("preferred",),
        )

        self.assertEqual(result.status, "unresolved")
        self.assertIn("side_effect_mismatch read_only != external_effect", " ".join(result.unresolved_reasons))

    def test_skill_cannot_weaken_candidate_confirmation_requirement(self) -> None:
        candidate = _candidate(
            action_type="intent.execute",
            side_effect_class="external",
            requires_confirmation=True,
        )
        catalog = SkillCatalog(
            executors=[
                ExecutorDescriptor(
                    executor_id="codex",
                    skills=[
                        SkillDescriptor(
                            skill_id="intent.execute.codex",
                            source="executor",
                            supported_action_types=["intent.execute"],
                            side_effect_class="external_effect",
                            requires_confirmation=False,
                        )
                    ],
                )
            ]
        )

        result = resolve_skill_for_candidate(candidate, catalog)

        self.assertEqual(result.status, "resolved")
        self.assertTrue(result.resolved_skill.requires_confirmation)  # type: ignore[union-attr]
        self.assertTrue(
            result.resolved_skill.metadata["requires_confirmation_from_candidate"]  # type: ignore[union-attr]
        )

    def test_executor_priority_is_deterministic_tiebreaker(self) -> None:
        candidate = _candidate(action_type="intent.execute", side_effect_class="external")
        catalog = SkillCatalog(
            executors=[
                _executor_with_intent_skill("slow", priority=50),
                _executor_with_intent_skill("fast", priority=10),
            ]
        )

        first = resolve_skill_for_candidate(candidate, catalog)
        second = resolve_skill_for_candidate(candidate, catalog)

        self.assertEqual(first.resolved_skill.executor_id, "fast")  # type: ignore[union-attr]
        self.assertEqual(
            first.resolved_skill.to_payload(),  # type: ignore[union-attr]
            second.resolved_skill.to_payload(),  # type: ignore[union-attr]
        )

    def test_preferred_executor_overrides_priority(self) -> None:
        candidate = _candidate(action_type="intent.execute", side_effect_class="external")
        catalog = SkillCatalog(
            executors=[
                _executor_with_intent_skill("slow", priority=50),
                _executor_with_intent_skill("fast", priority=10),
            ]
        )

        result = resolve_skill_for_candidate(
            candidate,
            catalog,
            preferred_executor_ids=("slow",),
        )

        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.resolved_skill.executor_id, "slow")  # type: ignore[union-attr]

    def test_unresolved_when_no_skill_supports_action(self) -> None:
        candidate = _candidate(action_type="task.split", requires_confirmation=False)
        catalog = SkillCatalog()

        result = resolve_skill_for_candidate(candidate, catalog)

        self.assertEqual(result.status, "unresolved")
        self.assertIn("no skill supports action_type task.split", result.unresolved_reasons)
        self.assertEqual(result.considered_skill_ids, [])

    def test_result_round_trip_is_json_serializable(self) -> None:
        candidate = _candidate(action_type="state.record", requires_confirmation=False)
        catalog = SkillCatalog(
            builtin_skills=[
                SkillDescriptor(
                    skill_id="state.record.builtin",
                    source="builtin",
                    supported_action_types=["state.record"],
                    side_effect_class="state_change",
                    requires_confirmation=False,
                    input_schema={"type": "state_record_input.v1"},
                    output_schema={"type": "state_record.v1"},
                    instructions=["Record the observation without external side effects."],
                )
            ]
        )

        result = resolve_skill_for_candidate(candidate, catalog)
        restored = SkillResolutionResult.from_payload(json.loads(json.dumps(result.to_payload())))

        self.assertEqual(restored.status, "resolved")
        self.assertEqual(restored.resolved_skill.skill_id, "state.record.builtin")  # type: ignore[union-attr]
        self.assertEqual(restored.resolved_skill.executor_id, "spice.builtin")  # type: ignore[union-attr]
        self.assertEqual(restored.resolved_skill.instructions, ["Record the observation without external side effects."])  # type: ignore[union-attr]
        self.assertEqual(restored.resolved_skill.input_schema["type"], "state_record_input.v1")  # type: ignore[union-attr]
        self.assertEqual(restored.resolved_skill.output_schema["type"], "state_record.v1")  # type: ignore[union-attr]

    def test_resolver_does_not_mutate_candidate_or_catalog(self) -> None:
        candidate = _candidate(action_type="context.prepare", requires_confirmation=False)
        catalog = SkillCatalog(
            project_skills=[
                SkillDescriptor(
                    skill_id="context.prepare.project",
                    source="project",
                    supported_action_types=["context.prepare"],
                    side_effect_class="read_only",
                    requires_confirmation=False,
                )
            ]
        )
        candidate_before = copy.deepcopy(candidate.to_payload())
        catalog_before = copy.deepcopy(catalog.to_payload())

        resolve_skill_for_candidate(candidate, catalog)

        self.assertEqual(candidate.to_payload(), candidate_before)
        self.assertEqual(catalog.to_payload(), catalog_before)

    def test_resolve_skills_for_candidates_preserves_order(self) -> None:
        candidates = [
            _candidate(action_type="context.prepare", requires_confirmation=False),
            _candidate(action_type="task.split", requires_confirmation=False),
        ]
        catalog = SkillCatalog(
            project_skills=[
                SkillDescriptor(
                    skill_id="context.prepare.project",
                    source="project",
                    supported_action_types=["context.prepare"],
                    side_effect_class="read_only",
                    requires_confirmation=False,
                )
            ]
        )

        results = resolve_skills_for_candidates(candidates, catalog)

        self.assertEqual([item.action_type for item in results], ["context.prepare", "task.split"])
        self.assertEqual([item.status for item in results], ["resolved", "unresolved"])

    def test_candidate_side_effect_normalization(self) -> None:
        self.assertEqual(
            candidate_required_side_effect_class(
                _candidate(action_type="state.record", side_effect_class="none")
            ),
            "state_change",
        )
        self.assertEqual(
            candidate_required_side_effect_class(
                _candidate(action_type="context.prepare", side_effect_class="none")
            ),
            "read_only",
        )
        self.assertEqual(
            candidate_required_side_effect_class(
                _candidate(action_type="intent.execute", side_effect_class="external")
            ),
            "external_effect",
        )


def _candidate(
    *,
    action_type: str,
    required_capability: str = "",
    side_effect_class: str = "none",
    requires_confirmation: bool = True,
) -> GenericCandidate:
    return GenericCandidate(
        candidate_id=f"candidate.{action_type}.sample",
        action_type=action_type,
        intent=f"Handle {action_type}",
        target_refs=["target.sample"],
        required_capability=required_capability,
        estimated_cost=EstimatedCost(attention="low"),
        risk_profile=RiskProfile(level="low"),
        expected_state_delta=ExpectedStateDelta(updates_refs=["target.sample"]),
        execution_boundary=ExecutionBoundary(
            mode="skill_resolution",
            requires_confirmation=requires_confirmation,
            side_effect_class=side_effect_class,
        ),
        side_effect_class=side_effect_class,
        requires_confirmation=requires_confirmation,
        why_available=["unit test candidate"],
    )


def _executor_with_intent_skill(executor_id: str, priority: int) -> ExecutorDescriptor:
    return ExecutorDescriptor(
        executor_id=executor_id,
        priority=priority,
        skills=[
            SkillDescriptor(
                skill_id=f"intent.execute.{executor_id}",
                source="executor",
                supported_action_types=["intent.execute"],
                side_effect_class="external_effect",
                requires_confirmation=True,
            )
        ],
    )


if __name__ == "__main__":
    unittest.main()
