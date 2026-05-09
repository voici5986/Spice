from __future__ import annotations

import json
import unittest

from spice.executors import (
    CapabilityDescriptor,
    ExecutorDescriptor,
    ResolvedSkill,
    SkillCatalog,
    SkillDescriptor,
    builtin_fallback_skill_catalog,
)


class ExecutorSkillCatalogTests(unittest.TestCase):
    def test_skill_descriptor_round_trip_ignores_unknown_fields(self) -> None:
        skill = SkillDescriptor.from_payload(
            {
                "skill_id": "repo.triage.read_only",
                "display_name": "Repo triage",
                "source": "user",
                "supported_action_types": ["item.triage", "context.prepare"],
                "required_capabilities": ["repo_read"],
                "side_effect_class": "read_only",
                "requires_confirmation": False,
                "input_schema": {"type": "context_pack.v1"},
                "output_schema": {"type": "triage_report.v1"},
                "instructions": ["Do not edit files."],
                "metadata": {"path": ".spice/skills/repo_triage.json"},
                "newer_snapshot_field": "ignored",
            }
        )

        payload = skill.to_payload()
        restored = SkillDescriptor.from_payload(json.loads(json.dumps(payload)))

        self.assertEqual(restored.skill_id, "repo.triage.read_only")
        self.assertEqual(restored.source, "user")
        self.assertEqual(restored.supported_action_types, ["item.triage", "context.prepare"])
        self.assertEqual(restored.required_capabilities, ["repo_read"])
        self.assertEqual(restored.side_effect_class, "read_only")
        self.assertFalse(restored.requires_confirmation)
        self.assertNotIn("newer_snapshot_field", restored.to_payload())

    def test_executor_descriptor_declares_capabilities_and_skills(self) -> None:
        executor = ExecutorDescriptor(
            executor_id="codex",
            display_name="Codex",
            adapter_type="sdep-wrapper",
            capabilities=[
                CapabilityDescriptor(
                    capability_id="repo_read",
                    display_name="Repository read",
                    side_effect_classes=["read_only"],
                ),
                CapabilityDescriptor(
                    capability_id="repo_write",
                    display_name="Repository write",
                    side_effect_classes=["state_change"],
                ),
            ],
            skills=[
                SkillDescriptor(
                    skill_id="repo.triage.read_only",
                    supported_action_types=["item.triage"],
                    required_capabilities=["repo_read"],
                    side_effect_class="read_only",
                    requires_confirmation=False,
                )
            ],
        )
        executor.validate()

        payload = executor.to_payload()
        restored = ExecutorDescriptor.from_payload(payload)

        self.assertEqual(restored.executor_id, "codex")
        self.assertEqual(restored.capability_ids(), ["repo_read", "repo_write"])
        self.assertEqual(restored.skill_ids(), ["repo.triage.read_only"])

    def test_skill_catalog_orders_user_project_executor_then_builtin_skills(self) -> None:
        catalog = SkillCatalog(
            executors=[
                ExecutorDescriptor(
                    executor_id="codex",
                    skills=[
                        SkillDescriptor(
                            skill_id="executor.skill",
                            source="executor",
                            supported_action_types=["item.triage"],
                        )
                    ],
                )
            ],
            builtin_skills=[
                SkillDescriptor(
                    skill_id="builtin.skill",
                    source="builtin",
                    supported_action_types=["item.triage"],
                )
            ],
            user_skills=[
                SkillDescriptor(
                    skill_id="user.skill",
                    source="user",
                    supported_action_types=["item.triage"],
                )
            ],
            project_skills=[
                SkillDescriptor(
                    skill_id="project.skill",
                    source="project",
                    supported_action_types=["item.triage"],
                )
            ],
        )
        catalog.validate()

        self.assertEqual(
            [skill.skill_id for skill in catalog.all_skills()],
            ["user.skill", "project.skill", "executor.skill", "builtin.skill"],
        )
        self.assertEqual(
            [skill.skill_id for skill in catalog.find_skills_for_action("item.triage")],
            ["user.skill", "project.skill", "executor.skill", "builtin.skill"],
        )
        self.assertEqual(catalog.find_executor("codex").executor_id, "codex")  # type: ignore[union-attr]
        self.assertIsNone(catalog.find_executor("missing"))

    def test_catalog_round_trip_is_json_serializable(self) -> None:
        catalog = SkillCatalog(
            executors=[
                ExecutorDescriptor(
                    executor_id="hermes",
                    display_name="Hermes",
                    capabilities=[
                        CapabilityDescriptor(
                            capability_id="work_item_triage",
                            side_effect_classes=["read_only"],
                        )
                    ],
                    skills=[
                        SkillDescriptor(
                            skill_id="work_item.triage.hermes",
                            supported_action_types=["item.triage"],
                            required_capabilities=["work_item_triage"],
                            side_effect_class="read_only",
                            requires_confirmation=False,
                        )
                    ],
                )
            ],
            metadata={"source": "unit-test"},
        )

        payload = json.loads(json.dumps(catalog.to_payload()))
        restored = SkillCatalog.from_payload(payload)

        self.assertEqual(restored.executors[0].executor_id, "hermes")
        self.assertEqual(restored.executors[0].skills[0].skill_id, "work_item.triage.hermes")
        self.assertEqual(restored.metadata["source"], "unit-test")

    def test_invalid_side_effect_class_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "side_effect_class"):
            SkillDescriptor(
                skill_id="bad.skill",
                supported_action_types=["item.triage"],
                side_effect_class="shell_admin",
            ).validate()

    def test_skill_descriptor_requires_stable_action_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "supported_action_types"):
            SkillDescriptor(
                skill_id="bad.skill",
                supported_action_types=[],
            ).validate()

    def test_builtin_fallback_catalog_covers_minimal_general_actions(self) -> None:
        catalog = builtin_fallback_skill_catalog()

        self.assertTrue(catalog.find_skills_for_action("state.record"))
        self.assertTrue(catalog.find_skills_for_action("user.clarify"))
        self.assertTrue(catalog.find_skills_for_action("item.triage"))
        self.assertTrue(catalog.find_skills_for_action("intent.execute"))

    def test_resolved_skill_is_data_only_and_round_trips(self) -> None:
        resolved = ResolvedSkill(
            executor_id="codex",
            skill_id="repo.triage.read_only",
            action_type="item.triage",
            capability_id="repo_read",
            side_effect_class="read_only",
            requires_confirmation=False,
            resolution_reason="matched action_type and capability",
            confidence=1.0,
        )
        resolved.validate()

        restored = ResolvedSkill.from_payload(json.loads(json.dumps(resolved.to_payload())))

        self.assertEqual(restored.executor_id, "codex")
        self.assertEqual(restored.skill_id, "repo.triage.read_only")
        self.assertEqual(restored.action_type, "item.triage")
        self.assertFalse(restored.requires_confirmation)


if __name__ == "__main__":
    unittest.main()
