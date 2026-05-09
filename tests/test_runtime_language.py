from __future__ import annotations

import unittest

from spice.language import detect_display_language, language_instruction


class RuntimeLanguageTests(unittest.TestCase):
    def test_detect_display_language_recognizes_chinese_intent_with_paths(self) -> None:
        text = (
            "在 .spice-smoke/hermes_executor_smoke.txt 中添加一个简短的烟雾提示文件，"
            "内容为：SPICE_HERMES_EXECUTOR_OK。请勿修改任何其他文件。"
        )

        self.assertEqual(detect_display_language(text), "zh")

    def test_detect_display_language_defaults_to_english(self) -> None:
        self.assertEqual(
            detect_display_language(
                "Add .spice-smoke/codex_executor_smoke.txt with exact text OK."
            ),
            "en",
        )

    def test_language_instruction_keeps_protocol_values_literal(self) -> None:
        instruction = language_instruction("zh")

        self.assertIn("Simplified Chinese", instruction)
        self.assertIn("Do not translate IDs", instruction)


if __name__ == "__main__":
    unittest.main()
