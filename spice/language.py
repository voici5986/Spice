from __future__ import annotations

from typing import Literal


DisplayLanguage = Literal["en", "zh"]


def detect_display_language(text: str) -> DisplayLanguage:
    total_letters = 0
    cjk_letters = 0
    for char in text:
        if char.isspace():
            continue
        codepoint = ord(char)
        is_cjk = (
            0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
        )
        if is_cjk:
            cjk_letters += 1
            total_letters += 1
        elif char.isalpha():
            total_letters += 1
    if total_letters and (cjk_letters / total_letters >= 0.2 or cjk_letters >= 4):
        return "zh"
    return "en"


def language_instruction(display_language: str) -> str:
    if display_language == "zh":
        return (
            "Write all user-facing card copy in Simplified Chinese. "
            "Do not translate IDs, action_type values, enum values, file paths, commands, "
            "API keys, environment variable names, or literal strings."
        )
    return (
        "Write all user-facing card copy in English. "
        "Do not alter IDs, action_type values, enum values, file paths, commands, "
        "API keys, environment variable names, or literal strings."
    )
