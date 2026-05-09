from __future__ import annotations

import unittest

from spice.llm.core.runtime import resolve_llm_runtime


class LLMRuntimeResolverTests(unittest.TestCase):
    def test_resolves_openai_runtime(self) -> None:
        runtime = resolve_llm_runtime(
            provider_id="openai",
            model_id="gpt-4o-mini",
            env={"OPENAI_API_KEY": "sk-test"},
        )

        self.assertEqual(runtime.provider_id, "openai")
        self.assertEqual(runtime.provider_family, "openai_compatible")
        self.assertEqual(runtime.api_mode, "chat_completions")
        self.assertEqual(runtime.model_id, "gpt-4o-mini")
        self.assertEqual(runtime.base_url, "https://api.openai.com/v1")
        self.assertEqual(runtime.base_url_source, "default")
        self.assertEqual(runtime.api_key_env, "OPENAI_API_KEY")
        self.assertTrue(runtime.api_key_present)
        self.assertEqual(runtime.status, "ready")

    def test_resolves_mimo_xiaomi_alias_and_credentials(self) -> None:
        runtime = resolve_llm_runtime(
            provider_id="xiaomi",
            model_id="mimo-v2.5-pro",
            env={
                "XIAOMI_API_KEY": "key",
                "XIAOMI_BASE_URL": "https://gateway.example/v1",
            },
        )

        self.assertEqual(runtime.requested_provider_id, "xiaomi")
        self.assertEqual(runtime.provider_id, "mimo")
        self.assertEqual(runtime.provider_family, "openai_compatible")
        self.assertEqual(runtime.api_mode, "chat_completions")
        self.assertEqual(runtime.base_url, "https://gateway.example/v1")
        self.assertEqual(runtime.base_url_env, "XIAOMI_BASE_URL")
        self.assertEqual(runtime.api_key_env, "XIAOMI_API_KEY")
        self.assertTrue(runtime.api_key_present)
        self.assertEqual(runtime.metadata["vendor"], "xiaomi")
        self.assertEqual(runtime.status, "ready")

    def test_resolves_mimo_legacy_aliases(self) -> None:
        runtime = resolve_llm_runtime(
            provider_id="mimo",
            model_id="mimo-v2.5-pro",
            env={
                "MIMO_API_KEY": "key",
                "SPICE_MIMO_BASE_URL": "https://legacy.example/v1",
            },
        )

        self.assertEqual(runtime.provider_id, "mimo")
        self.assertEqual(runtime.api_key_env, "MIMO_API_KEY")
        self.assertTrue(runtime.api_key_present)
        self.assertEqual(runtime.base_url, "https://legacy.example/v1")
        self.assertEqual(runtime.base_url_env, "SPICE_MIMO_BASE_URL")
        self.assertEqual(runtime.status, "ready")

    def test_configured_api_key_env_takes_precedence(self) -> None:
        runtime = resolve_llm_runtime(
            provider_id="mimo",
            model_id="mimo-v2.5-pro",
            configured_api_key_env="CUSTOM_MIMO_KEY",
            env={"CUSTOM_MIMO_KEY": "key", "XIAOMI_API_KEY": "other"},
        )

        self.assertEqual(runtime.api_key_env, "CUSTOM_MIMO_KEY")
        self.assertTrue(runtime.api_key_present)
        self.assertEqual(runtime.status, "ready")

    def test_warns_when_external_provider_key_missing(self) -> None:
        runtime = resolve_llm_runtime(
            provider_id="deepseek",
            model_id="deepseek-chat",
            env={},
        )

        self.assertEqual(runtime.provider_id, "deepseek")
        self.assertEqual(runtime.api_key_env, "DEEPSEEK_API_KEY")
        self.assertFalse(runtime.api_key_present)
        self.assertEqual(runtime.status, "warning")
        self.assertIn("DEEPSEEK_API_KEY", runtime.detail)

    def test_deterministic_runtime_needs_no_auth(self) -> None:
        runtime = resolve_llm_runtime(provider_id="deterministic", env={})

        self.assertEqual(runtime.provider_id, "deterministic")
        self.assertEqual(runtime.api_mode, "deterministic")
        self.assertFalse(runtime.auth_required)
        self.assertFalse(runtime.api_key_present)
        self.assertEqual(runtime.status, "ready")

    def test_subprocess_runtime_requires_command_model_id(self) -> None:
        missing = resolve_llm_runtime(provider_id="subprocess", env={})
        ready = resolve_llm_runtime(
            provider_id="subprocess",
            model_id="python -m my_llm",
            env={},
        )

        self.assertEqual(missing.status, "warning")
        self.assertEqual(ready.status, "ready")
        self.assertEqual(ready.api_mode, "subprocess")

    def test_unsupported_provider_is_structured(self) -> None:
        runtime = resolve_llm_runtime(provider_id="unknown", model_id="x", env={})

        self.assertEqual(runtime.status, "unsupported")
        self.assertEqual(runtime.provider_family, "unknown")
        self.assertEqual(runtime.api_mode, "unknown")
        self.assertFalse(runtime.api_key_present)

    def test_payload_is_json_safe_and_does_not_include_secret(self) -> None:
        runtime = resolve_llm_runtime(
            provider_id="openai",
            model_id="gpt-4o-mini",
            env={"OPENAI_API_KEY": "secret-value"},
        )
        payload = runtime.to_payload()

        self.assertTrue(payload["api_key_present"])
        self.assertEqual(payload["api_key_env"], "OPENAI_API_KEY")
        self.assertNotIn("secret-value", str(payload))


if __name__ == "__main__":
    unittest.main()
