import os
import unittest
from unittest.mock import patch

from human_eval.agent_executor import AgentExecutorConfig, AgentRunExecutor


class LocalProviderDefaultTests(unittest.TestCase):
    def test_empty_reasoning_effort_omits_optional_field(self):
        with patch.dict(os.environ, {'DOF_REASONING_EFFORT': ''}, clear=True):
            config = AgentExecutorConfig.from_env('.')
            self.assertIsNone(config.reasoning_effort)
            backend = AgentRunExecutor(config)._backend()
            self.assertIsNone(backend.reasoning_effort)
            self.assertTrue(backend.enable_thinking)

    def test_local_server_is_default_and_can_be_overridden(self):
        with patch.dict(os.environ, {}, clear=True):
            config = AgentExecutorConfig.from_env(".")
            self.assertEqual(config.provider, "llama-server")
            self.assertEqual(config.model, "Qwen3.8-Flash-Next")
            self.assertEqual(config.base_url, "http://127.0.0.1:8080/v1")
            self.assertEqual(config.reasoning_effort, "low")
        with patch.dict(os.environ, {
            "DOF_AGENT_PROVIDER": "openai-responses", "DOF_AGENT_MODEL": "custom",
            "DOF_AGENT_BASE_URL": "https://example.com/v1",
        }, clear=True):
            config = AgentExecutorConfig.from_env(".")
            self.assertEqual(config.model, "custom")
            self.assertEqual(config.base_url, "https://example.com/v1")
            self.assertIsNone(config.reasoning_effort)

    def test_thinking_is_provider_specific_not_model_specific(self):
        with patch.dict(os.environ, {"DOF_AGENT_MODEL": "local-alias"}, clear=True):
            backend = AgentRunExecutor(AgentExecutorConfig.from_env("."))._backend()
            self.assertTrue(backend.enable_thinking)
            self.assertEqual(backend.reasoning_effort, "low")
        with patch.dict(os.environ, {
            "DOF_AGENT_PROVIDER": "openai-responses", "DOF_AGENT_MODEL": "custom",
            "DOF_REASONING_EFFORT": "medium", "OPENAI_API_KEY": "test",
        }, clear=True):
            backend = AgentRunExecutor(AgentExecutorConfig.from_env("."))._backend()
            self.assertEqual(backend.reasoning_effort, "medium")


if __name__ == "__main__":
    unittest.main()
