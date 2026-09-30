import os
import unittest
from unittest.mock import patch
from human_eval.agent_executor import AgentExecutorConfig


class LocalProviderDefaultTests(unittest.TestCase):
    def test_local_server_is_default_and_can_be_overridden(self):
        with patch.dict(os.environ, {}, clear=True):
            config = AgentExecutorConfig.from_env(".")
            self.assertEqual(config.provider, "llama-server")
            self.assertEqual(config.model, "Qwen3.8-Flash-Next")
            self.assertEqual(config.base_url, "http://192.168.1.117:8888/v1")
        with patch.dict(os.environ, {
            "DOF_AGENT_PROVIDER": "openai-responses", "DOF_AGENT_MODEL": "custom",
            "DOF_AGENT_BASE_URL": "https://example.com/v1",
        }, clear=True):
            config = AgentExecutorConfig.from_env(".")
            self.assertEqual(config.model, "custom")
            self.assertEqual(config.base_url, "https://example.com/v1")


if __name__ == "__main__":
    unittest.main()
