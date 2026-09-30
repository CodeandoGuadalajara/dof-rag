import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from human_eval.agent_executor import AgentExecutorConfig, AgentRunExecutor
from human_eval.budget_backend import TokenCeilingReached
from human_eval.contracts import RunRequest
from human_eval.service import PublicExecutionError


class PartialFlushTests(unittest.TestCase):
    def test_last_buffer_is_persisted_when_generation_raises(self):
        config = AgentExecutorConfig(repo_root=Path('.'), provider='llama-server',
            model='Qwen3.8-Flash-Next', corpus_db=Path('unused'), chunks_db=Path('unused'))
        executor = AgentRunExecutor(config)
        backend = SimpleNamespace(enable_thinking=True, reasoning_effort='low')
        events = []
        def interrupted(*args, **kwargs):
            backend.on_delta('content', 'Último fragmento')
            raise TokenCeilingReached('budget exhausted', used=7)
        with patch.object(executor, '_backend', return_value=backend), \
             patch('human_eval.agent_executor.monotonic', return_value=0), \
             patch('human_eval.qwen_token_counter.QwenTokenCounter'), \
             patch('human_eval.agent_executor.DofRetriever'), \
             patch('human_eval.chat_runner.run_chat', side_effect=interrupted):
            with self.assertRaises(PublicExecutionError):
                executor.execute(RunRequest('Pregunta', token_limit=10),
                    on_progress=lambda kind, payload: events.append((kind, payload)))
        self.assertEqual(events[0][1]['text'], 'Último fragmento')
        self.assertEqual(events[0][1]['chat_delta'], 'content')


if __name__ == '__main__':
    unittest.main()
