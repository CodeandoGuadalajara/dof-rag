import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from openai.types.chat import ChatCompletionChunk

from agent_tools.agent import OpenAIChatCompletionsBackend


class Stream:
    def __init__(self, chunks):
        self.chunks = chunks
    def __enter__(self):
        return iter(self.chunks)
    def __exit__(self, *args):
        pass


def chunk(delta, *, finish=None, usage=None):
    return ChatCompletionChunk.model_validate({
        'id': 'stream-1', 'object': 'chat.completion.chunk', 'created': 0, 'model': 'qwen',
        'choices': [] if usage else [{'index': 0, 'delta': delta, 'finish_reason': finish}],
        'usage': usage,
    })


class ChatStreamTests(unittest.TestCase):
    def test_stream_without_usage_fails_closed_after_preserving_text(self):
        from human_eval.budget_backend import BudgetBackend, TokenCeilingReached
        create = Mock(return_value=Stream([chunk({'content': 'Partial answer'}, finish='stop')]))
        deltas = []
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        backend = OpenAIChatCompletionsBackend(
            model='qwen', api_key='local', base_url='http://local', client=client,
            on_delta=lambda *args: deltas.append(args),
        )
        guarded = BudgetBackend(backend, 1000, lambda **kwargs: 10)
        with self.assertRaisesRegex(TokenCeilingReached, 'reliable token usage'):
            guarded.create_turn(input_items=[], tools=[], instructions='Chat')
        self.assertEqual(deltas, [('content', 'Partial answer'), ('done', '')])
        self.assertTrue(guarded.blocked)
        with self.assertRaises(TokenCeilingReached):
            guarded.create_turn(input_items=[], tools=[], instructions='Chat')
        create.assert_called_once()

    def test_text_reasoning_tool_fragments_and_usage(self):
        create = Mock(return_value=Stream([
            chunk({'reasoning_content': 'Buscando'}),
            chunk({'content': 'Hola '}), chunk({'content': 'mundo'}),
            chunk({'tool_calls': [{'index': 0, 'id': 'call-1', 'type': 'function',
                                  'function': {'name': 'read_chunks', 'arguments': '{"chunk_ids":'}}]}),
            chunk({'tool_calls': [{'index': 0, 'function': {'arguments': '[4]}'}}]}, finish='tool_calls'),
            chunk({}, usage={'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}),
        ]))
        deltas = []
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        backend = OpenAIChatCompletionsBackend(model='qwen', api_key='local', base_url='http://local',
                                               client=client, reasoning_effort='low', enable_thinking=True,
                                               on_delta=lambda *args: deltas.append(args))
        turn = backend.create_turn(input_items=[{'role': 'user', 'content': 'Hola'}], tools=[], instructions='Chat')
        self.assertEqual(deltas, [('reasoning_content', 'Buscando'), ('content', 'Hola '), ('content', 'mundo'), ('done', '')])
        self.assertEqual(turn.final_text, 'Hola mundo')
        self.assertEqual(turn.output_items[0]['reasoning_content'], 'Buscando')
        self.assertEqual(turn.tool_calls[0].arguments, {'chunk_ids': [4]})
        self.assertEqual(turn.usage['total_tokens'], 15)
        self.assertEqual(create.call_args.kwargs['reasoning_effort'], 'low')
        self.assertEqual(create.call_args.kwargs['extra_body']['chat_template_kwargs'],
                         {'enable_thinking': True, 'reasoning_effort': 'low'})
        self.assertTrue(create.call_args.kwargs['stream'])
        self.assertEqual(create.call_args.kwargs['stream_options'], {'include_usage': True})


if __name__ == '__main__':
    unittest.main()
