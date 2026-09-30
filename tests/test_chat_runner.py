import unittest

from agent_tools.agent import DofToolbox, ModelTurn, ToolCall
from human_eval.chat_runner import run_chat
from human_eval.contracts import RunRequest
from tests.test_agent_evidence import FakeRetriever, ScriptedBackend


class ChatRunnerTests(unittest.TestCase):
    def test_tool_boolean_strings_are_normalized_by_schema(self):
        from agent_tools.agent import _normalize_nullable_literals
        schema = {'properties': {'prefer_recent': {'type': ['boolean', 'null']}, 'query': {'type': 'string'}}}
        self.assertEqual(_normalize_nullable_literals(schema, {'prefer_recent': 'false', 'query': 'true'}),
                         {'prefer_recent': False, 'query': 'true'})
        self.assertEqual(_normalize_nullable_literals(schema, {'prefer_recent': 'TRUE'})['prefer_recent'], True)
        self.assertEqual(_normalize_nullable_literals(schema, {'prefer_recent': 'yes'})['prefer_recent'], 'yes')

    def test_clarification_needs_no_tools_or_json(self):
        backend = ScriptedBackend([ModelTurn('reply', [], final_text='¿Qué decreto te interesa?')])
        result = run_chat(backend, DofToolbox(FakeRetriever()), RunRequest('Hola, ayúdame'))
        from human_eval.agent_executor import _public_result
        self.assertEqual(_public_result(result)['answer']['text'], '¿Qué decreto te interesa?')
        self.assertEqual(result['answer']['answer'], '¿Qué decreto te interesa?')
        self.assertEqual(result['reasoning'], [])
        self.assertEqual(result['stop_reason'], 'completed')
        self.assertEqual(result['answer']['citations'], [])
        self.assertNotIn('Cobertura obligatoria', backend.calls[0]['input_items'][-1]['content'])

    def test_tool_loop_and_citation_validation(self):
        toolbox = DofToolbox(FakeRetriever())
        backend = ScriptedBackend([
            ModelTurn('outline', [], tool_calls=[ToolCall('1', 'get_document_outline', {'document_id': 2})]),
            ModelTurn('read', [], tool_calls=[ToolCall('2', 'read_chunks', {'chunk_ids': [4], 'neighbor_window': 0})]),
            ModelTurn('reply', [], final_text='Aquí está la evidencia [chunk 4]. Otra [chunk 999].'),
        ])
        # Discover the candidate through the same public retrieval path first.
        original_begin = toolbox.begin
        def begin(**kwargs):
            original_begin(**kwargs)
            toolbox.visible_document_ids.add(2)
        toolbox.begin = begin
        result = run_chat(backend, toolbox, RunRequest('Explica el decreto'))
        self.assertEqual(result['tool_calls'], 2)
        self.assertEqual(result['answer']['citations'], [4])
        self.assertIn('[cita no verificada]', result['answer']['answer'])
        self.assertTrue(any(item.get('type') == 'function_call_output' for item in backend.calls[-1]['input_items']))

    def test_raw_tool_markup_is_not_shown_as_a_reply(self):
        backend = ScriptedBackend([ModelTurn('reply', [], final_text='<tool_call>bad</tool_call>')])
        result = run_chat(backend, DofToolbox(FakeRetriever()), RunRequest('Una pregunta'))
        self.assertNotIn('<tool_call>', result['answer']['answer'])
        self.assertEqual(result['stop_reason'], 'unparsed_tool_call')
        self.assertTrue(backend.calls[0]['tools'])

    def test_tools_remain_available_beyond_evaluation_limits(self):
        backend = ScriptedBackend([
            *[ModelTurn(str(n), [], tool_calls=[
                ToolCall(str(n), 'get_document_outline', {'document_id': 2})
            ]) for n in range(10)],
            ModelTurn('reply', [], final_text='Respuesta libre.'),
        ])
        result = run_chat(backend, DofToolbox(FakeRetriever()), RunRequest('Investiga'))
        self.assertEqual(result['model_turns'], 11)
        self.assertEqual(result['tool_calls'], 10)
        self.assertEqual(result['stop_reason'], 'completed')
        self.assertEqual(result['answer']['answer'], 'Respuesta libre.')
        for call in backend.calls:
            self.assertTrue(call['tools'])
            self.assertNotIn('No quedan herramientas', call['instructions'])

    def test_history_is_preserved(self):
        history = ({'role': 'user', 'content': 'Un decreto'}, {'role': 'assistant', 'content': '¿De qué año?'})
        backend = ScriptedBackend([ModelTurn('reply', [], final_text='Gracias por aclararlo.')])
        run_chat(backend, DofToolbox(FakeRetriever()), RunRequest('De 2006', history=history))
        self.assertEqual(backend.calls[0]['input_items'][:2], list(history))


if __name__ == '__main__':
    unittest.main()
