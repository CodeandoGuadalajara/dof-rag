import unittest

from agent_tools.agent import DofToolbox, ModelTurn, ToolCall
from agent_tools.models import (
    DocumentHit,
    DocumentSearchResult,
    RetrievalStrategy,
    SearchFilters,
)
from human_eval.chat_runner import chat_instructions, run_chat
from human_eval.contracts import RunRequest
from tests.test_agent_evidence import FakeRetriever, ScriptedBackend


class FakeEmbedder:
    def embed_query(self, query):
        return b"0"


class SearchFakeRetriever(FakeRetriever):
    def __init__(self):
        self.search_calls = []

    def search_documents(self, query, *, strategy, filters=None, query_vector=None,
                         top_k=20, **kwargs):
        self.search_calls.append({"query": query, "strategy": strategy, "top_k": top_k})
        return DocumentSearchResult(
            query=query,
            strategy=RetrievalStrategy(strategy),
            filters=filters or SearchFilters(),
            documents=[DocumentHit(2, "doc.md", "2025-01-01", "MAT", 1.0,
                                   title="Reglas aplicables en 2025")],
        )


class ChatRunnerTests(unittest.TestCase):
    def test_tool_boolean_strings_are_normalized_by_schema(self):
        from agent_tools.agent import _normalize_nullable_literals
        schema = {'properties': {'prefer_recent': {'type': ['boolean', 'null']}, 'query': {'type': 'string'}}}
        self.assertEqual(_normalize_nullable_literals(schema, {'prefer_recent': 'false', 'query': 'true'}),
                         {'prefer_recent': False, 'query': 'true'})
        self.assertEqual(_normalize_nullable_literals(schema, {'prefer_recent': 'TRUE'})['prefer_recent'], True)
        self.assertEqual(_normalize_nullable_literals(schema, {'prefer_recent': 'yes'})['prefer_recent'], 'yes')

    def test_tool_nullable_integer_strings_are_normalized(self):
        # TensorFold stringifies values of union-typed fields (probe: var/flash-next-schema-probe.py).
        from agent_tools.agent import _normalize_nullable_literals
        schema = {'properties': {'n': {'type': ['integer', 'null']}, 'plain': {'type': 'integer'},
                                 'f': {'type': ['number', 'null']}, 's': {'type': 'string'}}}
        out = _normalize_nullable_literals(schema, {'n': '42', 'plain': '7', 'f': '3.5', 's': '42'})
        self.assertEqual(out, {'n': 42, 'plain': 7, 'f': 3.5, 's': '42'})
        self.assertEqual(_normalize_nullable_literals(schema, {'n': 'cuatro'})['n'], 'cuatro')

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

    def _search_backend(self):
        return ScriptedBackend([
            ModelTurn('search', [], tool_calls=[ToolCall('1', 'search_documents', {'query': 'apoyos'})]),
            ModelTurn('reply', [], final_text='Listo.'),
        ])

    def test_search_documents_defaults_to_eight_candidates_and_configured_strategy(self):
        retriever = SearchFakeRetriever()
        backend = self._search_backend()
        result = run_chat(backend, DofToolbox(retriever, embedder=FakeEmbedder()),
                          RunRequest('Apoyos'), retrieval_strategy='hybrid')
        self.assertEqual(retriever.search_calls[0]['top_k'], 8)
        self.assertEqual(retriever.search_calls[0]['strategy'], 'hybrid')
        trace = result['traces'][0]
        self.assertEqual(trace['arguments']['top_k'], 8)
        self.assertEqual(trace['arguments']['strategy'], 'hybrid')

    def test_configured_hybrid_falls_back_to_lexical_without_embedder(self):
        # Never label a search hybrid when no vector index/embedder is live.
        retriever = SearchFakeRetriever()
        result = run_chat(self._search_backend(), DofToolbox(retriever),
                          RunRequest('Apoyos'), retrieval_strategy='hybrid')
        self.assertEqual(retriever.search_calls[0]['strategy'], 'lexical')
        self.assertEqual(result['traces'][0]['arguments']['strategy'], 'lexical')

    def test_optional_strategy_and_top_k_overrides_are_honored(self):
        retriever = SearchFakeRetriever()
        backend = ScriptedBackend([
            ModelTurn('search', [], tool_calls=[
                ToolCall('1', 'search_documents', {'query': 'apoyos', 'top_k': 2})]),
            ModelTurn('reply', [], final_text='Listo.'),
        ])
        result = run_chat(backend, DofToolbox(retriever), RunRequest('Apoyos'))
        self.assertEqual(retriever.search_calls[0]['top_k'], 2)
        self.assertEqual(result['traces'][0]['arguments']['top_k'], 2)

    def test_invalid_strategy_override_is_rejected_and_the_loop_continues(self):
        retriever = SearchFakeRetriever()
        backend = ScriptedBackend([
            ModelTurn('search', [], tool_calls=[
                ToolCall('1', 'search_documents', {'query': 'apoyos', 'strategy': 'hybrid'})]),
            ModelTurn('reply', [], final_text='Sin búsqueda vectorial disponible.'),
        ])
        result = run_chat(backend, DofToolbox(retriever), RunRequest('Apoyos'))
        self.assertEqual(retriever.search_calls, [])
        self.assertFalse(result['traces'][0]['output']['ok'])
        self.assertEqual(result['traces'][0]['output']['error']['type'], 'invalid_arguments')
        self.assertEqual(result['stop_reason'], 'completed')

    def test_strategy_and_top_k_are_exposed_as_optional_parameters(self):
        backend = ScriptedBackend([ModelTurn('reply', [], final_text='Hola.')])
        run_chat(backend, DofToolbox(FakeRetriever()), RunRequest('Hola'))
        tools = {tool['name']: tool for tool in backend.calls[0]['tools']}
        search = tools['search_documents']['parameters']
        self.assertIn('strategy', search['properties'])
        self.assertIn('top_k', search['properties'])
        self.assertEqual(search['required'], ['query'])
        self.assertNotIn('limit', tools['list_publications']['parameters']['properties'])
        self.assertNotIn('neighbor_window', tools['read_chunks']['parameters']['properties'])

    def test_instructions_include_today_and_user_cutoff(self):
        backend = ScriptedBackend([ModelTurn('reply', [], final_text='Hola.')])
        run_chat(backend, DofToolbox(FakeRetriever()),
                 RunRequest('Hola', as_of='2025-12-31'), today='2026-02-10')
        instructions = backend.calls[0]['instructions']
        self.assertIn('La fecha de hoy es 2026-02-10', instructions)
        self.assertIn('hasta el 2025-12-31', instructions)
        self.assertIn('hipótesis de búsqueda, no evidencia', instructions)

    def test_instructions_without_cutoff_omit_cutoff_line(self):
        instructions = chat_instructions(today='2026-02-10')
        self.assertIn('La fecha de hoy es 2026-02-10', instructions)
        self.assertNotIn('corte explícito', instructions)


if __name__ == '__main__':
    unittest.main()
