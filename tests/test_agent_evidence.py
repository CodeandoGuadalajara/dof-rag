import json
import unittest
from types import SimpleNamespace

from agent_tools.agent import (
    AgentRunner,
    DofToolbox,
    ModelTurn,
    OpenAIChatCompletionsBackend,
    _coverage_requirements,
    _parse_final_answer,
)
from human_eval.agent_executor import _public_result
from tests.test_agent_tools import (
    ChatCompletionsClient,
    DumpableItem,
    FakeRetriever,
    ScriptedBackend,
)


class EvidenceDisciplineTests(unittest.TestCase):
    def test_thinking_is_history_not_answer_or_public_trace(self):
        draft = '{"answer":"SECRET DRAFT","citations":[4],"premise_status":"supported"}'
        final = (
            '{"answer":"Evidence answer","citations":[4],"premise_status":"supported"}'
        )
        message = DumpableItem(
            role="assistant", content=f"<think>{draft}</think>{final}", tool_calls=[]
        )
        client = ChatCompletionsClient(
            SimpleNamespace(
                id="chat",
                choices=[SimpleNamespace(message=message, finish_reason="stop")],
                usage=None,
            )
        )
        backend = OpenAIChatCompletionsBackend(
            model="local",
            api_key="local",
            base_url="http://localhost",
            client=client,
            enable_thinking=True,
        )
        turn = backend.create_turn(input_items=[], tools=[], instructions="test")
        self.assertEqual(
            client.kwargs["extra_body"]["chat_template_kwargs"],
            {"enable_thinking": True},
        )
        self.assertEqual(turn.final_text, final)
        self.assertEqual(turn.output_items[0]["reasoning_content"], draft)
        self.assertNotIn("SECRET", turn.output_items[0]["content"])
        self.assertEqual(
            _parse_final_answer(message.content, {4}).answer, "Evidence answer"
        )
        with self.assertRaises(ValueError):
            _parse_final_answer(f"<think>{draft}", {4})

        class ReadToolbox(DofToolbox):
            def begin(self, **kwargs):
                super().begin(**kwargs)
                self.read_chunk_ids.add(4)
                self.read_chunk_documents[4] = 2
                self.read_document_ids.add(2)

        events = []
        run = AgentRunner(
            ScriptedBackend([turn]), ReadToolbox(FakeRetriever()), max_model_turns=1
        ).run(
            "A question",
            on_progress=lambda event, data: events.append(data),
        )
        self.assertNotIn("SECRET", json.dumps(_public_result(run.to_dict())))
        self.assertNotIn("SECRET", json.dumps(events))
        self.assertNotIn("SECRET", json.dumps(run.to_dict()))

    def test_truncated_output_is_not_json_error_or_executed_tool(self):
        backend = ScriptedBackend(
            [
                ModelTurn(
                    response_id="cut",
                    output_items=[],
                    final_text='<think>{"answer":"draft"}',
                    finish_reason="length",
                )
            ]
        )
        run = AgentRunner(backend, DofToolbox(FakeRetriever())).run("A question")
        self.assertEqual(run.stop_reason, "output_token_limit")
        self.assertEqual(run.model_turns, 1)
        self.assertEqual(run.answer.citations, [])
        self.assertEqual(run.turns[0].final_text, "")

    def test_reading_evidence_does_not_disable_further_research(self):
        toolbox = DofToolbox(FakeRetriever())
        toolbox.begin(as_of=None)
        toolbox.read_chunk_ids.add(4)
        toolbox.read_document_ids.add(2)
        runner = AgentRunner(ScriptedBackend([]), toolbox)
        self.assertFalse(toolbox.missing_coverage)
        self.assertIn(
            "get_document_outline", [t["name"] for t in runner._available_tools()]
        )
        self.assertIn("read_chunks", [t["name"] for t in runner._available_tools()])

    def test_indicator_coverage_requires_each_indicator_in_read_text(self):
        requirements = _coverage_requirements(
            "¿Qué valores de INPC y UMA publicó el INEGI?"
        )
        self.assertIn("indicador INPC", requirements)
        self.assertIn("indicador UMA", requirements)
        hit = SimpleNamespace(
            text="Índice Nacional de Precios al Consumidor: 143.042",
            path="",
            heading_path=[],
        )
        self.assertTrue(DofToolbox._hit_covers("indicador INPC", hit, set()))
        self.assertFalse(
            DofToolbox._hit_covers("indicador UMA", hit, set(), title="UMA")
        )
        hit.text = "Unidad de Medida y Actualización: 117.31"
        self.assertTrue(DofToolbox._hit_covers("indicador UMA", hit, set()))

    def test_incomplete_final_answer_is_explicitly_partial(self):
        class PartialToolbox(DofToolbox):
            def begin(self, **kwargs):
                super().begin(**kwargs)
                self.read_chunk_ids.add(4)
                self.read_chunk_documents[4] = 2
                self.read_document_ids.add(2)
                self.covered_requirements.add("indicador INPC")

        turn = ModelTurn(
            response_id="partial",
            output_items=[],
            final_text='{"answer":"INPC: 143.042","citations":[4],"premise_status":"supported"}',
        )
        run = AgentRunner(
            ScriptedBackend([turn]), PartialToolbox(FakeRetriever()), max_model_turns=1
        ).run("INPC y UMA")
        self.assertEqual(run.answer.premise_status, "unclear")
        self.assertIn("No se verificó: indicador UMA", run.answer.answer)
        self.assertNotEqual(run.stop_reason, "completed")
