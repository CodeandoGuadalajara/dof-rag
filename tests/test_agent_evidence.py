import json
import unittest
from types import SimpleNamespace

from agent_tools.agent import (
    AGENT_INSTRUCTIONS,
    AgentRunner,
    DofToolbox,
    ModelTurn,
    OpenAIChatCompletionsBackend,
    ToolCall,
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

    def test_date_roles_and_minimal_answer_policy_reach_the_backend(self):
        # Prompt contract only: this does not prove a real model obeys the policy.
        backend = ScriptedBackend([ModelTurn("empty", [], final_text="{}")])
        AgentRunner(backend, DofToolbox(FakeRetriever()), max_model_turns=1).run(
            "¿Cuál fue el tipo de cambio obtenido el 9 de agosto de 2006?"
        )
        instructions = backend.calls[0]["instructions"]
        self.assertIn("publication_date sólo establece cuándo se publicó", instructions)
        self.assertIn("obtención o periodo del dato", instructions)
        self.assertIn("entrada en vigor (disposición o", instructions)
        self.assertIn("el aviso puede publicarse después", instructions)
        self.assertIn("Cada afirmación debe estar sustentada por los chunks citados", instructions)
        self.assertIn("No calcules una fecha límite", instructions)
        self.assertIn("MAT significa edición matutina", instructions)
        self.assertNotIn("Indica la fecha de publicación de las", AGENT_INSTRUCTIONS)
        for tool in DofToolbox(FakeRetriever()).tool_definitions():
            props = tool["parameters"]["properties"]
            for field in ("date_from", "date_to"):
                if field in props:
                    self.assertIn("publicación", props[field]["description"])
                    self.assertIn("no obtención ni vigencia", props[field]["description"])

    def test_budget_reminder_exposes_pending_read_before_last_tool_turn(self):
        backend = ScriptedBackend([
            ModelTurn("list", [], tool_calls=[ToolCall("1", "list_publications", {
                "as_of": None, "date_from": None, "date_to": None,
                "section": None, "limit": 1,
            })]),
            ModelTurn("outline", [], tool_calls=[ToolCall("2", "get_document_outline", {
                "document_id": 2,
            })]),
            ModelTurn("read", [], tool_calls=[ToolCall("3", "read_chunks", {
                "chunk_ids": [4], "neighbor_window": 0,
            })]),
            ModelTurn("final", [], final_text=json.dumps({
                "answer": "evidencia", "citations": [4], "premise_status": "supported",
            })),
        ])
        run = AgentRunner(backend, DofToolbox(FakeRetriever()), max_model_turns=4).run(
            "Una pregunta"
        )
        self.assertEqual(run.stop_reason, "completed")
        before_read = backend.calls[2]["input_items"][-1]["content"]
        self.assertIn("Turnos restantes antes del cierre obligatorio: 1", before_read)
        self.assertIn("Candidatos no leídos (hasta 8 de 1): [4]", before_read)
        self.assertIn("documentos distintos (mínimo 1)", before_read)
        self.assertIn("puedes citar si sustentan la respuesta: []", before_read)
        final = backend.calls[3]["input_items"][-1]["content"]
        self.assertIn("puedes citar si sustentan la respuesta: [4]", final)
        self.assertIn("Candidatos no leídos (hasta 8 de 0): []", final)
        self.assertEqual(backend.calls[3]["tools"], [])
        for call in backend.calls:
            reminders = [i for i in call["input_items"]
                         if "Turnos restantes antes del cierre" in i.get("content", "")]
            self.assertEqual(len(reminders), 1)

    def test_exhausted_tool_budget_closes_with_explicit_partial_answer(self):
        class PartialToolbox(DofToolbox):
            def begin(self, **kwargs):
                super().begin(**kwargs)
                self.visible_document_ids.add(2)
                self.visible_chunk_ids.add(4)
                self.covered_requirements.add("indicador INPC")

        backend = ScriptedBackend([
            ModelTurn("read", [], tool_calls=[ToolCall("1", "read_chunks", {
                "chunk_ids": [4], "neighbor_window": 0,
            })]),
            ModelTurn("partial", [], final_text=json.dumps({
                "answer": "INPC: 143.042", "citations": [4], "premise_status": "supported",
            })),
        ])
        run = AgentRunner(
            backend, PartialToolbox(FakeRetriever()), max_model_turns=8, max_tool_calls=1
        ).run("INPC y UMA")
        self.assertEqual(run.model_turns, 2)
        self.assertEqual(run.tool_calls, 1)
        self.assertEqual(backend.calls[1]["tools"], [])
        reminder = backend.calls[1]["input_items"][-1]["content"]
        self.assertIn("Llamadas a herramientas restantes: 0", reminder)
        self.assertIn("indicador UMA", reminder)
        self.assertEqual(run.answer.premise_status, "unclear")
        self.assertIn("No se verificó: indicador UMA", run.answer.answer)
        self.assertNotEqual(run.stop_reason, "completed")

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
