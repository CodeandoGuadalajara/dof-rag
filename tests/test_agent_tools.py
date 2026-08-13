import unittest
from types import SimpleNamespace

from agent_tools.agent import (
    AgentRunner,
    DofToolbox,
    ModelTurn,
    OpenAIChatCompletionsBackend,
    OpenAIResponsesBackend,
    ToolCall,
    _comparison_years,
    _coverage_requirements,
)
from agent_tools.headers import extract_document_header
from agent_tools.llm import _parse_json, answer_with_context
from agent_tools.models import (
    DocumentOutline,
    EvidenceHit,
    IndexVersions,
    OutlineChunk,
    PublicationHit,
    SearchFilters,
    SearchResult,
)
from agent_tools.retrieval import (
    _bm25_chunk_scores,
    _fuse_documents,
    _normative_title_boost,
    _rrf,
)
from scripts.eval_v4_agent import calculate_metrics, fatal_provider_error


class FakeClient:
    def complete(self, system, user, *, max_tokens=1200):
        return '{"answer":"ok","citations":[4,999],"premise_status":"supported"}'


class FakeRetriever:
    versions = IndexVersions("corpus", "chunks", True)

    def list_publications(self, filters, *, limit=50):
        return [
            PublicationHit(
                2,
                "doc.md",
                "2025-01-01",
                "MAT",
                title="Resolución aplicable en 2025",
                institution="Institución",
            )
        ]

    def get_document_outline(self, document_id):
        return DocumentOutline(
            document_id=document_id,
            path="doc.md",
            publication_date="2025-01-01",
            section="MAT",
            chunks=[OutlineChunk(4, 0, [], 10)],
        )

    def read_chunks(self, chunk_ids, *, neighbor_window=0):
        return [
            EvidenceHit(
                chunk_id=chunk_id,
                document_id=2,
                path="doc.md",
                publication_date="2025-01-01",
                section="MAT",
                chunk_index=0,
                heading_path=[],
                text="evidencia",
                score=0.0,
                source="read",
                rank=1,
            )
            for chunk_id in chunk_ids
        ]


class ScriptedBackend:
    model = "scripted"

    def __init__(self, turns):
        self.turns = iter(turns)
        self.calls = []

    def create_turn(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.turns)


class DumpableItem(SimpleNamespace):
    def model_dump(self, **kwargs):
        return vars(self)


class ResponsesClient:
    def __init__(self, response):
        self.response = response
        self.responses = self
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


class ChatCompletionsClient:
    def __init__(self, response):
        self.response = response
        self.chat = SimpleNamespace(completions=self)
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


class QuotaError(Exception):
    code = "credit_balance_exhausted"


class AgentToolsTests(unittest.TestCase):
    def test_header_extraction_separates_institution_and_title(self):
        header = extract_document_header(
            "# SECRETARIA DEL TRABAJO\n\n"
            "## NORMA Oficial Mexicana NOM-035-STPS-2018.\n\nTexto"
        )
        self.assertEqual(header.institution, "SECRETARIA DEL TRABAJO")
        self.assertEqual(header.title, "NORMA Oficial Mexicana NOM-035-STPS-2018")

    def test_normative_title_boost_prefers_the_issuing_norm(self):
        query = "NOM-035 segundo transitorio numeral 5.2 centros de trabajo"
        source = _normative_title_boost(
            query,
            "NORMA Oficial Mexicana NOM-035-STPS-2018, Factores de riesgo "
            "psicosocial en el trabajo",
        )
        reference = _normative_title_boost(
            query,
            "CONVOCATORIA sobre normas oficiales mexicanas de seguridad",
        )
        self.assertGreater(source, reference)

    def test_comparison_years_are_explicit_coverage_requirements(self):
        self.assertEqual(
            _comparison_years("¿Cómo cambiaron los salarios de 2025 a 2026?"),
            ["2025", "2026"],
        )
        self.assertEqual(_comparison_years("¿Qué rige en 2026?"), [])
        self.assertEqual(
            _coverage_requirements(
                "¿Qué dice el segundo transitorio sobre el numeral 5.2?"
            ),
            ["transitorio", "numeral 5.2"],
        )

    def test_exact_provision_heading_beats_a_reference_to_it(self):
        scores = _bm25_chunk_scores(
            "5.2",
            [
                "El segundo transitorio menciona los numerales 5.2 y 5.3.",
                "**5.2** Identificar y analizar los factores de riesgo.",
            ],
        )
        self.assertGreater(scores[1], scores[0])

    def test_read_chunks_reports_missing_comparison_coverage(self):
        toolbox = DofToolbox(FakeRetriever())
        toolbox.begin(as_of="2026-01-01", coverage_requirements=["2025", "2026"])
        listed = toolbox.call(
            "list_publications",
            {
                "as_of": "2026-01-01",
                "date_from": None,
                "date_to": None,
                "section": None,
                "limit": 5,
            },
        )
        self.assertTrue(listed["ok"])
        toolbox.visible_chunk_ids.add(4)
        read = toolbox.call("read_chunks", {"chunk_ids": [4], "neighbor_window": 0})
        self.assertEqual(read["data"]["coverage"], {"2025": True, "2026": False})
        self.assertEqual(toolbox.missing_coverage, ["2026"])

    def test_metrics_do_not_count_limited_run_as_completed(self):
        metrics = calculate_metrics(
            [{"run": {"stop_reason": "model_turn_limit: no final answer"}}]
        )
        self.assertEqual(metrics["runs"], 1)
        self.assertEqual(metrics["completed"], 0)

    def test_metrics_report_explicit_comparison_coverage(self):
        metrics = calculate_metrics(
            [
                {
                    "category": "multi_document",
                    "gold_documents": [{"evidence": [{"chunk_id": 4}]}],
                    "run": {
                        "stop_reason": "completed",
                        "answer": {"citations": []},
                        "traces": [],
                        "usage": {},
                        "tool_calls": 2,
                        "model_turns": 3,
                        "elapsed_ms": 1.0,
                        "coverage": {"2025": True, "2026": True},
                    },
                }
            ]
        )
        self.assertEqual(metrics["coverage_completion_rate"], 1.0)

    def test_fatal_provider_error_distinguishes_quota_from_transient_rate_limit(self):
        self.assertTrue(fatal_provider_error(QuotaError("insufficient_quota")))
        self.assertFalse(fatal_provider_error(Exception("temporary rate limit")))

    def test_weighted_fusion_prefers_lexical_when_weight_is_high(self):
        fused = _fuse_documents([(1, 10.0), (2, 1.0)], [(2, 0.9), (1, 0.1)], 0.75)
        self.assertEqual(fused[0][0], 1)

    def test_rrf_keeps_items_from_both_lists(self):
        self.assertEqual(_rrf([[1, 2], [3, 2]])[0], 2)

    def test_parse_json_allows_fenced_model_output(self):
        data = _parse_json('```json\n{"answer":"ok","citations":[4]}\n```')
        self.assertEqual(data, {"answer": "ok", "citations": [4]})

    def test_parse_json_returns_first_complete_object(self):
        data = _parse_json('before {"answer":"first"} after {"answer":"second"}')
        self.assertEqual(data, {"answer": "first"})

    def test_filters_validate_dates_and_normalize_section(self):
        filters = SearchFilters(
            date_from="2026-01-01", date_to="2026-01-31", section=" mat "
        )
        self.assertEqual(filters.section, "MAT")
        with self.assertRaises(ValueError):
            SearchFilters(date_from="2026-02-01", date_to="2026-01-01")

    def test_bounded_bm25_does_not_reward_irrelevant_length(self):
        scores = _bm25_chunk_scores(
            "agua", ["agua potable", "agua " + "relleno " * 100]
        )
        self.assertGreater(scores[0], scores[1])

    def test_answer_rejects_citations_outside_supplied_context(self):
        result = SearchResult(
            query="pregunta",
            as_of="2026-01-01",
            evidence=[
                EvidenceHit(
                    chunk_id=4,
                    document_id=2,
                    path="doc.md",
                    publication_date="2025-01-01",
                    section="MAT",
                    chunk_index=0,
                    heading_path=[],
                    text="evidencia",
                    score=1.0,
                    source="test",
                    rank=1,
                )
            ],
        )
        answer = answer_with_context(FakeClient(), "pregunta", result)
        self.assertEqual(answer.citations, [4])
        self.assertEqual(answer.invalid_citations, [999])

    def test_tool_schemas_are_strict_and_hide_unavailable_vector_search(self):
        toolbox = DofToolbox(FakeRetriever())
        for tool in toolbox.tool_definitions():
            schema = tool["parameters"]
            self.assertTrue(tool["strict"])
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(set(schema["properties"]), set(schema["required"]))
        search = next(
            tool
            for tool in toolbox.tool_definitions()
            if tool["name"] == "search_documents"
        )
        self.assertEqual(
            search["parameters"]["properties"]["strategy"]["enum"], ["lexical"]
        )

    def test_toolbox_enforces_evaluation_cutoff(self):
        toolbox = DofToolbox(FakeRetriever())
        toolbox.begin(as_of="2026-01-01")
        output = toolbox.call(
            "list_publications",
            {
                "as_of": "2026-02-01",
                "date_from": None,
                "date_to": None,
                "section": None,
                "limit": 5,
            },
        )
        self.assertFalse(output["ok"])
        self.assertIn("exceeds the run cutoff", output["error"]["message"])

    def test_agent_only_accepts_citations_from_read_chunks(self):
        backend = ScriptedBackend(
            [
                ModelTurn(
                    response_id="one",
                    output_items=[],
                    tool_calls=[
                        ToolCall(
                            call_id="call-list",
                            name="list_publications",
                            arguments={
                                "as_of": None,
                                "date_from": "2025-01-01",
                                "date_to": "2025-01-01",
                                "section": "MAT",
                                "limit": 5,
                            },
                        )
                    ],
                ),
                ModelTurn(
                    response_id="two",
                    output_items=[],
                    tool_calls=[
                        ToolCall(
                            call_id="call-0",
                            name="get_document_outline",
                            arguments={"document_id": 2},
                        )
                    ],
                ),
                ModelTurn(
                    response_id="three",
                    output_items=[],
                    tool_calls=[
                        ToolCall(
                            call_id="call-1",
                            name="read_chunks",
                            arguments={"chunk_ids": [4], "neighbor_window": 0},
                        )
                    ],
                ),
                ModelTurn(
                    response_id="four",
                    output_items=[],
                    final_text=(
                        '```json\n{"answer":"ok","citations":[4,999],'
                        '"premise_status":"supported"}\n```'
                    ),
                    usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                ),
            ]
        )
        run = AgentRunner(backend, DofToolbox(FakeRetriever())).run(
            "pregunta",
            as_of="2026-01-01",
        )
        self.assertEqual(run.answer.citations, [4])
        self.assertEqual(run.answer.invalid_citations, [999])
        self.assertEqual(run.tool_calls, 3)
        self.assertEqual(run.stop_reason, "completed")
        self.assertEqual(backend.calls[-1]["tools"], [])
        self.assertEqual(
            {tool["name"] for tool in backend.calls[0]["tools"]},
            {"list_publications", "search_documents"},
        )
        self.assertEqual(
            [tool["name"] for tool in backend.calls[2]["tools"]], ["read_chunks"]
        )

    def test_unknown_tool_is_returned_as_structured_error(self):
        toolbox = DofToolbox(FakeRetriever())
        self.assertEqual(toolbox.call("missing", {})["error"]["type"], "unknown_tool")

    def test_openai_adapter_serializes_function_calls_and_strict_output(self):
        item = DumpableItem(
            type="function_call",
            call_id="call-1",
            name="read_chunks",
            arguments='{"chunk_ids":[4],"neighbor_window":0}',
        )
        usage = DumpableItem(input_tokens=3, output_tokens=2, total_tokens=5)
        response = SimpleNamespace(
            id="response-1",
            output=[item],
            output_text="",
            usage=usage,
            error=None,
        )
        client = ResponsesClient(response)
        backend = OpenAIResponsesBackend(model="test", client=client)
        turn = backend.create_turn(input_items=[], tools=[], instructions="test")
        self.assertEqual(turn.tool_calls[0].arguments["chunk_ids"], [4])
        self.assertFalse(client.kwargs["store"])
        self.assertNotIn("tools", client.kwargs)
        self.assertIn("reasoning.encrypted_content", client.kwargs["include"])
        self.assertTrue(client.kwargs["text"]["format"]["strict"])

    def test_chat_adapter_preserves_reasoning_and_translates_tool_outputs(self):
        function = SimpleNamespace(
            name="read_chunks",
            arguments='{"chunk_ids":[4],"neighbor_window":0}',
        )
        call = SimpleNamespace(type="function", id="call-1", function=function)
        message = DumpableItem(
            role="assistant",
            content=None,
            reasoning_content="reasoning",
            tool_calls=[call],
        )
        usage = DumpableItem(prompt_tokens=10, completion_tokens=4, total_tokens=14)
        response = SimpleNamespace(
            id="chat-1",
            choices=[SimpleNamespace(message=message)],
            usage=usage,
        )
        client = ChatCompletionsClient(response)
        backend = OpenAIChatCompletionsBackend(
            model="kimi-for-coding",
            api_key="test",
            base_url="https://example.test/v1",
            client=client,
        )
        turn = backend.create_turn(
            input_items=[
                {"role": "user", "content": "question"},
                {
                    "type": "function_call_output",
                    "call_id": "earlier-call",
                    "output": "result",
                },
            ],
            tools=[
                {
                    "type": "function",
                    "name": "read_chunks",
                    "description": "read",
                    "parameters": {"type": "object"},
                    "strict": True,
                }
            ],
            instructions="system",
        )
        self.assertEqual(turn.tool_calls[0].name, "read_chunks")
        self.assertEqual(turn.usage["input_tokens"], 10)
        self.assertEqual(turn.output_items[0]["reasoning_content"], "reasoning")
        self.assertEqual(client.kwargs["messages"][-1]["role"], "tool")
        self.assertEqual(client.kwargs["tools"][0]["function"]["name"], "read_chunks")
        backend.create_turn(input_items=[], tools=[], instructions="final")
        self.assertEqual(client.kwargs["tool_choice"], "none")
        self.assertNotIn("tools", client.kwargs)
