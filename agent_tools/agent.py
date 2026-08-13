"""Bounded, replayable tool-calling loop for DOF research.

The orchestration and tool router are provider-neutral. ``OpenAIResponsesBackend``
is a small adapter around the Responses API; tests can use a scripted backend
without network access or model nondeterminism.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any, Protocol

from jsonschema import Draft202012Validator

from .models import RetrievalStrategy, SearchFilters
from .retrieval import DofRetriever, QueryEmbedder

YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
PROVISION_RE = re.compile(r"\b(?:numeral|art[ií]culo)\s+(\d+(?:\.\d+)*)", re.I)


def _comparison_years(question: str) -> list[str]:
    """Return explicit years only when the question asks across multiple years."""
    years = list(dict.fromkeys(YEAR_RE.findall(question)))
    return years if len(years) > 1 else []


def _coverage_requirements(question: str) -> list[str]:
    requirements = _comparison_years(question)
    if "transitorio" not in question.casefold():
        return requirements
    requirements.append("transitorio")
    requirements.extend(
        f"numeral {number}" for number in PROVISION_RE.findall(question)
    )
    return list(dict.fromkeys(requirements))


FINAL_ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "integer"}},
        "premise_status": {
            "type": "string",
            "enum": ["supported", "false", "unclear"],
        },
    },
    "required": ["answer", "citations", "premise_status"],
    "additionalProperties": False,
}

AGENT_INSTRUCTIONS = """Eres un investigador del Diario Oficial de la Federación.
Usa las herramientas para localizar documentos, buscar pasajes y leer los chunks
que sostengan la respuesta. No respondas con conocimiento externo. Distingue la
fecha de publicación de la fecha de entrada en vigor y respeta la fecha de corte.
Una coincidencia de búsqueda no es una cita: sólo puedes citar IDs devueltos por
read_chunks. Si la evidencia es insuficiente, dilo. Si la pregunta contiene una
premisa falsa, márcala como false y explica la corrección con evidencia cuando sea
posible. Mantén la respuesta concreta.
Al terminar devuelve SOLO JSON con la forma
{"answer":"...","citations":[123],"premise_status":"supported|false|unclear"}.

Política de herramientas:
- Haz una sola llamada por turno y usa la ruta más corta: search_documents,
  search_evidence, read_chunks y respuesta.
- No repitas la misma búsqueda con variaciones menores.
- Usa get_document_outline sólo para estructura o referencias cruzadas, y
  list_publications cuando la fecha de publicación sea el dato de entrada.
- El año sobre el que rige una norma o cantidad no implica que se publicara ese
  año. No fijes date_from sólo a partir del año mencionado en la pregunta.
- Conserva todas las partes de la pregunta desde la primera búsqueda. En una
  comparación entre años, busca evidencia para ambos años antes de responder.
"""


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any] | None
    raw_arguments: str = ""


@dataclass
class ModelTurn:
    response_id: str
    output_items: list[dict[str, Any]]
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_text: str = ""
    usage: dict[str, Any] = field(default_factory=dict)


class AgentBackend(Protocol):
    model: str

    def create_turn(
        self,
        *,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        instructions: str,
    ) -> ModelTurn:
        """Return one model turn, including any requested function calls."""


@dataclass
class ToolTrace:
    sequence: int
    model_turn: int
    call_id: str
    name: str
    arguments: dict[str, Any] | None
    output: dict[str, Any]
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelTurnTrace:
    sequence: int
    response_id: str
    output_types: list[str]
    tool_call_ids: list[str]
    final_text: str
    usage: dict[str, Any]


@dataclass
class AgentAnswer:
    answer: str
    citations: list[int]
    invalid_citations: list[int]
    premise_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentRun:
    question: str
    as_of: str | None
    model: str
    answer: AgentAnswer
    traces: list[ToolTrace]
    turns: list[ModelTurnTrace]
    model_turns: int
    tool_calls: int
    stop_reason: str
    usage: dict[str, int]
    elapsed_ms: float
    coverage: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["traces"] = [trace.to_dict() for trace in self.traces]
        data["answer"] = self.answer.to_dict()
        return data


def _nullable(kind: str) -> dict[str, Any]:
    return {"type": [kind, "null"]}


def _object_schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


class DofToolbox:
    """Validate and execute the five retrieval tools exposed to the model."""

    def __init__(
        self,
        retriever: DofRetriever,
        *,
        embedder: QueryEmbedder | None = None,
        snippet_chars: int = 600,
    ):
        self.retriever = retriever
        self.embedder = embedder
        self.snippet_chars = snippet_chars
        self.as_of: str | None = None
        self.read_chunk_ids: set[int] = set()
        self.visible_document_ids: set[int] = set()
        self.visible_document_titles: dict[int, str] = {}
        self.visible_chunk_ids: set[int] = set()
        self.coverage_requirements: set[str] = set()
        self.covered_requirements: set[str] = set()
        self._vector_cache: dict[str, bytes] = {}
        self._schemas = self._build_schemas()

    @property
    def strategies(self) -> list[str]:
        if self.embedder is not None and self.retriever.versions.vector_available:
            return [strategy.value for strategy in RetrievalStrategy]
        return [RetrievalStrategy.LEXICAL.value]

    def begin(
        self, *, as_of: str | None, coverage_requirements: list[str] | None = None
    ) -> None:
        self.as_of = as_of
        self.read_chunk_ids.clear()
        self.visible_document_ids.clear()
        self.visible_document_titles.clear()
        self.visible_chunk_ids.clear()
        self.coverage_requirements = set(coverage_requirements or [])
        self.covered_requirements.clear()
        self._vector_cache.clear()

    @property
    def missing_coverage(self) -> list[str]:
        return sorted(self.coverage_requirements - self.covered_requirements)

    @property
    def coverage(self) -> dict[str, bool]:
        return {
            requirement: requirement in self.covered_requirements
            for requirement in sorted(self.coverage_requirements)
        }

    def _remember_documents(self, hits: Any) -> None:
        for hit in hits:
            self.visible_document_ids.add(hit.document_id)
            if hit.title:
                self.visible_document_titles[hit.document_id] = hit.title

    @staticmethod
    def _hit_covers(requirement: str, hit: Any, title: str) -> bool:
        if requirement.isdigit():
            return requirement in title
        if requirement == "transitorio":
            headings = " ".join(hit.heading_path).casefold()
            return "transitorio" in headings or bool(
                re.search(r"(?im)^\s*(?:\*\*)?(?:primero|segundo)\.?\**\s", hit.text)
            )
        if requirement.startswith("numeral "):
            number = re.escape(requirement.removeprefix("numeral "))
            return bool(
                re.search(
                    rf"(?im)^\s*(?:>\s*)?(?:\*\*)?{number}(?:\*\*)?(?:\s|$)",
                    hit.text,
                )
            )
        return False

    def _build_schemas(self) -> dict[str, dict[str, Any]]:
        strategy = {"type": "string", "enum": self.strategies}
        filters = {
            "as_of": _nullable("string")
            | {"description": "Fecha de corte YYYY-MM-DD."},
            "date_from": _nullable("string")
            | {"description": "Fecha inicial YYYY-MM-DD."},
            "date_to": _nullable("string") | {"description": "Fecha final YYYY-MM-DD."},
            "section": _nullable("string") | {"description": "Sección del DOF o null."},
        }
        return {
            "list_publications": _object_schema(
                {
                    **filters,
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                }
            ),
            "search_documents": _object_schema(
                {
                    "query": {"type": "string", "minLength": 1, "maxLength": 1000},
                    "strategy": strategy,
                    **filters,
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                }
            ),
            "search_evidence": _object_schema(
                {
                    "query": {"type": "string", "minLength": 1, "maxLength": 1000},
                    "document_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 1,
                        "maxItems": 10,
                    },
                    "strategy": strategy,
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                }
            ),
            "get_document_outline": _object_schema(
                {"document_id": {"type": "integer"}}
            ),
            "read_chunks": _object_schema(
                {
                    "chunk_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 1,
                        "maxItems": 8,
                    },
                    "neighbor_window": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 1,
                    },
                }
            ),
        }

    def tool_definitions(self) -> list[dict[str, Any]]:
        descriptions = {
            "list_publications": "Lista publicaciones por fecha y sección sin buscar texto.",
            "search_documents": "Encuentra documentos candidatos. No devuelve evidencia citable.",
            "search_evidence": "Busca chunks relevantes dentro de documentos candidatos.",
            "get_document_outline": "Muestra encabezados y chunks de un documento sin leer su texto.",
            "read_chunks": "Lee texto verificable. Sólo los IDs leídos pueden citarse al responder.",
        }
        return [
            {
                "type": "function",
                "name": name,
                "description": descriptions[name],
                "parameters": schema,
                "strict": True,
            }
            for name, schema in self._schemas.items()
        ]

    def _filters(self, arguments: dict[str, Any]) -> SearchFilters:
        requested_as_of = arguments.get("as_of") or self.as_of
        if self.as_of and requested_as_of and requested_as_of > self.as_of:
            raise ValueError(
                f"as_of {requested_as_of} exceeds the run cutoff {self.as_of}"
            )
        date_to = arguments.get("date_to")
        if self.as_of and date_to and date_to > self.as_of:
            raise ValueError(f"date_to {date_to} exceeds the run cutoff {self.as_of}")
        return SearchFilters(
            as_of=requested_as_of,
            date_from=arguments.get("date_from"),
            date_to=date_to,
            section=arguments.get("section"),
        )

    def _query_vector(self, query: str, strategy: str) -> bytes | None:
        if strategy == RetrievalStrategy.LEXICAL.value:
            return None
        if self.embedder is None:
            raise ValueError("vector and hybrid strategies require a query embedder")
        if query not in self._vector_cache:
            self._vector_cache[query] = self.embedder.embed_query(query)
        return self._vector_cache[query]

    def call(self, name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
        if name not in self._schemas:
            return {"ok": False, "error": {"type": "unknown_tool", "message": name}}
        if arguments is None:
            return {
                "ok": False,
                "error": {
                    "type": "invalid_json",
                    "message": "arguments are not valid JSON",
                },
            }
        errors = sorted(
            Draft202012Validator(self._schemas[name]).iter_errors(arguments),
            key=lambda error: list(error.path),
        )
        if errors:
            return {
                "ok": False,
                "error": {
                    "type": "invalid_arguments",
                    "message": "; ".join(error.message for error in errors),
                },
            }
        try:
            data = getattr(self, f"_call_{name}")(arguments)
        except (KeyError, ValueError) as exc:
            return {
                "ok": False,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        return {"ok": True, "data": data}

    def _call_list_publications(self, arguments: dict[str, Any]) -> dict[str, Any]:
        filters = self._filters(arguments)
        hits = self.retriever.list_publications(filters, limit=arguments["limit"])
        self._remember_documents(hits)
        return {
            "filters": filters.to_dict(),
            "publications": [asdict(hit) for hit in hits],
        }

    def _call_search_documents(self, arguments: dict[str, Any]) -> dict[str, Any]:
        strategy = arguments["strategy"]
        filters = self._filters(arguments)
        result = self.retriever.search_documents(
            arguments["query"],
            strategy=strategy,
            filters=filters,
            query_vector=self._query_vector(arguments["query"], strategy),
            bm25_depth=100,
            vector_k=300,
            top_k=arguments["top_k"],
        )
        self._remember_documents(result.documents)
        return result.to_dict()

    def _call_search_evidence(self, arguments: dict[str, Any]) -> dict[str, Any]:
        undiscovered = set(arguments["document_ids"]) - self.visible_document_ids
        if undiscovered:
            raise ValueError(
                f"document ids were not returned by an earlier tool: {sorted(undiscovered)}"
            )
        strategy = arguments["strategy"]
        result = self.retriever.search_evidence(
            arguments["query"],
            arguments["document_ids"],
            strategy=strategy,
            query_vector=self._query_vector(arguments["query"], strategy),
            top_k=arguments["top_k"],
            candidate_depth=300,
            vector_k=300,
        )
        self.visible_chunk_ids.update(result.evidence_ids)
        data = result.to_dict()
        for hit in data["evidence"]:
            text = hit.pop("text")
            hit["snippet"] = text[: self.snippet_chars]
            hit["snippet_truncated"] = len(text) > self.snippet_chars
        return data

    def _call_get_document_outline(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if arguments["document_id"] not in self.visible_document_ids:
            raise ValueError("document_id was not returned by an earlier tool")
        outline = self.retriever.get_document_outline(arguments["document_id"])
        if outline.title:
            self.visible_document_titles[outline.document_id] = outline.title
        self.visible_chunk_ids.update(chunk.chunk_id for chunk in outline.chunks)
        data = asdict(outline)
        data["chunks"] = data["chunks"][:200]
        data["outline_truncated"] = len(outline.chunks) > 200
        return data

    def _call_read_chunks(self, arguments: dict[str, Any]) -> dict[str, Any]:
        undiscovered = set(arguments["chunk_ids"]) - self.visible_chunk_ids
        if undiscovered:
            raise ValueError(
                f"chunk ids were not returned by an earlier tool: {sorted(undiscovered)}"
            )
        hits = self.retriever.read_chunks(
            arguments["chunk_ids"], neighbor_window=arguments["neighbor_window"]
        )
        self.read_chunk_ids.update(hit.chunk_id for hit in hits)
        for requirement in self.coverage_requirements:
            if any(
                self._hit_covers(
                    requirement,
                    hit,
                    self.visible_document_titles.get(hit.document_id, ""),
                )
                for hit in hits
            ):
                self.covered_requirements.add(requirement)
        return {"chunks": [asdict(hit) for hit in hits], "coverage": self.coverage}


class OpenAIResponsesBackend:
    """OpenAI Responses API adapter used by the provider-neutral runner."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        reasoning_effort: str | None = "low",
        max_output_tokens: int = 1400,
        client: Any = None,
    ):
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url)
        self.client = client
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens

    def create_turn(
        self,
        *,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        instructions: str,
    ) -> ModelTurn:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": instructions,
            "input": input_items,
            "parallel_tool_calls": False,
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "max_output_tokens": self.max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "dof_research_answer",
                    "schema": FINAL_ANSWER_SCHEMA,
                    "strict": True,
                },
                "verbosity": "low",
            },
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        response = self.client.responses.create(**kwargs)
        if getattr(response, "error", None):
            raise RuntimeError(str(response.error))
        output_items = [
            item.model_dump(mode="json", exclude_none=True) for item in response.output
        ]
        calls: list[ToolCall] = []
        for item in response.output:
            if item.type != "function_call":
                continue
            try:
                arguments = json.loads(item.arguments)
            except (TypeError, json.JSONDecodeError):
                arguments = None
            calls.append(
                ToolCall(
                    call_id=item.call_id,
                    name=item.name,
                    arguments=arguments,
                    raw_arguments=item.arguments,
                )
            )
        usage = (
            response.usage.model_dump(mode="json", exclude_none=True)
            if response.usage
            else {}
        )
        return ModelTurn(
            response_id=response.id,
            output_items=output_items,
            tool_calls=calls,
            final_text=response.output_text or "",
            usage=usage,
        )


class OpenAIChatCompletionsBackend:
    """Adapter for OpenAI-compatible Chat Completions providers such as Kimi."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        max_output_tokens: int = 2400,
        client: Any = None,
    ):
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url)
        self.client = client
        self.model = model
        self.max_output_tokens = max_output_tokens

    @staticmethod
    def _messages(
        input_items: list[dict[str, Any]], instructions: str
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": instructions}]
        for item in input_items:
            if item.get("type") == "function_call_output":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item["call_id"],
                        "content": item["output"],
                    }
                )
                continue
            if item.get("role") not in {"user", "assistant"}:
                continue
            message = {
                key: value
                for key, value in item.items()
                if key
                in {
                    "role",
                    "content",
                    "tool_calls",
                    "reasoning_content",
                    "refusal",
                }
            }
            messages.append(message)
        return messages

    @staticmethod
    def _chat_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                    "strict": tool["strict"],
                },
            }
            for tool in tools
        ]

    def create_turn(
        self,
        *,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        instructions: str,
    ) -> ModelTurn:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages(input_items, instructions),
            "parallel_tool_calls": False,
            "max_tokens": self.max_output_tokens,
        }
        if tools:
            kwargs["tools"] = self._chat_tools(tools)
            kwargs["tool_choice"] = "auto"
        else:
            kwargs["tool_choice"] = "none"
        response = self.client.chat.completions.create(**kwargs)
        if not response.choices:
            raise RuntimeError("chat completion returned no choices")
        message = response.choices[0].message
        message_data = message.model_dump(mode="json", exclude_none=True)
        message_data["type"] = "chat_message"
        calls: list[ToolCall] = []
        for call in message.tool_calls or []:
            if call.type != "function":
                continue
            try:
                arguments = json.loads(call.function.arguments)
            except (TypeError, json.JSONDecodeError):
                arguments = None
            calls.append(
                ToolCall(
                    call_id=call.id,
                    name=call.function.name,
                    arguments=arguments,
                    raw_arguments=call.function.arguments,
                )
            )
        raw_usage = (
            response.usage.model_dump(mode="json", exclude_none=True)
            if response.usage
            else {}
        )
        usage = {
            "input_tokens": int(raw_usage.get("prompt_tokens", 0)),
            "output_tokens": int(raw_usage.get("completion_tokens", 0)),
            "total_tokens": int(raw_usage.get("total_tokens", 0)),
        }
        return ModelTurn(
            response_id=response.id,
            output_items=[message_data],
            tool_calls=calls,
            final_text=message.content or "",
            usage=usage,
        )


def _parse_final_answer(text: str, allowed: set[int]) -> AgentAnswer:
    try:
        decoder = json.JSONDecoder()
        data = None
        for index, character in enumerate(text):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                data = candidate
                break
        if data is None:
            raise ValueError("response did not contain a JSON object")
        Draft202012Validator(FINAL_ANSWER_SCHEMA).validate(data)
    except Exception as exc:  # The concrete parse/validation error is useful in traces.
        raise ValueError(f"invalid final answer: {exc}") from exc
    proposed = list(dict.fromkeys(int(value) for value in data["citations"]))
    return AgentAnswer(
        answer=data["answer"].strip(),
        citations=[citation for citation in proposed if citation in allowed],
        invalid_citations=[
            citation for citation in proposed if citation not in allowed
        ],
        premise_status=data["premise_status"],
    )


def _add_usage(total: dict[str, int], usage: dict[str, Any]) -> None:
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            total[key] = total.get(key, 0) + value


class AgentRunner:
    """Run an auditable agent with hard limits on turns and tool calls."""

    def __init__(
        self,
        backend: AgentBackend,
        toolbox: DofToolbox,
        *,
        max_model_turns: int = 7,
        max_tool_calls: int = 8,
        instructions: str = AGENT_INSTRUCTIONS,
    ):
        if max_model_turns < 1 or max_tool_calls < 1:
            raise ValueError("agent limits must be positive")
        self.backend = backend
        self.toolbox = toolbox
        self.max_model_turns = max_model_turns
        self.max_tool_calls = max_tool_calls
        self.instructions = instructions

    def _available_tools(self) -> list[dict[str, Any]]:
        definitions = {tool["name"]: tool for tool in self.toolbox.tool_definitions()}
        if self.toolbox.read_chunk_ids and not self.toolbox.missing_coverage:
            return []
        if self.toolbox.read_chunk_ids:
            return [
                definitions[name]
                for name in ("search_documents", "search_evidence", "read_chunks")
            ]
        if self.toolbox.visible_chunk_ids:
            return [definitions["read_chunks"]]
        if self.toolbox.visible_document_ids:
            return [
                definitions[name]
                for name in (
                    "search_documents",
                    "search_evidence",
                    "get_document_outline",
                )
            ]
        return [definitions[name] for name in ("list_publications", "search_documents")]

    def run(self, question: str, *, as_of: str | None = None) -> AgentRun:
        started = perf_counter()
        coverage_requirements = _coverage_requirements(question)
        self.toolbox.begin(as_of=as_of, coverage_requirements=coverage_requirements)
        coverage_prompt = (
            "\nCobertura obligatoria antes de responder: "
            + ", ".join(coverage_requirements)
            if coverage_requirements
            else ""
        )
        prompt = (
            f"Fecha de corte obligatoria: {as_of or 'no indicada'}\n"
            f"Pregunta: {question}{coverage_prompt}"
        )
        input_items: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        traces: list[ToolTrace] = []
        turns: list[ModelTurnTrace] = []
        usage: dict[str, int] = {}
        last_parse_error = "no final answer"
        for turn_number in range(1, self.max_model_turns + 1):
            final_turn = turn_number == self.max_model_turns
            available_tools = [] if final_turn else self._available_tools()
            force_final = not available_tools
            turn_input = input_items
            if force_final:
                turn_input = [
                    *input_items,
                    {
                        "role": "user",
                        "content": (
                            "No solicites más herramientas. Responde ahora únicamente con "
                            "el objeto JSON final requerido, usando sólo los chunks leídos."
                        ),
                    },
                ]
            turn = self.backend.create_turn(
                input_items=turn_input,
                tools=available_tools,
                instructions=(
                    self.instructions
                    + "\nNo quedan más turnos de herramientas. Entrega ahora el JSON final "
                    "usando sólo los chunks ya leídos."
                    if not available_tools
                    else self.instructions
                ),
            )
            _add_usage(usage, turn.usage)
            turns.append(
                ModelTurnTrace(
                    sequence=turn_number,
                    response_id=turn.response_id,
                    output_types=[
                        str(item.get("type", "unknown")) for item in turn.output_items
                    ],
                    tool_call_ids=[call.call_id for call in turn.tool_calls],
                    final_text=turn.final_text,
                    usage=turn.usage,
                )
            )
            if force_final and turn.tool_calls:
                last_parse_error = "model requested a tool during forced finalization"
                continue
            input_items.extend(turn.output_items)
            if turn.tool_calls:
                for call in turn.tool_calls:
                    if len(traces) >= self.max_tool_calls:
                        output = {
                            "ok": False,
                            "error": {
                                "type": "tool_limit",
                                "message": f"maximum {self.max_tool_calls} tool calls reached",
                            },
                        }
                        elapsed_ms = 0.0
                    else:
                        tool_started = perf_counter()
                        output = self.toolbox.call(call.name, call.arguments)
                        elapsed_ms = (perf_counter() - tool_started) * 1000.0
                        traces.append(
                            ToolTrace(
                                sequence=len(traces) + 1,
                                model_turn=turn_number,
                                call_id=call.call_id,
                                name=call.name,
                                arguments=call.arguments,
                                output=output,
                                elapsed_ms=elapsed_ms,
                            )
                        )
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": json.dumps(output, ensure_ascii=False),
                        }
                    )
                continue
            if turn.final_text:
                if self.toolbox.missing_coverage and not final_turn:
                    last_parse_error = "faltan requisitos de cobertura: " + ", ".join(
                        self.toolbox.missing_coverage
                    )
                    input_items.append(
                        {
                            "role": "user",
                            "content": (
                                "Aún no puedes cerrar. Lee evidencia de documentos cuyo "
                                "título cubra: "
                                + ", ".join(self.toolbox.missing_coverage)
                            ),
                        }
                    )
                    continue
                try:
                    answer = _parse_final_answer(
                        turn.final_text, self.toolbox.read_chunk_ids
                    )
                except ValueError as exc:
                    last_parse_error = str(exc)
                    input_items.append(
                        {
                            "role": "user",
                            "content": f"Corrige la respuesta final: {last_parse_error}",
                        }
                    )
                    continue
                return AgentRun(
                    question=question,
                    as_of=as_of,
                    model=self.backend.model,
                    answer=answer,
                    traces=traces,
                    turns=turns,
                    model_turns=turn_number,
                    tool_calls=len(traces),
                    stop_reason=(
                        "completed"
                        if not self.toolbox.missing_coverage
                        else "coverage_incomplete: "
                        + ",".join(self.toolbox.missing_coverage)
                    ),
                    usage=usage,
                    elapsed_ms=(perf_counter() - started) * 1000.0,
                    coverage=self.toolbox.coverage,
                )
            last_parse_error = "model returned neither tool calls nor a final answer"
        answer = AgentAnswer(
            answer="No se obtuvo una respuesta final verificable dentro de los límites.",
            citations=[],
            invalid_citations=[],
            premise_status="unclear",
        )
        return AgentRun(
            question=question,
            as_of=as_of,
            model=self.backend.model,
            answer=answer,
            traces=traces,
            turns=turns,
            model_turns=self.max_model_turns,
            tool_calls=len(traces),
            stop_reason=f"model_turn_limit: {last_parse_error}",
            usage=usage,
            elapsed_ms=(perf_counter() - started) * 1000.0,
            coverage=self.toolbox.coverage,
        )
