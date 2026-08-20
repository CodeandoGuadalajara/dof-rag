"""Bounded, replayable tool-calling loop for DOF research.

The orchestration and tool router are provider-neutral. ``OpenAIResponsesBackend``
is a small adapter around the Responses API; tests can use a scripted backend
without network access or model nondeterminism.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any, Protocol

from jsonschema import Draft202012Validator

from .models import RetrievalStrategy, SearchFilters
from .retrieval import DofRetriever, QueryEmbedder

LOGGER = logging.getLogger(__name__)

YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
YEAR_COMPARISON_RE = re.compile(
    r"\b(?:de\s+((?:19|20)\d{2})\s+a|entre\s+((?:19|20)\d{2})\s+y)\s+"
    r"((?:19|20)\d{2})\b",
    re.I,
)
PROVISION_LIST_RE = re.compile(r"\bnumeral(?:es)?\s+([^:;?]+)", re.I)
RANGE_PATTERNS = (
    (re.compile(r"\bhasta\s+(\d+)\b", re.I), "rango hasta {0}"),
    (
        re.compile(r"\bentre\s+(\d+)\s+y\s+(\d+)\b", re.I),
        "rango entre {0} y {1}",
    ),
    (re.compile(r"\bm[aá]s\s+de\s+(\d+)\b", re.I), "rango más de {0}"),
)
EXPLICIT_TERMS = ("diario", "mensual", "anual")
NUMBER_WORDS = {
    "quince": "15",
    "dieciseis": "16",
    "cincuenta": "50",
}
SPANISH_MONTHS = {
    "enero": "01",
    "febrero": "02",
    "marzo": "03",
    "abril": "04",
    "mayo": "05",
    "junio": "06",
    "julio": "07",
    "agosto": "08",
    "septiembre": "09",
    "octubre": "10",
    "noviembre": "11",
    "diciembre": "12",
}
MONTH_NAMES_RE = "|".join(SPANISH_MONTHS)
DATED_PLAN_RE = re.compile(
    r"\b(?:Plan\s+Nacional\s+de\s+Desarrollo|PND)\s+"
    r"((?:19|20)\d{2}-(?:19|20)\d{2}"
    r"(?:\s*(?:,|y)\s*(?:19|20)\d{2}-(?:19|20)\d{2})*)\b",
    re.I,
)
NORM_ID_RE = re.compile(r"\b(?:NOM|NMX)-\d{3}(?:-[A-Z]+)*(?:-\d{4})?\b", re.I)
SOURCE_DATE_RE = re.compile(
    rf"\b(?:publicad[oa](?:\s+en\s+el\s+DOF)?(?:\s+el)?|"
    rf"(?:decreto|acuerdo|resoluci[oó]n|reforma)\s+del)\s+"
    rf"(\d{{1,2}})\s+de\s+({MONTH_NAMES_RE})\s+de\s+((?:19|20)\d{{2}})\b",
    re.I,
)
SOURCE_MONTH_RE = re.compile(
    rf"\b(?:reforma|decreto|acuerdo|resoluci[oó]n)"
    rf"(?:\s+[\wáéíóúüñ-]+){{0,6}}?\s+de\s+"
    rf"({MONTH_NAMES_RE})\s+de\s+((?:19|20)\d{{2}})\b",
    re.I,
)
REFORM_TOPIC_RE = re.compile(
    rf"\breforma(?:\s+constitucional)?\s+de\s+(.+?)"
    rf"(?=\s+de\s+(?:{MONTH_NAMES_RE})\s+de\s+(?:19|20)\d{{2}}"
    r"|\s+y\s+(?:la|el)\b|[?;,]|$)",
    re.I,
)
EXPLICIT_LEGAL_ACTIONS = (
    "declaratoria de utilidad pública",
    "decreto de expropiación",
)
TOPIC_STOP_WORDS = {
    "al",
    "de",
    "del",
    "digna",
    "dignas",
    "el",
    "en",
    "la",
    "las",
    "los",
}
SEARCH_FAILURE_RE = re.compile(
    r"\b(?:no\s+se\s+(?:encontr[oó]|localiz[oó]|hall[oó])|"
    r"no\s+fue\s+posible\s+(?:encontrar|localizar)|"
    r"evidencia\s+(?:disponible\s+)?insuficiente|"
    r"chunks?\s+(?:le[ií]dos?\s+)?no\s+(?:contienen?|muestran?))\b",
    re.I,
)
CORRECTION_ASSERTION_RE = re.compile(
    r"\b(?:reform\w*|expid\w*|derog\w*|establec\w*|dispus\w*|"
    r"comprend\w*|correspond\w*|consta\w*|incluy\w*|vigent\w*|"
    r"formad[oa]s?)\b",
    re.I,
)


def _comparison_years(question: str) -> list[str]:
    """Return explicit years only when the question asks across multiple years."""
    years: list[str] = []
    for match in YEAR_COMPARISON_RE.finditer(question):
        years.extend(value for value in match.groups() if value)
    return list(dict.fromkeys(years))


def _fold_for_coverage(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.casefold())
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    for word, number in NUMBER_WORDS.items():
        folded = re.sub(rf"\b{word}\b", number, folded)
    return re.sub(r"\s+", " ", folded)


def _enumeration_requirements(question: str) -> list[str]:
    requirements: list[str] = []
    for pattern, label in RANGE_PATTERNS:
        requirements.extend(
            label.format(*match.groups()) for match in pattern.finditer(question)
        )
    folded = _fold_for_coverage(question)
    explicit_terms = [term for term in EXPLICIT_TERMS if term in folded]
    if len(explicit_terms) > 1:
        requirements.extend(f"término {term}" for term in explicit_terms)
    return requirements


def _transitory_provisions(question: str) -> list[str]:
    if "transitorio" not in question.casefold():
        return []
    numbers: list[str] = []
    for match in PROVISION_LIST_RE.finditer(question):
        numbers.extend(re.findall(r"\b\d+\.\d+\b", match.group(1)))
    return list(dict.fromkeys(numbers))


def _explicit_question_requirements(question: str) -> list[str]:
    """Extract source and subject anchors stated by the question itself."""
    requirements: list[str] = []
    for match in DATED_PLAN_RE.finditer(question):
        requirements.extend(
            f"tema PND {period}"
            for period in re.findall(r"(?:19|20)\d{2}-(?:19|20)\d{2}", match.group(1))
        )
    requirements.extend(
        f"tema {match.group(0).upper()}" for match in NORM_ID_RE.finditer(question)
    )

    folded_question = _fold_for_coverage(question)
    for action in EXPLICIT_LEGAL_ACTIONS:
        if _fold_for_coverage(action) in folded_question:
            requirements.append(f"tema {action}")

    for match in REFORM_TOPIC_RE.finditer(question):
        topic = re.sub(r"\s+", " ", match.group(1)).strip(" .")
        folded_topic = _fold_for_coverage(topic)
        if topic and not re.fullmatch(
            rf"(?:{MONTH_NAMES_RE})(?:\s+de\s+(?:19|20)\d{{2}})?",
            folded_topic,
        ):
            requirements.append(f"tema reforma: {topic}")

    for match in SOURCE_DATE_RE.finditer(question):
        day, month, year = match.groups()
        requirements.append(
            f"publicación {year}-{SPANISH_MONTHS[month.casefold()]}-{int(day):02d}"
        )
    full_dates = {item for item in requirements if item.startswith("publicación ")}
    for match in SOURCE_MONTH_RE.finditer(question):
        month, year = match.groups()
        prefix = f"publicación {year}-{SPANISH_MONTHS[month.casefold()]}"
        if not any(item.startswith(prefix + "-") for item in full_dates):
            requirements.append(prefix)
    return list(dict.fromkeys(requirements))


def _coverage_requirements(question: str) -> list[str]:
    requirements = [
        *_comparison_years(question),
        *_enumeration_requirements(question),
        *_explicit_question_requirements(question),
    ]
    provisions = _transitory_provisions(question)
    if provisions:
        requirements.append("transitorio")
        requirements.extend(f"numeral {number}" for number in provisions)
    return list(dict.fromkeys(requirements))


def _has_affirmative_premise_correction(answer: str) -> bool:
    """Reject a false-premise answer that only reports an unsuccessful search."""
    for clause in re.split(r"[.;]\s*", _fold_for_coverage(answer)):
        if not clause:
            continue
        if " sino " in f" {clause} " and CORRECTION_ASSERTION_RE.search(clause):
            return True
        if clause.startswith("no ") or SEARCH_FAILURE_RE.search(clause):
            continue
        if CORRECTION_ASSERTION_RE.search(clause):
            return True
    return False


FINAL_ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 1,
        },
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
read_chunks y debes incluir al menos una cita válida. Si la evidencia es
insuficiente, dilo. Marca una premisa como false únicamente cuando los chunks
citados establezcan la corrección y exprésala de forma afirmativa. "No se encontró"
no demuestra que una premisa sea falsa; si no puedes documentar la corrección,
usa unclear. Mantén la respuesta concreta. Indica la fecha de publicación de las
fuentes que sostienen tu respuesta y, si la evidencia más reciente disponible es
antigua, advierte que la regla o el programa pudo haber cambiado.
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
- Si la pregunta trata sobre programas, apoyos, requisitos o reglas vigentes y
  no fija una fecha histórica, usa prefer_recent=true en search_documents y
  comprueba si el instrumento encontrado fue reformado, derogado o sustituido
  con posterioridad antes de responder. No uses un date_from rígido que excluya
  la ley o programa base todavía vigente.
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


class CitationRequiredError(ValueError):
    """A nominal final answer has no valid citation from a read chunk."""


class CitationCoverageError(ValueError):
    """Citations do not span the document count required by the question."""


class PremiseCorrectionRequiredError(ValueError):
    """A false-premise answer does not state a substantive correction."""


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
    required_hops: int = 1
    coverage: dict[str, bool] = field(default_factory=dict)
    verification: dict[str, Any] = field(default_factory=dict)

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


def _query_snippet(text: str, query: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    folded_text = _fold_for_coverage(text)
    folded_query = _fold_for_coverage(query)
    anchors = [
        *re.findall(r"\barticulo\s+\d+(?:\.\d+)*\b", folded_query),
        *re.findall(r"\bnumeral\s+\d+(?:\.\d+)*\b", folded_query),
        *re.findall(r"\b\d+\.\d+\b", folded_query),
    ]
    position = -1
    for anchor in anchors:
        position = folded_text.find(anchor)
        if position >= 0:
            break
    if position < 0:
        terms = {term for term in re.findall(r"\w+", folded_query) if len(term) >= 4}
        candidates = [folded_text.find(term) for term in terms]
        position = min((value for value in candidates if value >= 0), default=0)
    start = max(0, position - max_chars // 4)
    end = min(len(text), start + max_chars)
    start = max(0, end - max_chars)
    snippet = text[start:end]
    if start:
        snippet = "…" + snippet[1:]
    if end < len(text):
        snippet = snippet[:-1] + "…"
    return snippet, True


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
        self.read_chunk_documents: dict[int, int] = {}
        self.read_document_ids: set[int] = set()
        self.visible_document_ids: set[int] = set()
        self.visible_document_titles: dict[int, str] = {}
        self.visible_document_years: dict[int, set[str]] = {}
        self.visible_chunk_ids: set[int] = set()
        self.coverage_requirements: set[str] = set()
        self.covered_requirements: set[str] = set()
        self.required_hops = 1
        self._vector_cache: dict[str, bytes] = {}
        self._schemas = self._build_schemas()

    @property
    def strategies(self) -> list[str]:
        if self.embedder is not None and self.retriever.versions.vector_available:
            return [strategy.value for strategy in RetrievalStrategy]
        return [RetrievalStrategy.LEXICAL.value]

    def begin(
        self,
        *,
        as_of: str | None,
        coverage_requirements: list[str] | None = None,
        required_hops: int = 1,
    ) -> None:
        if required_hops < 1:
            raise ValueError("required_hops must be positive")
        self.as_of = as_of
        self.read_chunk_ids.clear()
        self.read_chunk_documents.clear()
        self.read_document_ids.clear()
        self.visible_document_ids.clear()
        self.visible_document_titles.clear()
        self.visible_document_years.clear()
        self.visible_chunk_ids.clear()
        self.coverage_requirements = set(coverage_requirements or [])
        self.covered_requirements.clear()
        self.required_hops = required_hops
        self._vector_cache.clear()

    @property
    def missing_coverage(self) -> list[str]:
        missing = sorted(self.coverage_requirements - self.covered_requirements)
        if len(self.read_document_ids) < self.required_hops:
            missing.append(self.document_coverage_label)
        return missing

    @property
    def document_coverage_label(self) -> str:
        return f"documentos distintos (mínimo {self.required_hops})"

    @property
    def coverage(self) -> dict[str, bool]:
        coverage = {
            requirement: requirement in self.covered_requirements
            for requirement in sorted(self.coverage_requirements)
        }
        if self.required_hops > 1:
            coverage[self.document_coverage_label] = (
                len(self.read_document_ids) >= self.required_hops
            )
        return coverage

    def _remember_documents(self, hits: Any) -> None:
        for hit in hits:
            self.visible_document_ids.add(hit.document_id)
            if hit.title:
                self.visible_document_titles[hit.document_id] = hit.title
            year_hints = set(YEAR_RE.findall(hit.title or ""))
            if hit.publication_date:
                year_hints.add(hit.publication_date[:4])
            self.visible_document_years[hit.document_id] = year_hints

    @staticmethod
    def _hit_covers(
        requirement: str,
        hit: Any,
        year_hints: set[str],
        title: str = "",
    ) -> bool:
        if requirement.isdigit():
            return requirement in year_hints
        folded_text = _fold_for_coverage(hit.text)
        folded_source = _fold_for_coverage(
            " ".join(
                value
                for value in (
                    title,
                    getattr(hit, "path", ""),
                    getattr(hit, "text", ""),
                )
                if value
            )
        )
        if requirement.startswith("publicación "):
            expected = requirement.removeprefix("publicación ")
            publication_date = getattr(hit, "publication_date", None) or ""
            return publication_date == expected or (
                len(expected) == 7 and publication_date.startswith(expected + "-")
            )
        if requirement.startswith("tema PND "):
            period = _fold_for_coverage(requirement.removeprefix("tema PND "))
            return period in folded_source and (
                "plan nacional de desarrollo" in folded_source
                or re.search(r"\bpnd\b", folded_source) is not None
            )
        if requirement.startswith("tema NOM-") or requirement.startswith("tema NMX-"):
            identifier = _fold_for_coverage(requirement.removeprefix("tema "))
            return identifier in folded_source
        if requirement.startswith("tema reforma: "):
            topic = _fold_for_coverage(requirement.removeprefix("tema reforma: "))
            terms = [
                term
                for term in re.findall(r"\w+", topic)
                if len(term) >= 4 and term not in TOPIC_STOP_WORDS
            ]
            return "reform" in folded_source and any(
                term[:7] in folded_source for term in terms
            )
        if requirement.startswith("tema "):
            terms = [
                term
                for term in re.findall(
                    r"\w+", _fold_for_coverage(requirement.removeprefix("tema "))
                )
                if len(term) >= 4 and term not in TOPIC_STOP_WORDS
            ]
            return bool(terms) and all(term in folded_source for term in terms)
        if requirement.startswith("término "):
            return requirement.removeprefix("término ") in folded_text
        if requirement.startswith("rango hasta "):
            limit = re.escape(requirement.removeprefix("rango hasta "))
            return bool(re.search(rf"\bhasta\s+{limit}\b", folded_text))
        if requirement.startswith("rango entre "):
            bounds = re.findall(r"\d+", requirement)
            return len(bounds) == 2 and bool(
                re.search(
                    rf"\bentre\s+{re.escape(bounds[0])}\s+y\s+{re.escape(bounds[1])}\b",
                    folded_text,
                )
            )
        if requirement.startswith("rango más de "):
            limit = re.escape(requirement.removeprefix("rango más de "))
            return bool(re.search(rf"\bmas\s+de\s+{limit}\b", folded_text))
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
                    "prefer_recent": _nullable("boolean")
                    | {
                        "description": (
                            "true da prioridad a las publicaciones más recientes; "
                            "úsalo para preguntas sobre la situación vigente."
                        )
                    },
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
            "search_documents": (
                "Encuentra documentos candidatos con su fecha de publicación. "
                "No devuelve evidencia citable."
            ),
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
            prefer_recent=bool(arguments.get("prefer_recent")),
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
            snippet, truncated = _query_snippet(
                text, arguments["query"], self.snippet_chars
            )
            hit["snippet"] = snippet
            hit["snippet_truncated"] = truncated
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
        self.read_chunk_documents.update(
            {hit.chunk_id: hit.document_id for hit in hits}
        )
        self.read_document_ids.update(hit.document_id for hit in hits)
        for requirement in self.coverage_requirements:
            if any(
                self._hit_covers(
                    requirement,
                    hit,
                    self.visible_document_years.get(hit.document_id, set()),
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


def _parse_final_answer(
    text: str,
    allowed: set[int],
    *,
    citation_documents: dict[int, int] | None = None,
    required_hops: int = 1,
) -> AgentAnswer:
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
        if not data.get("citations"):
            raise CitationRequiredError(
                "final answer requires at least one citation from a read chunk"
            )
        Draft202012Validator(FINAL_ANSWER_SCHEMA).validate(data)
    except CitationRequiredError:
        raise
    except Exception as exc:  # The concrete parse/validation error is useful in traces.
        raise ValueError(f"invalid final answer: {exc}") from exc
    proposed = list(dict.fromkeys(int(value) for value in data["citations"]))
    citations = [citation for citation in proposed if citation in allowed]
    if not citations:
        raise CitationRequiredError(
            "final answer requires at least one valid citation from a read chunk"
        )
    cited_documents = {
        citation_documents[citation]
        for citation in citations
        if citation_documents and citation in citation_documents
    }
    if required_hops > 1 and len(cited_documents) < required_hops:
        raise CitationCoverageError(
            f"final answer requires citations from {required_hops} distinct documents"
        )
    if data["premise_status"] == "false" and not _has_affirmative_premise_correction(
        data["answer"]
    ):
        raise PremiseCorrectionRequiredError(
            "false premise requires an affirmative correction, not only a failed search"
        )
    return AgentAnswer(
        answer=data["answer"].strip(),
        citations=citations,
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


def _verification(
    answer: AgentAnswer,
    toolbox: DofToolbox,
    required_hops: int,
) -> dict[str, Any]:
    cited_documents = {
        toolbox.read_chunk_documents[citation]
        for citation in answer.citations
        if citation in toolbox.read_chunk_documents
    }
    false_premise = answer.premise_status == "false"
    return {
        "citation_from_read_chunk": bool(answer.citations),
        "coverage_requirements_met": not toolbox.missing_coverage,
        "distinct_cited_documents_met": len(cited_documents) >= required_hops,
        "false_premise_correction_form": (
            _has_affirmative_premise_correction(answer.answer)
            if false_premise
            else None
        ),
        "correction_supported_by_citations": (
            "human_review_required" if false_premise else "not_applicable"
        ),
    }


class AgentRunner:
    """Run an auditable agent with hard limits on turns and tool calls."""

    def __init__(
        self,
        backend: AgentBackend,
        toolbox: DofToolbox,
        *,
        max_model_turns: int = 8,
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

    def run(
        self,
        question: str,
        *,
        as_of: str | None = None,
        required_hops: int = 1,
        on_progress: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> AgentRun:
        started = perf_counter()
        coverage_requirements = _coverage_requirements(question)
        self.toolbox.begin(
            as_of=as_of,
            coverage_requirements=coverage_requirements,
            required_hops=required_hops,
        )
        _emit_progress(
            on_progress,
            "agent_started",
            {
                "message": "El agente comenzó a investigar la pregunta.",
                "why": (
                    "Primero localizará documentos relevantes y después leerá "
                    "pasajes que puedan sostener citas verificables."
                ),
                "as_of": as_of,
                "required_hops": required_hops,
                "coverage_requirements": coverage_requirements,
            },
        )
        coverage_prompt = (
            "\nCobertura obligatoria antes de responder: "
            + ", ".join(coverage_requirements)
            if coverage_requirements
            else ""
        )
        prompt = (
            f"Fecha de corte obligatoria: {as_of or 'no indicada'}\n"
            f"Pregunta: {question}{coverage_prompt}\n"
            f"Documentos distintos requeridos por la tarea: {required_hops}. "
            "Lee y cita evidencia de cada uno antes de responder."
        )
        input_items: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        traces: list[ToolTrace] = []
        turns: list[ModelTurnTrace] = []
        usage: dict[str, int] = {}
        last_parse_error = "no final answer"
        terminal_stop_reason = "model_turn_limit"
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
            _emit_progress(
                on_progress,
                "model_turn_started",
                {
                    "message": (
                        "Organizando la evidencia y preparando la respuesta final."
                        if force_final
                        else "Analizando lo encontrado y decidiendo el siguiente paso."
                    ),
                    "turn": turn_number,
                    "max_turns": self.max_model_turns,
                    "available_tools": [tool["name"] for tool in available_tools],
                },
            )
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
                    _emit_progress(
                        on_progress,
                        "tool_started",
                        {
                            "message": _tool_start_message(call.name, call.arguments),
                            "why": _tool_reason(call.name),
                            "tool": call.name,
                            "arguments": call.arguments or {},
                            "turn": turn_number,
                        },
                    )
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
                    _emit_progress(
                        on_progress,
                        "tool_completed",
                        _public_tool_progress(
                            call.name,
                            call.arguments,
                            output,
                            elapsed_ms=elapsed_ms,
                            turn=turn_number,
                        ),
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
                if not self.toolbox.read_chunk_ids:
                    last_parse_error = "no se ha leído evidencia"
                    terminal_stop_reason = "evidence_not_read"
                    if final_turn:
                        break
                    input_items.append(
                        {
                            "role": "user",
                            "content": (
                                "Aún no puedes cerrar. Usa read_chunks para leer al menos "
                                "un pasaje verificable antes de responder."
                            ),
                        }
                    )
                    continue
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
                        turn.final_text,
                        self.toolbox.read_chunk_ids,
                        citation_documents=self.toolbox.read_chunk_documents,
                        required_hops=required_hops,
                    )
                except CitationCoverageError as exc:
                    last_parse_error = str(exc)
                    terminal_stop_reason = "citation_coverage_incomplete"
                    _emit_progress(
                        on_progress,
                        "answer_revision_requested",
                        {
                            "message": "La verificación pidió cubrir más documentos.",
                            "reason": "citation_coverage_incomplete",
                            "turn": turn_number,
                        },
                    )
                    input_items.append(
                        {
                            "role": "user",
                            "content": (
                                "Corrige la respuesta final: cita evidencia leída de "
                                f"{required_hops} documentos distintos."
                            ),
                        }
                    )
                    continue
                except PremiseCorrectionRequiredError as exc:
                    last_parse_error = str(exc)
                    terminal_stop_reason = "premise_correction_required"
                    _emit_progress(
                        on_progress,
                        "answer_revision_requested",
                        {
                            "message": "La verificación pidió sustentar la corrección de la premisa.",
                            "reason": "premise_correction_required",
                            "turn": turn_number,
                        },
                    )
                    input_items.append(
                        {
                            "role": "user",
                            "content": (
                                "Corrige la respuesta final: si marcas premise_status "
                                "como false, afirma qué ocurrió realmente y cita el "
                                "pasaje que lo demuestra. Una búsqueda fallida no basta."
                            ),
                        }
                    )
                    continue
                except CitationRequiredError as exc:
                    last_parse_error = str(exc)
                    terminal_stop_reason = "citation_required"
                    _emit_progress(
                        on_progress,
                        "answer_revision_requested",
                        {
                            "message": "La verificación pidió una cita válida de un pasaje leído.",
                            "reason": "citation_required",
                            "turn": turn_number,
                        },
                    )
                    input_items.append(
                        {
                            "role": "user",
                            "content": (
                                "Corrige la respuesta final: incluye al menos una cita "
                                "válida de los chunks leídos que sostenga la respuesta."
                            ),
                        }
                    )
                    continue
                except ValueError as exc:
                    last_parse_error = str(exc)
                    _emit_progress(
                        on_progress,
                        "answer_revision_requested",
                        {
                            "message": "La respuesta provisional no cumplió el contrato y será corregida.",
                            "reason": "invalid_final_answer",
                            "turn": turn_number,
                        },
                    )
                    input_items.append(
                        {
                            "role": "user",
                            "content": f"Corrige la respuesta final: {last_parse_error}",
                        }
                    )
                    continue
                verification = _verification(answer, self.toolbox, required_hops)
                stop_reason = (
                    "completed"
                    if self.toolbox.read_chunk_ids and not self.toolbox.missing_coverage
                    else (
                        "evidence_not_read"
                        if not self.toolbox.read_chunk_ids
                        else "coverage_incomplete: "
                        + ",".join(self.toolbox.missing_coverage)
                    )
                )
                _emit_progress(
                    on_progress,
                    "verification_completed",
                    {
                        "message": "La respuesta y sus citas fueron verificadas.",
                        "turn": turn_number,
                        "citation_ids": answer.citations,
                        "coverage": self.toolbox.coverage,
                        "verification": verification,
                        "stop_reason": stop_reason,
                    },
                )
                return AgentRun(
                    question=question,
                    as_of=as_of,
                    model=self.backend.model,
                    answer=answer,
                    traces=traces,
                    turns=turns,
                    model_turns=turn_number,
                    tool_calls=len(traces),
                    stop_reason=stop_reason,
                    usage=usage,
                    elapsed_ms=(perf_counter() - started) * 1000.0,
                    coverage=self.toolbox.coverage,
                    required_hops=required_hops,
                    verification=verification,
                )
            last_parse_error = "model returned neither tool calls nor a final answer"
        answer = AgentAnswer(
            answer="No se obtuvo una respuesta final verificable dentro de los límites.",
            citations=[],
            invalid_citations=[],
            premise_status="unclear",
        )
        verification = _verification(answer, self.toolbox, required_hops)
        stop_reason = (
            terminal_stop_reason
            if terminal_stop_reason != "model_turn_limit"
            else f"model_turn_limit: {last_parse_error}"
        )
        _emit_progress(
            on_progress,
            "verification_completed",
            {
                "message": "La ejecución terminó sin una respuesta final verificable.",
                "citation_ids": [],
                "coverage": self.toolbox.coverage,
                "verification": verification,
                "stop_reason": stop_reason,
            },
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
            stop_reason=stop_reason,
            usage=usage,
            elapsed_ms=(perf_counter() - started) * 1000.0,
            coverage=self.toolbox.coverage,
            required_hops=required_hops,
            verification=verification,
        )


def _emit_progress(
    callback: Callable[[str, dict[str, Any]], None] | None,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    """Emit observable agent work without making telemetry a run dependency."""
    if callback is None:
        return
    try:
        callback(event_type, payload)
    except Exception:
        LOGGER.exception("agent progress callback failed for %s", event_type)


def _tool_start_message(name: str, arguments: dict[str, Any] | None = None) -> str:
    arguments = arguments or {}
    query = str(arguments.get("query") or "").strip()
    if name == "list_publications":
        return "Buscando publicaciones del DOF dentro del periodo relevante."
    if name == "search_documents":
        return (
            f"Buscando documentos sobre “{query}”."
            if query
            else "Buscando documentos relevantes."
        )
    if name == "search_evidence":
        return (
            f"Buscando pasajes sobre “{query}”."
            if query
            else "Buscando pasajes dentro de los documentos candidatos."
        )
    if name == "get_document_outline":
        return f"Revisando la estructura del documento {arguments.get('document_id', '')}.".replace(
            "  ", " "
        )
    if name == "read_chunks":
        chunk_ids = ", ".join(str(item) for item in arguments.get("chunk_ids", []))
        return f"Leyendo los chunks {chunk_ids} para comprobar la evidencia."
    return "Consultando una fuente del agente."


def _tool_reason(name: str) -> str:
    return {
        "list_publications": "Necesita acotar qué publicaciones existían antes de seleccionar evidencia.",
        "search_documents": "Necesita identificar documentos candidatos antes de buscar pasajes concretos.",
        "search_evidence": "Ya tiene documentos candidatos y ahora busca dónde se afirma lo relevante.",
        "get_document_outline": "La estructura ayuda a ubicar las secciones que conviene leer.",
        "read_chunks": "Sólo los chunks leídos pueden convertirse en evidencia y citas de la respuesta.",
    }.get(name, "Esta consulta aporta evidencia observable para el siguiente paso.")


def _public_tool_progress(
    name: str,
    arguments: dict[str, Any] | None,
    output: dict[str, Any],
    *,
    elapsed_ms: float,
    turn: int,
) -> dict[str, Any]:
    """Build a bounded public decision log; never expose private model reasoning."""
    payload: dict[str, Any] = {
        "message": "La herramienta no pudo completar esta consulta.",
        "tool": name,
        "arguments": arguments or {},
        "ok": bool(output.get("ok")),
        "elapsed_ms": round(elapsed_ms, 1),
        "turn": turn,
    }
    if not output.get("ok"):
        error = output.get("error", {})
        payload["error"] = {
            "type": error.get("type", "tool_error"),
            "message": error.get("message", "La consulta falló."),
        }
        return payload

    data = output.get("data", {})
    documents = data.get("documents", data.get("publications", []))
    if documents:
        payload["documents"] = [
            {
                key: item.get(key)
                for key in (
                    "document_id",
                    "title",
                    "publication_date",
                    "section",
                    "institution",
                    "path",
                )
            }
            for item in documents[:20]
        ]
        payload["result_count"] = len(documents)
    evidence = data.get("evidence", [])
    if evidence:
        payload["chunks"] = [
            {
                key: item.get(key)
                for key in (
                    "chunk_id",
                    "document_id",
                    "path",
                    "heading_path",
                    "snippet",
                    "snippet_truncated",
                    "source",
                    "rank",
                )
            }
            for item in evidence[:30]
        ]
        payload["result_count"] = len(evidence)
    # An outline also contains entries named "chunks", but those are structural
    # metadata, not passages the agent has read. Only read_chunks can promote
    # text to public, expandable evidence.
    chunks = data.get("chunks", []) if name == "read_chunks" else []
    if chunks:
        payload["chunks"] = [
            {
                **{
                    key: item.get(key)
                    for key in (
                        "chunk_id",
                        "document_id",
                        "path",
                        "heading_path",
                        "chunk_index",
                    )
                },
                "document_id": item.get("document_id") or data.get("document_id"),
                "excerpt": str(item.get("text") or "")[:1600],
                "excerpt_truncated": len(str(item.get("text") or "")) > 1600,
            }
            for item in chunks[:30]
        ]
        payload["result_count"] = len(chunks)
    if "document_id" in data:
        payload["document_id"] = data.get("document_id")
    if "coverage" in data:
        payload["coverage"] = data["coverage"]
    payload["message"] = _tool_completed_message(
        name, arguments or {}, payload.get("result_count", 0)
    )
    return payload


def _tool_completed_message(
    name: str, arguments: dict[str, Any], result_count: int
) -> str:
    noun = "resultado" if result_count == 1 else "resultados"
    if name == "list_publications":
        return f"Encontró {result_count} publicaciones dentro del periodo."
    if name == "search_documents":
        return f"Encontró {result_count} documentos candidatos."
    if name == "search_evidence":
        return f"Encontró {result_count} pasajes candidatos para revisar."
    if name == "get_document_outline":
        return f"Revisó la estructura del documento {arguments.get('document_id', '')}.".replace(
            "  ", " "
        )
    if name == "read_chunks":
        return f"Leyó {result_count} pasajes que ahora pueden sostener citas."
    return f"La consulta devolvió {result_count} {noun}."
