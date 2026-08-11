"""Minimal provider-neutral chat client and evidence-grounded answerer."""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any

from .models import SearchResult


@dataclass(frozen=True)
class Answer:
    answer: str
    citations: list[int]
    invalid_citations: list[int]
    premise_status: str
    raw: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "invalid_citations": self.invalid_citations,
            "premise_status": self.premise_status,
            "raw": self.raw,
        }


class ChatClient:
    """Call either an OpenAI-compatible or Anthropic Messages endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        protocol: str = "openai",
        timeout: int = 180,
    ):
        if protocol not in {"openai", "anthropic"}:
            raise ValueError("protocol must be 'openai' or 'anthropic'")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = (
            api_key
            or os.environ.get("LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        self.protocol = protocol
        self.timeout = timeout

    def complete(self, system: str, user: str, *, max_tokens: int = 1200) -> str:
        if not self.api_key:
            raise RuntimeError("set LLM_API_KEY or OPENAI_API_KEY")
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": user}],
        }
        if self.protocol == "openai":
            body["messages"].insert(0, {"role": "system", "content": system})
            url = f"{self.base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        else:
            body["system"] = system
            url = f"{self.base_url}/v1/messages"
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
        request = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"), headers=headers
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.load(response)
        if self.protocol == "openai":
            return payload["choices"][0]["message"].get("content", "")
        return "".join(
            block.get("text", "")
            for block in payload.get("content", [])
            if block.get("type") == "text"
        )


ANSWER_SYSTEM = """Eres un asistente de investigación del Diario Oficial de la Federación.
Responde únicamente con la evidencia proporcionada. No inventes datos ni citas.
Si la evidencia no basta, dilo claramente. Verifica las fechas: el corpus solo
es válido hasta la fecha de corte indicada. Si la pregunta contiene una premisa
falsa, corrígela en vez de aceptarla. Devuelve SOLO JSON válido con esta forma:
{"answer":"...","citations":[123],"premise_status":"supported|false|unclear"}
En citations usa únicamente los IDs de chunk que realmente sostienen la respuesta.
"""


def _context(result: SearchResult) -> str:
    return "\n\n".join(
        f"[CHUNK {hit.chunk_id}] documento={hit.document_id} "
        f"fecha={hit.publication_date} ruta={hit.path} sección={hit.section}\n{hit.text}"
        for hit in result.evidence
    )


def _parse_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("LLM response did not contain a valid JSON object")


def answer_with_context(
    client: ChatClient, question: str, result: SearchResult
) -> Answer:
    user = (
        f"Fecha de corte: {result.as_of or 'no indicada'}\n"
        f"Pregunta: {question}\n\nEVIDENCIA:\n{_context(result)}"
    )
    raw = client.complete(ANSWER_SYSTEM, user)
    data = _parse_json(raw)
    proposed_citations: list[int] = []
    for value in data.get("citations", []):
        try:
            proposed_citations.append(int(value))
        except (TypeError, ValueError):
            continue
    allowed = set(result.evidence_ids)
    citations = list(
        dict.fromkeys(
            citation for citation in proposed_citations if citation in allowed
        )
    )
    invalid_citations = list(
        dict.fromkeys(
            citation for citation in proposed_citations if citation not in allowed
        )
    )
    status = str(data.get("premise_status", "unclear"))
    if status not in {"supported", "false", "unclear"}:
        status = "unclear"
    return Answer(
        answer=str(data.get("answer", "")).strip(),
        citations=citations,
        invalid_citations=invalid_citations,
        premise_status=status,
        raw=raw,
    )
