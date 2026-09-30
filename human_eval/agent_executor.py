"""Adapter from persisted human-evaluation requests to the existing agent."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import subprocess
import threading
from time import monotonic
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agent_tools.agent import (
    AgentRunner,
    DofToolbox,
    OpenAIChatCompletionsBackend,
    OpenAIResponsesBackend,
)
from agent_tools.retrieval import DofRetriever, LlamaQueryEmbedder, QueryEmbedder

from .contracts import RunRequest
from .service import ProgressCallback, PublicExecutionError

DEFAULT_GGUF_MODEL = "~/dof-gguf/jina-v5-small-retrieval-F16.gguf"
DEFAULT_VEC0_DB = "dof_db/dof_vec0_jina_binary.sqlite"
RETRIEVAL_MODES = frozenset({"lexical", "vector", "hybrid"})
LOCAL_AGENT_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _endpoint_port(url: str) -> tuple[str | None, int | None]:
    parsed = urlparse(url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"invalid DOF_AGENT_BASE_URL: {exc}") from exc
    if port is None:
        if parsed.scheme == "https":
            port = 443
        elif parsed.scheme == "http":
            port = 80
    return parsed.hostname, port


@dataclass(frozen=True)
class AgentExecutorConfig:
    repo_root: Path
    provider: str
    model: str
    corpus_db: Path
    chunks_db: Path
    vec0_db: Path | None = None
    gguf_model: Path | None = None
    embed_port: int = 8086
    reasoning_effort: str | None = "low"
    max_model_turns: int = 8
    max_tool_calls: int = 8
    model_concurrency: int = 1
    retrieval_mode: str = "lexical"
    base_url: str | None = None

    @classmethod
    def from_env(cls, repo_root: str | Path) -> "AgentExecutorConfig":
        root = Path(repo_root).resolve()
        provider = os.environ.get("DOF_AGENT_PROVIDER", "llama-server")
        if provider not in {"openai-responses", "kimi-code", "llama-server"}:
            raise ValueError(
                "DOF_AGENT_PROVIDER must be openai-responses, kimi-code, "
                "or llama-server"
            )
        model = os.environ.get(
            "DOF_AGENT_MODEL",
            os.environ.get("OPENAI_MODEL", "Qwen3.8-Flash-Next" if provider == "llama-server" else ""),
        )
        if not model:
            raise ValueError("set DOF_AGENT_MODEL or OPENAI_MODEL")
        retrieval_mode = os.environ.get("DOF_RETRIEVAL_MODE", "lexical")
        if retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError(
                "DOF_RETRIEVAL_MODE must be one of: lexical, vector, hybrid"
            )
        vec0_value = os.environ.get("DOF_VEC0_DB")
        if vec0_value:
            vec0_db: Path | None = Path(vec0_value)
        elif retrieval_mode != "lexical":
            default_vec0 = root / DEFAULT_VEC0_DB
            vec0_db = default_vec0 if default_vec0.exists() else None
        else:
            vec0_db = None
        gguf_value = os.environ.get("DOF_GGUF_MODEL", DEFAULT_GGUF_MODEL)
        gguf_model = Path(gguf_value).expanduser() if gguf_value else None
        if retrieval_mode != "lexical":
            if vec0_db is None or not vec0_db.exists():
                raise ValueError(
                    f"DOF_RETRIEVAL_MODE={retrieval_mode} requires an existing "
                    "vector index (DOF_VEC0_DB)"
                )
            if gguf_model is None or not gguf_model.exists():
                raise ValueError(
                    f"DOF_RETRIEVAL_MODE={retrieval_mode} requires an existing "
                    "GGUF embedding model (DOF_GGUF_MODEL)"
                )
        embed_port = int(os.environ.get("DOF_EMBED_PORT", "8086"))
        if not 1 <= embed_port <= 65535:
            raise ValueError("DOF_EMBED_PORT must be between 1 and 65535")
        model_concurrency = int(os.environ.get("DOF_MODEL_CONCURRENCY", "1"))
        if model_concurrency < 1:
            raise ValueError("DOF_MODEL_CONCURRENCY must be positive")
        base_url = os.environ.get("DOF_AGENT_BASE_URL")
        if provider == "llama-server" and not base_url:
            base_url = "http://192.168.1.117:8888/v1"
        if provider == "llama-server" and retrieval_mode != "lexical":
            host, agent_port = _endpoint_port(
                base_url or "http://127.0.0.1:8080/v1"
            )
            if host in LOCAL_AGENT_HOSTS and agent_port == embed_port:
                raise ValueError(
                    "DOF_AGENT_BASE_URL and DOF_EMBED_PORT must use different "
                    "local ports"
                )
        return cls(
            repo_root=root,
            provider=provider,
            model=model,
            corpus_db=Path(
                os.environ.get("DOF_CORPUS_DB", root / "dof_db/dof_corpus_l3.sqlite")
            ),
            chunks_db=Path(
                os.environ.get("DOF_CHUNKS_DB", root / "dof_db/dof_chunks.sqlite")
            ),
            vec0_db=vec0_db,
            gguf_model=gguf_model,
            embed_port=embed_port,
            reasoning_effort=os.environ.get("DOF_REASONING_EFFORT", "low") or None,
            max_model_turns=int(os.environ.get("DOF_MAX_MODEL_TURNS", "8")),
            max_tool_calls=int(os.environ.get("DOF_MAX_TOOL_CALLS", "8")),
            model_concurrency=model_concurrency,
            retrieval_mode=retrieval_mode,
            base_url=base_url,
        )


def _git_snapshot(repo_root: Path) -> tuple[str, bool]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True
    return revision, dirty


def _read_index_versions(config: AgentExecutorConfig) -> dict[str, Any]:
    corpus_version: str | None = None
    chunker_version: str | None = None
    try:
        with sqlite3.connect(f"file:{config.corpus_db}?mode=ro", uri=True) as corpus:
            row = corpus.execute(
                "SELECT value FROM corpus_meta WHERE key = 'corpus_version'"
            ).fetchone()
            corpus_version = row[0] if row else None
        with sqlite3.connect(f"file:{config.chunks_db}?mode=ro", uri=True) as chunks:
            row = chunks.execute(
                "SELECT chunker_version, corpus_version FROM chunks LIMIT 1"
            ).fetchone()
            if row:
                chunker_version = row[0]
                corpus_version = row[1] or corpus_version
    except sqlite3.Error:
        pass
    vector_available = bool(config.vec0_db and config.vec0_db.exists())
    vector_version = None
    if vector_available and config.vec0_db:
        stat = config.vec0_db.stat()
        fingerprint = f"{config.vec0_db.name}:{stat.st_size}:{stat.st_mtime_ns}"
        vector_version = hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
    return {
        "corpus_version": corpus_version,
        "chunker_version": chunker_version,
        "vector_available": vector_available,
        "vector_index_version": vector_version,
    }


def provenance_for_config(
    config: AgentExecutorConfig, *, vector_used: bool = False
) -> dict[str, Any]:
    """Provenance derivable from configuration alone (no executor needed).

    Web processes use this for the capabilities endpoint; the scheduler's
    executor delegates with the live embedder state when stamping runs.
    """
    revision, dirty = _git_snapshot(config.repo_root)
    return {
        "code_revision": revision,
        "code_dirty": dirty,
        **_read_index_versions(config),
        # vector_available describes the on-disk asset; vector_used records
        # whether the executor could actually query it (embedder is live).
        "vector_used": vector_used,
        "provider": config.provider,
        "model": config.model,
        "configuration": {
            "retrieval_mode": config.retrieval_mode,
            "max_model_turns": config.max_model_turns,
            "max_tool_calls": config.max_tool_calls,
            "model_concurrency": config.model_concurrency,
            "reasoning_effort": config.reasoning_effort,
        },
    }


class AgentRunExecutor:
    """Run the agent, sharing one llama-server for query embeddings.

    The embedding server starts lazily on the first vector-capable run and
    stays alive for the executor's lifetime (spawning per run is slow).
    ``close`` stops it; ``EvaluationService.close`` calls it on shutdown.
    """

    def __init__(self, config: AgentExecutorConfig):
        self.config = config
        self._embedder: QueryEmbedder | None = None
        self._embedder_lock = threading.Lock()

    def query_embedder(self) -> QueryEmbedder | None:
        """Return the shared embedder, starting llama-server once if needed."""
        if self.config.retrieval_mode == "lexical":
            return None
        with self._embedder_lock:
            if self._embedder is None:
                if self.config.vec0_db is None or not self.config.vec0_db.exists():
                    raise PublicExecutionError(
                        "provider_unavailable",
                        "El índice vectorial no está disponible.",
                    )
                if (
                    self.config.gguf_model is None
                    or not self.config.gguf_model.exists()
                ):
                    raise PublicExecutionError(
                        "provider_unavailable",
                        "El modelo de embeddings no está disponible.",
                    )
                self._embedder = LlamaQueryEmbedder(
                    self.config.gguf_model, port=self.config.embed_port
                )
            return self._embedder

    def prepare(self) -> None:
        """Prepare resources needed before a run is persisted."""
        self.query_embedder()

    def close(self) -> None:
        """Stop the shared llama-server, if it was started."""
        with self._embedder_lock:
            if self._embedder is not None:
                self._embedder.close()
                self._embedder = None

    def provenance(self) -> dict[str, Any]:
        return provenance_for_config(
            self.config, vector_used=self._embedder is not None
        )

    def _backend(self) -> Any:
        if self.config.provider == "llama-server":
            # Any OpenAI-compatible local server (llama.cpp llama-server,
            # LM Studio, vLLM, ...). The API key is ignored by llama-server;
            # the placeholder only satisfies the OpenAI client.
            return OpenAIChatCompletionsBackend(
                model=self.config.model,
                api_key=os.environ.get("DOF_AGENT_API_KEY", "llama-server"),
                enable_thinking=True if self.config.model == "Qwen3.8-Flash-Next" else None,
                base_url=self.config.base_url or "http://127.0.0.1:8080/v1",
                reasoning_effort=self.config.reasoning_effort,
            )
        if self.config.provider == "kimi-code":
            api_key = os.environ.get("KIMI_API_KEY", "")
            if not api_key:
                raise PublicExecutionError(
                    "provider_unavailable",
                    "El proveedor del agente no está configurado.",
                )
            return OpenAIChatCompletionsBackend(
                model=self.config.model,
                api_key=api_key,
                base_url=self.config.base_url or "https://api.kimi.com/coding/v1",
            )
        return OpenAIResponsesBackend(
            model=self.config.model,
            base_url=self.config.base_url or os.environ.get("OPENAI_BASE_URL"),
            reasoning_effort=self.config.reasoning_effort,
        )

    def execute(
        self,
        request: RunRequest,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        flush_stream = None
        try:
            backend = self._backend()
            if request.is_chat or request.token_limit is not None:
                if on_progress is not None:
                    pending = {"content": "", "reasoning_content": ""}
                    last_flush = monotonic()
                    stream_turn = 1
                    def stream_delta(kind, text):
                        nonlocal last_flush, stream_turn
                        if kind in pending:
                            pending[kind] += text
                        if kind == "done" or monotonic() - last_flush >= 0.1:
                            for field, value in pending.items():
                                if value:
                                    on_progress("model_turn_started", {
                                        "chat_delta": field, "text": value, "turn": stream_turn,
                                    })
                                    pending[field] = ""
                            last_flush = monotonic()
                        if kind == "done":
                            stream_turn += 1
                    backend.on_delta = stream_delta
                    flush_stream = stream_delta
                from .budget_backend import BudgetBackend
                from .qwen_token_counter import QwenTokenCounter
                if (self.config.provider != "llama-server"
                        or self.config.model != "Qwen3.8-Flash-Next"):
                    raise PublicExecutionError(
                        "unsupported_token_counter",
                        "El modelo no tiene un contador de tokens verificado.",
                    )
                backend = BudgetBackend(
                    backend, request.token_limit,
                    QwenTokenCounter(enable_thinking=backend.enable_thinking is not False,
                                     reasoning_effort=backend.reasoning_effort),
                )
            embedder = self.query_embedder()
            with DofRetriever(
                corpus_db=self.config.corpus_db,
                chunks_db=self.config.chunks_db,
                vec0_db=(
                    self.config.vec0_db
                    if self.config.retrieval_mode != "lexical"
                    else None
                ),
            ) as retriever:
                if request.is_chat or request.token_limit is not None:
                    from .chat_runner import run_chat
                    return _public_result(run_chat(
                        backend, DofToolbox(retriever, embedder=embedder), request,
                        max_turns=self.config.max_model_turns,
                        max_tool_calls=self.config.max_tool_calls,
                        on_progress=on_progress,
                    ))
                run = AgentRunner(
                    backend,
                    DofToolbox(retriever, embedder=embedder),
                    max_model_turns=self.config.max_model_turns,
                    max_tool_calls=self.config.max_tool_calls,
                ).run(
                    request.question,
                    as_of=request.as_of,
                    required_hops=request.required_hops,
                    on_progress=on_progress,
                )
        except PublicExecutionError:
            raise
        except Exception as exc:
            from .budget_backend import TokenCeilingReached
            if isinstance(exc, TokenCeilingReached):
                raise PublicExecutionError(
                    "context_limit_exhausted" if request.is_chat and request.token_limit is None else "token_budget_exhausted",
                    "La consulta alcanzó el contexto máximo del modelo." if request.is_chat and request.token_limit is None else "La consulta alcanzó su presupuesto de tokens.",
                    used_tokens=exc.used,
                ) from exc
            name = type(exc).__name__
            status_code = getattr(exc, "status_code", None)
            if name in {
                "APIConnectionError",
                "APITimeoutError",
                "APIStatusError",
                "AuthenticationError",
                "BadRequestError",
                "InternalServerError",
                "NotFoundError",
                "PermissionDeniedError",
                "RateLimitError",
            }:
                code = (
                    "rate_limited"
                    if name == "RateLimitError" or status_code == 429
                    else "provider_unavailable"
                )
                raise PublicExecutionError(
                    code, "El proveedor del agente no está disponible."
                ) from exc
            raise
        finally:
            if flush_stream is not None:
                flush_stream("done", "")
        return _public_result(run.to_dict())


def _public_result(run: dict[str, Any]) -> dict[str, Any]:
    citations = set(run["answer"]["citations"])
    evidence: dict[int, dict[str, Any]] = {}
    documents: dict[int, dict[str, Any]] = {}
    for trace in run["traces"]:
        if not trace["output"].get("ok"):
            continue
        data = trace["output"].get("data", {})
        for document in [*data.get("documents", []), *data.get("publications", [])]:
            document_id = int(document["document_id"])
            documents.setdefault(
                document_id,
                {
                    key: document.get(key)
                    for key in (
                        "document_id",
                        "path",
                        "publication_date",
                        "section",
                        "title",
                        "institution",
                    )
                },
            )
        # Only read_chunks returns citable text. Outlines also use a "chunks"
        # key, but those entries intentionally omit document_id and text.
        readable_chunks = (
            data.get("chunks", []) if trace["name"] == "read_chunks" else []
        )
        for chunk in readable_chunks:
            chunk_id = int(chunk["chunk_id"])
            item = dict(chunk)
            item["cited"] = chunk_id in citations
            evidence[chunk_id] = item
            document_id = int(chunk["document_id"])
            documents.setdefault(
                document_id,
                {
                    "document_id": document_id,
                    "path": chunk.get("path"),
                    "publication_date": chunk.get("publication_date"),
                    "section": chunk.get("section"),
                    "title": None,
                    "institution": None,
                },
            )
    cited_document_ids = {
        int(item["document_id"])
        for item in evidence.values()
        if item["chunk_id"] in citations
    }
    for document_id, document in documents.items():
        document["used_as_evidence"] = any(
            item["document_id"] == document_id for item in evidence.values()
        )
        document["cited"] = document_id in cited_document_ids
    missing = [key for key, complete in run.get("coverage", {}).items() if not complete]
    warnings: list[str] = []
    if missing:
        warnings.append("coverage_incomplete")
    if run["answer"].get("invalid_citations"):
        warnings.append("invalid_citations_removed")
    if run.get("verification", {}).get("premise_status_review_required"):
        warnings.append("premise_status_review_required")
    if run["stop_reason"] != "completed":
        warnings.append(run["stop_reason"])
    return {
        "answer": {
            "text": run["answer"]["answer"],
            "citation_ids": run["answer"]["citations"],
            "premise_status": run["answer"]["premise_status"],
        },
        "evidence": sorted(evidence.values(), key=lambda item: item["chunk_id"]),
        "documents": sorted(documents.values(), key=lambda item: item["document_id"]),
        "coverage": {
            "required": sorted(run.get("coverage", {})),
            "missing": missing,
            "complete": run["stop_reason"] == "completed" and not missing,
        },
        "verification": run.get("verification", {}),
        "trace": run["traces"],
        "warnings": warnings,
        "stop_reason": run["stop_reason"],
        "model_turns": run["model_turns"],
        "tool_calls": run["tool_calls"],
        "usage": run["usage"],
        "elapsed_ms": run["elapsed_ms"],
        "reasoning": run.get("reasoning", []),
    }
