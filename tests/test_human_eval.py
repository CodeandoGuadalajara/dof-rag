from __future__ import annotations

import os
import re
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from starlette.testclient import TestClient

from human_eval.agent_executor import (
    AgentExecutorConfig,
    AgentRunExecutor,
    _public_result,
)
from human_eval.app import (
    WebSettings,
    _attach_chunk_html,
    _progress_chunk_html,
    _progress_timeline,
    _sanitize_login_next,
    create_app,
)
from human_eval.auth import FakeAuthBackend, User
from human_eval.contracts import ContractError, FeedbackRequest, RunRequest
from human_eval.markdown_render import render_markdown_html
from human_eval.scheduler import RunScheduler
from human_eval.service import (
    ActiveRunError,
    EvaluationService,
    IdempotencyConflictError,
    PublicExecutionError,
    QueueFullError,
)
from human_eval.store import SCHEMA_VERSION, EvaluationStore
from scripts.seed_human_eval_v4_hybrid import SEED_USER, seed_live_run

PROVENANCE = {
    "code_revision": "abc123",
    "code_dirty": False,
    "corpus_version": "corpus-v1",
    "chunker_version": "chunks-v1",
    "vector_available": False,
    "vector_index_version": None,
    "vector_used": False,
    "provider": "fake",
    "model": "fake-model",
    "configuration": {
        "retrieval_mode": "lexical",
        "max_model_turns": 8,
        "max_tool_calls": 8,
    },
}


class FakeExecutor:
    def execute(self, request: RunRequest, *, on_progress=None):
        if on_progress:
            on_progress(
                "agent_started",
                {"message": "El agente comenzó a investigar la pregunta."},
            )
            on_progress(
                "tool_completed",
                {
                    "message": "La consulta terminó.",
                    "why": "Sólo los chunks leídos pueden sostener citas.",
                    "tool": "read_chunks",
                    "ok": True,
                    "chunks": [
                        {
                            "chunk_id": 123,
                            "document_id": 45,
                            "path": "2026/documento.md",
                            "excerpt": "Pasaje verificable.",
                        }
                    ],
                },
            )
        return {
            "answer": {
                "text": f"Respuesta: {request.question}",
                "citation_ids": [123],
                "premise_status": "supported",
            },
            "evidence": [{"chunk_id": 123, "document_id": 45, "cited": True}],
            "documents": [{"document_id": 45, "cited": True}],
            "coverage": {"required": [], "missing": [], "complete": True},
            "trace": [],
        }

    def provenance(self):
        return dict(PROVENANCE)


class BlockingExecutor(FakeExecutor):
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def execute(self, request: RunRequest, *, on_progress=None):
        self.started.set()
        if not self.release.wait(timeout=3):
            raise RuntimeError("test timed out")
        return super().execute(request, on_progress=on_progress)


def wait_for_terminal(service: EvaluationService, run_id: str) -> dict:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        run = service.public_run(run_id, admin=True)
        if run["status"] in {"succeeded", "failed"}:
            return run
        time.sleep(0.01)
    raise AssertionError("run did not reach a terminal state")


def start_scheduler(
    store: EvaluationStore, executor, **kwargs
) -> tuple[RunScheduler, threading.Thread]:
    """Run a background scheduler against the store, like production."""
    scheduler = RunScheduler(store, executor, poll_seconds=0.02, **kwargs)
    scheduler.prepare()
    thread = threading.Thread(
        target=scheduler.run, name="test-scheduler", daemon=True
    )
    thread.start()
    return scheduler, thread


def stop_scheduler(scheduler: RunScheduler, thread: threading.Thread) -> None:
    scheduler.stop()
    thread.join(timeout=3)


class SchedulerTestCase(unittest.TestCase):
    """Web service plus a background scheduler against one temporary store."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.store = EvaluationStore(Path(self.tempdir.name) / "evaluation.sqlite")

    def make_service(self, executor=None, **service_kwargs):
        executor = executor or FakeExecutor()
        harness = start_scheduler(self.store, executor)
        self.addCleanup(stop_scheduler, *harness)
        service = EvaluationService(self.store, **service_kwargs)
        service.start()
        self.addCleanup(service.close)
        return service, executor
class ContractTests(unittest.TestCase):
    def test_run_request_validates_and_normalizes(self):
        request = RunRequest.from_dict(
            {
                "question": "  ¿Qué establece el decreto?  ",
                "as_of": "2026-08-17",
                "required_hops": 2,
                "client_request_id": "browser:123",
            }
        )
        self.assertEqual(request.question, "¿Qué establece el decreto?")
        self.assertEqual(request.required_hops, 2)

    def test_run_request_rejects_tool_parameters(self):
        with self.assertRaises(ContractError):
            RunRequest.from_dict({"question": "pregunta válida", "top_k": 1000})

    def test_feedback_uses_closed_vocabularies(self):
        feedback = FeedbackRequest.from_dict(
            {
                "rating": "partially_helpful",
                "problem_types": ["missing_evidence", "missing_evidence"],
                "comment": "Falta una fuente.",
            }
        )
        self.assertEqual(feedback.problem_types, ("missing_evidence",))
        with self.assertRaises(ContractError):
            FeedbackRequest.from_dict({"rating": "five_stars"})


class AgentResultTests(unittest.TestCase):
    def test_public_result_resolves_citations_and_coverage(self):
        result = _public_result(
            {
                "answer": {
                    "answer": "Respuesta con evidencia.",
                    "citations": [123],
                    "invalid_citations": [999],
                    "premise_status": "supported",
                },
                "traces": [
                    {
                        "name": "get_document_outline",
                        "output": {
                            "ok": True,
                            "data": {
                                "document_id": 45,
                                "chunks": [
                                    {
                                        "chunk_id": 122,
                                        "chunk_index": 0,
                                        "heading_path": ["Encabezado"],
                                    }
                                ],
                            },
                        },
                    },
                    {
                        "name": "search_documents",
                        "output": {
                            "ok": True,
                            "data": {
                                "documents": [
                                    {
                                        "document_id": 45,
                                        "path": "2026/documento.md",
                                        "publication_date": "2026-01-01",
                                        "section": "MAT",
                                        "title": "Documento de prueba",
                                        "institution": "Institución",
                                    }
                                ]
                            },
                        },
                    },
                    {
                        "name": "read_chunks",
                        "output": {
                            "ok": True,
                            "data": {
                                "chunks": [
                                    {
                                        "chunk_id": 123,
                                        "document_id": 45,
                                        "path": "2026/documento.md",
                                        "text": "Pasaje verificable.",
                                    }
                                ]
                            },
                        },
                    },
                ],
                "coverage": {"año 2025": False},
                "verification": {"citation_from_read_chunk": True},
                "stop_reason": "coverage_incomplete: año 2025",
                "model_turns": 3,
                "tool_calls": 2,
                "usage": {"total_tokens": 100},
                "elapsed_ms": 12.5,
            }
        )
        self.assertTrue(result["evidence"][0]["cited"])
        self.assertEqual([item["chunk_id"] for item in result["evidence"]], [123])
        self.assertEqual(result["documents"][0]["title"], "Documento de prueba")
        self.assertEqual(result["coverage"]["missing"], ["año 2025"])
        self.assertFalse(result["coverage"]["complete"])
        self.assertIn("invalid_citations_removed", result["warnings"])
        self.assertNotIn("premise_status_review_required", result["warnings"])

    def test_public_result_flags_a_premise_status_needing_review(self):
        result = _public_result(
            {
                "answer": {
                    "answer": "No reformó el artículo 123; reformó el 76.",
                    "citations": [123],
                    "invalid_citations": [],
                    "premise_status": "unclear",
                },
                "traces": [
                    {
                        "name": "read_chunks",
                        "output": {
                            "ok": True,
                            "data": {
                                "chunks": [
                                    {
                                        "chunk_id": 123,
                                        "document_id": 45,
                                        "path": "2026/documento.md",
                                        "text": "Pasaje verificable.",
                                    }
                                ]
                            },
                        },
                    },
                ],
                "coverage": {},
                "verification": {"premise_status_review_required": True},
                "stop_reason": "completed",
                "model_turns": 3,
                "tool_calls": 1,
                "usage": {"total_tokens": 100},
                "elapsed_ms": 12.5,
            }
        )
        self.assertEqual(result["answer"]["premise_status"], "unclear")
        self.assertIn("premise_status_review_required", result["warnings"])

    def test_provenance_distinguishes_available_from_used_vector_index(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            vector = root / "vectors.sqlite"
            vector.touch()
            executor = AgentRunExecutor(
                AgentExecutorConfig(
                    repo_root=root,
                    provider="openai-responses",
                    model="test-model",
                    corpus_db=root / "missing-corpus.sqlite",
                    chunks_db=root / "missing-chunks.sqlite",
                    vec0_db=vector,
                    retrieval_mode="lexical",
                )
            )

            provenance = executor.provenance()

        self.assertTrue(provenance["vector_available"])
        self.assertFalse(provenance["vector_used"])


class AgentExecutorConfigTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def _config(self, **env: str) -> AgentExecutorConfig:
        merged = {"DOF_AGENT_MODEL": "test-model", **env}
        with mock.patch.dict(os.environ, merged, clear=True):
            return AgentExecutorConfig.from_env(self.root)

    def test_defaults_to_lexical_with_default_asset_paths(self):
        config = self._config()
        self.assertEqual(config.retrieval_mode, "lexical")
        self.assertIsNone(config.vec0_db)
        self.assertEqual(
            config.gguf_model,
            Path("~/dof-gguf/jina-v5-small-retrieval-F16.gguf").expanduser(),
        )
        self.assertEqual(config.embed_port, 8086)

    def test_lexical_mode_does_not_default_to_vector_database(self):
        vec0_dir = self.root.resolve() / "dof_db"
        vec0_dir.mkdir()
        default_vec0 = vec0_dir / "dof_vec0_jina_binary.sqlite"
        default_vec0.touch()
        config = self._config()
        self.assertIsNone(config.vec0_db)

    def test_non_lexical_mode_defaults_to_canonical_vector_database(self):
        vec0_dir = self.root.resolve() / "dof_db"
        vec0_dir.mkdir()
        default_vec0 = vec0_dir / "dof_vec0_jina_binary.sqlite"
        default_vec0.touch()
        gguf = self.root / "model.gguf"
        gguf.touch()
        config = self._config(DOF_RETRIEVAL_MODE="hybrid", DOF_GGUF_MODEL=str(gguf))
        self.assertEqual(config.vec0_db, default_vec0)

    def test_rejects_unknown_retrieval_mode(self):
        with self.assertRaises(ValueError):
            self._config(DOF_RETRIEVAL_MODE="weird")

    def test_accepts_llama_server_provider(self):
        config = self._config(DOF_AGENT_PROVIDER="llama-server")
        self.assertEqual(config.provider, "llama-server")

    def test_rejects_unknown_provider(self):
        with self.assertRaisesRegex(ValueError, "DOF_AGENT_PROVIDER"):
            self._config(DOF_AGENT_PROVIDER="weird")

    def test_llama_server_backend_uses_local_openai_compatible_endpoint(self):
        executor = AgentRunExecutor(
            AgentExecutorConfig(
                repo_root=self.root,
                provider="llama-server",
                model="ornith",
                corpus_db=self.root / "missing-corpus.sqlite",
                chunks_db=self.root / "missing-chunks.sqlite",
            )
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            backend = executor._backend()
        self.assertEqual(backend.model, "ornith")
        self.assertEqual(
            str(backend.client.base_url), "http://127.0.0.1:8080/v1/"
        )
        self.assertEqual(backend.reasoning_effort, "low")

    def test_llama_server_backend_honors_base_url_override(self):
        executor = AgentRunExecutor(
            AgentExecutorConfig(
                repo_root=self.root,
                provider="llama-server",
                model="ornith",
                corpus_db=self.root / "missing-corpus.sqlite",
                chunks_db=self.root / "missing-chunks.sqlite",
                base_url="http://127.0.0.1:9999/v1/",
            )
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            backend = executor._backend()
        self.assertEqual(str(backend.client.base_url), "http://127.0.0.1:9999/v1/")

    def test_empty_reasoning_effort_disables_the_optional_field(self):
        config = self._config(
            DOF_AGENT_PROVIDER="llama-server", DOF_REASONING_EFFORT=""
        )
        self.assertIsNone(config.reasoning_effort)

    def test_non_lexical_mode_requires_existing_vector_index(self):
        with self.assertRaisesRegex(ValueError, "vector index"):
            self._config(DOF_RETRIEVAL_MODE="hybrid")

    def test_non_lexical_mode_requires_existing_gguf(self):
        vec0_dir = self.root.resolve() / "dof_db"
        vec0_dir.mkdir()
        (vec0_dir / "dof_vec0_jina_binary.sqlite").touch()
        with self.assertRaisesRegex(ValueError, "GGUF"):
            self._config(
                DOF_RETRIEVAL_MODE="hybrid",
                DOF_GGUF_MODEL=str(self.root / "missing.gguf"),
            )

    def test_non_lexical_mode_accepts_configured_assets(self):
        vec0 = self.root / "vectors.sqlite"
        gguf = self.root / "model.gguf"
        vec0.touch()
        gguf.touch()
        config = self._config(
            DOF_RETRIEVAL_MODE="hybrid",
            DOF_VEC0_DB=str(vec0),
            DOF_GGUF_MODEL=str(gguf),
            DOF_EMBED_PORT="9999",
        )
        self.assertEqual(config.retrieval_mode, "hybrid")
        self.assertEqual(config.vec0_db, vec0)
        self.assertEqual(config.gguf_model, gguf)
        self.assertEqual(config.embed_port, 9999)

    def test_rejects_invalid_embedding_port(self):
        with self.assertRaisesRegex(ValueError, "DOF_EMBED_PORT"):
            self._config(DOF_EMBED_PORT="0")

    def test_rejects_shared_local_chat_and_embedding_port(self):
        vec0_dir = self.root.resolve() / "dof_db"
        vec0_dir.mkdir()
        (vec0_dir / "dof_vec0_jina_binary.sqlite").touch()
        gguf = self.root / "model.gguf"
        gguf.touch()
        with self.assertRaisesRegex(ValueError, "different local ports"):
            self._config(
                DOF_AGENT_PROVIDER="llama-server",
                DOF_RETRIEVAL_MODE="hybrid",
                DOF_GGUF_MODEL=str(gguf),
                DOF_AGENT_BASE_URL="http://127.0.0.1:8080/v1",
                DOF_EMBED_PORT="8080",
            )


class AgentExecutorEmbedderTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.vec0 = self.root / "vectors.sqlite"
        self.gguf = self.root / "model.gguf"
        self.vec0.touch()
        self.gguf.touch()

    def tearDown(self):
        self.tempdir.cleanup()

    def _executor(self, mode: str, **overrides) -> AgentRunExecutor:
        fields = {
            "repo_root": self.root,
            "provider": "openai-responses",
            "model": "test-model",
            "corpus_db": self.root / "missing-corpus.sqlite",
            "chunks_db": self.root / "missing-chunks.sqlite",
            "vec0_db": self.vec0,
            "gguf_model": self.gguf,
            "retrieval_mode": mode,
            **overrides,
        }
        return AgentRunExecutor(AgentExecutorConfig(**fields))

    def test_lexical_mode_never_starts_an_embedder(self):
        executor = self._executor("lexical")
        self.assertIsNone(executor.query_embedder())
        self.assertFalse(executor.provenance()["vector_used"])

    def test_embedder_starts_once_per_executor_and_closes(self):
        fake = mock.Mock()
        with mock.patch(
            "human_eval.agent_executor.LlamaQueryEmbedder", return_value=fake
        ) as factory:
            executor = self._executor("hybrid")
            self.assertFalse(executor.provenance()["vector_used"])
            executor.prepare()
            self.assertTrue(executor.provenance()["vector_used"])
            first = executor.query_embedder()
            second = executor.query_embedder()
            factory.assert_called_once_with(self.gguf, port=8086)
            self.assertTrue(executor.provenance()["vector_used"])
            executor.close()
            executor.close()
        self.assertIs(first, fake)
        self.assertIs(second, fake)
        fake.close.assert_called_once()
        self.assertFalse(executor.provenance()["vector_used"])

    def test_missing_vector_index_fails_publicly(self):
        executor = self._executor("hybrid", vec0_db=self.root / "missing.sqlite")
        with self.assertRaises(PublicExecutionError) as caught:
            executor.query_embedder()
        self.assertEqual(caught.exception.code, "provider_unavailable")

    def test_missing_gguf_fails_publicly(self):
        executor = self._executor("hybrid", gguf_model=self.root / "missing.gguf")
        with self.assertRaises(PublicExecutionError) as caught:
            executor.query_embedder()
        self.assertEqual(caught.exception.code, "provider_unavailable")


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "evaluation.sqlite"
        self.store = EvaluationStore(self.path)
        self.store.initialize()

    def tearDown(self):
        self.tempdir.cleanup()

    def test_run_events_and_feedback_are_append_only(self):
        request = RunRequest("pregunta válida", required_hops=2)
        run, created = self.store.create_run(
            request, user_id="evaluator", provenance=PROVENANCE
        )
        self.assertTrue(created)
        self.store.append_event(run["run_id"], "started")
        self.store.append_event(run["run_id"], "succeeded", {"answer": {}})
        feedback = self.store.add_feedback(
            run["run_id"],
            FeedbackRequest("not_helpful", ("incorrect_answer",), "No coincide."),
            user_id="evaluator",
        )
        finished = self.store.get_run(run["run_id"])
        self.assertEqual(finished["status"], "succeeded")
        self.assertEqual(finished["result"], {"answer": {}})
        self.assertEqual(
            self.store.feedback_for_run(run["run_id"])[0]["feedback_id"],
            feedback["feedback_id"],
        )
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM run_events WHERE run_id = ?",
                    (run["run_id"],),
                ).fetchone()[0],
                3,
            )

    def test_progress_events_are_ordered_and_replayable(self):
        run, _ = self.store.create_run(
            RunRequest("pregunta válida"),
            user_id="evaluator",
            provenance=PROVENANCE,
        )
        first = self.store.append_progress(
            run["run_id"], "agent_started", {"message": "inicio"}
        )
        second = self.store.append_progress(
            run["run_id"], "model_turn_started", {"turn": 1}
        )
        self.assertEqual(first["sequence"], 1)
        self.assertEqual(second["sequence"], 2)
        self.assertEqual(
            [event["sequence"] for event in self.store.progress_for_run(run["run_id"])],
            [1, 2],
        )
        self.assertEqual(
            self.store.progress_for_run(run["run_id"], after=1)[0]["event_type"],
            "model_turn_started",
        )

    def test_schema_one_is_migrated_without_losing_runs(self):
        run, _ = self.store.create_run(
            RunRequest("pregunta conservada"),
            user_id="evaluator",
            provenance=PROVENANCE,
        )
        with sqlite3.connect(self.path) as connection:
            connection.execute("DROP TABLE run_progress")
            connection.execute(
                "UPDATE schema_meta SET value = '1' WHERE key = 'schema_version'"
            )
        self.store.initialize()
        with sqlite3.connect(self.path) as connection:
            version = connection.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()[0]
            progress_table = connection.execute(
                "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'run_progress'"
            ).fetchone()
        self.assertEqual(version, SCHEMA_VERSION)
        self.assertIsNotNone(progress_table)
        self.assertEqual(
            self.store.get_run(run["run_id"])["question"], "pregunta conservada"
        )

    def test_schema_two_is_migrated_with_user_ids_and_publish_columns(self):
        with sqlite3.connect(self.path) as connection:
            connection.executescript(
                """
                DROP TABLE IF EXISTS schema_meta;
                DROP TABLE IF EXISTS runs;
                DROP TABLE IF EXISTS run_events;
                DROP TABLE IF EXISTS run_progress;
                DROP TABLE IF EXISTS feedback;
                CREATE TABLE schema_meta (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT INTO schema_meta VALUES ('schema_version', '2');
                CREATE TABLE runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    question TEXT NOT NULL,
                    as_of TEXT,
                    required_hops INTEGER NOT NULL,
                    evaluator_hash TEXT NOT NULL,
                    client_request_id TEXT,
                    provenance_json TEXT NOT NULL,
                    UNIQUE (evaluator_hash, client_request_id));
                INSERT INTO runs VALUES ('run-1', '2026-08-01T00:00:00Z',
                    'pregunta heredada', NULL, 1, 'tokenhash123', NULL, '{}');
                CREATE TABLE run_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    UNIQUE (run_id, sequence));
                INSERT INTO run_events(run_id, sequence, event_type, created_at,
                    payload_json) VALUES ('run-1', 1, 'queued',
                    '2026-08-01T00:00:00Z', '{}'),
                    ('run-1', 2, 'succeeded', '2026-08-01T00:05:00Z',
                    '{"answer": {}}');
                CREATE TABLE feedback (
                    feedback_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    created_at TEXT NOT NULL,
                    evaluator_hash TEXT NOT NULL,
                    rating TEXT NOT NULL,
                    problem_types_json TEXT NOT NULL,
                    comment TEXT NOT NULL);
                INSERT INTO feedback VALUES ('fb-1', 'run-1',
                    '2026-08-01T01:00:00Z', 'tokenhash123', 'helpful', '[]', '');
                """
            )
        self.store.initialize()
        with sqlite3.connect(self.path) as connection:
            run_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(runs)")
            }
            feedback_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(feedback)")
            }
            version = connection.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()[0]
        self.assertEqual(version, SCHEMA_VERSION)
        self.assertNotIn("evaluator_hash", run_columns)
        self.assertNotIn("evaluator_hash", feedback_columns)
        self.assertTrue({"user_id", "published_at", "published_by"} <= run_columns)
        self.assertIn("user_id", feedback_columns)
        run = self.store.get_run("run-1")
        self.assertEqual(run["question"], "pregunta heredada")
        self.assertIsNone(run["published_at"])
        self.assertEqual(
            self.store.feedback_for_run("run-1")[0]["user_id"], "tokenhash123"
        )

    def test_delete_seed_runs_removes_only_seed_fixtures(self):
        seed_run, _ = self.store.create_run(
            RunRequest("pregunta semilla"),
            user_id="seed:eval-v4",
            provenance=PROVENANCE,
        )
        self.store.append_event(seed_run["run_id"], "started")
        self.store.append_progress(
            seed_run["run_id"], "agent_started", {"message": "inicio"}
        )
        self.store.append_event(seed_run["run_id"], "succeeded", {"answer": {}})
        self.store.add_feedback(
            seed_run["run_id"],
            FeedbackRequest("helpful", (), ""),
            user_id="evaluator",
        )
        real_run, _ = self.store.create_run(
            RunRequest("pregunta real"), user_id="evaluator", provenance=PROVENANCE
        )

        with self.assertRaises(ValueError):
            self.store.delete_seed_runs(user_prefix="evaluator")
        self.assertEqual(self.store.delete_seed_runs(), 1)

        self.assertIsNone(self.store.get_run(seed_run["run_id"]))
        self.assertEqual(self.store.progress_for_run(seed_run["run_id"]), [])
        self.assertEqual(self.store.feedback_for_run(seed_run["run_id"]), [])
        self.assertIsNotNone(self.store.get_run(real_run["run_id"]))

    def test_delete_seed_run_removes_only_the_requested_seed_run(self):
        first, _ = self.store.create_run(
            RunRequest("primera semilla"),
            user_id="seed:eval-v4",
            provenance=PROVENANCE,
        )
        second, _ = self.store.create_run(
            RunRequest("segunda semilla"),
            user_id="seed:eval-v4",
            provenance=PROVENANCE,
        )
        self.assertTrue(self.store.delete_seed_run(first["run_id"]))
        self.assertIsNone(self.store.get_run(first["run_id"]))
        self.assertIsNotNone(self.store.get_run(second["run_id"]))
        self.assertFalse(self.store.delete_seed_run(first["run_id"]))
        with self.assertRaises(ValueError):
            self.store.delete_seed_run(second["run_id"], user_prefix="evaluator")

    def test_delete_run_removes_run_and_related_data(self):
        run, _ = self.store.create_run(
            RunRequest("pregunta a borrar"), user_id="evaluator", provenance=PROVENANCE
        )
        self.store.append_event(run["run_id"], "started")
        self.store.append_progress(
            run["run_id"], "agent_started", {"message": "inicio"}
        )
        self.store.append_event(run["run_id"], "succeeded", {"answer": {}})
        self.store.add_feedback(
            run["run_id"], FeedbackRequest("helpful", (), ""), user_id="evaluator"
        )

        self.store.delete_run(run["run_id"])

        self.assertIsNone(self.store.get_run(run["run_id"]))
        self.assertEqual(self.store.progress_for_run(run["run_id"]), [])
        self.assertEqual(self.store.feedback_for_run(run["run_id"]), [])
        with sqlite3.connect(self.path) as connection:
            for table in ("runs", "run_events", "run_progress", "feedback"):
                count = connection.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE run_id = ?",
                    (run["run_id"],),
                ).fetchone()[0]
                self.assertEqual(count, 0, table)
        with self.assertRaises(KeyError):
            self.store.delete_run(run["run_id"])

    def test_delete_run_refuses_active_runs(self):
        run, _ = self.store.create_run(
            RunRequest("pregunta en curso"), user_id="evaluator", provenance=PROVENANCE
        )
        with self.assertRaises(ValueError):
            self.store.delete_run(run["run_id"])  # queued
        self.store.append_event(run["run_id"], "started")
        with self.assertRaises(ValueError):
            self.store.delete_run(run["run_id"])  # running
        self.assertIsNotNone(self.store.get_run(run["run_id"]))

        self.store.append_event(run["run_id"], "failed", {"code": "x", "message": "y"})
        self.store.delete_run(run["run_id"])
        self.assertIsNone(self.store.get_run(run["run_id"]))

    def test_admin_runs_lists_all_runs_with_status(self):
        first, _ = self.store.create_run(
            RunRequest("primera pregunta"), user_id="evaluator", provenance=PROVENANCE
        )
        second, _ = self.store.create_run(
            RunRequest("segunda pregunta"), user_id="otra", provenance=PROVENANCE
        )
        self.store.append_event(first["run_id"], "started")
        self.store.append_event(first["run_id"], "succeeded", {"answer": {}})
        self.store.publish_run(first["run_id"], publisher_id="root")

        runs = self.store.admin_runs()
        self.assertEqual(
            [run["run_id"] for run in runs], [second["run_id"], first["run_id"]]
        )
        self.assertEqual(runs[0]["status"], "queued")
        self.assertEqual(runs[0]["user_id"], "otra")
        self.assertIsNone(runs[0]["published_at"])
        self.assertEqual(runs[1]["status"], "succeeded")
        self.assertIsNotNone(runs[1]["published_at"])
        with self.assertRaises(ValueError):
            self.store.admin_runs(limit=0)

    def test_failed_seed_run_is_retried_on_the_next_attempt(self):
        class FlakySeedExecutor:
            def __init__(self):
                self.calls = 0

            def provenance(self):
                return dict(PROVENANCE)

            def execute(self, request, *, on_progress=None):
                self.calls += 1
                if self.calls == 1:
                    raise PublicExecutionError("internal_error", "falló")
                return {"answer": {"text": "respuesta"}}

        executor = FlakySeedExecutor()
        item = {"id": "LI-001", "question": "pregunta semilla"}
        self.assertEqual(
            seed_live_run(self.store, executor, item, publish=False), "failed"
        )
        self.assertEqual(
            seed_live_run(self.store, executor, item, publish=False), "created"
        )
        record = self.store.find_idempotent_run(SEED_USER, "eval-v4-hybrid:LI-001")
        self.assertEqual(record["status"], "succeeded")
        self.assertEqual(executor.calls, 2)

    def test_client_request_id_is_idempotent_per_evaluator(self):
        request = RunRequest("pregunta válida", client_request_id="same-request")
        first, first_created = self.store.create_run(
            request, user_id="one", provenance=PROVENANCE
        )
        second, second_created = self.store.create_run(
            request, user_id="one", provenance=PROVENANCE
        )
        third, third_created = self.store.create_run(
            request, user_id="two", provenance=PROVENANCE
        )
        self.assertTrue(first_created)
        self.assertFalse(second_created)
        self.assertTrue(third_created)
        self.assertEqual(first["run_id"], second["run_id"])
        self.assertNotEqual(first["run_id"], third["run_id"])


class ServiceTests(SchedulerTestCase):
    def test_run_persists_success_and_provenance(self):
        service, _ = self.make_service()
        created = service.submit(
            RunRequest("pregunta válida", client_request_id="request-1"),
            user_id="evaluator",
            admin=True,
        )
        finished = wait_for_terminal(service, created["run_id"])
        self.assertEqual(finished["status"], "succeeded")
        self.assertEqual(finished["result"]["answer"]["citation_ids"], [123])
        self.assertEqual(
            [event["event_type"] for event in finished["progress"]],
            ["agent_started", "tool_completed"],
        )
        self.assertEqual(finished["provenance"]["code_revision"], "abc123")
        self.assertEqual(
            finished["events_url"], f"/runs/{created['run_id']}/events"
        )
        self.assertNotIn("status_url", finished)
        self.assertNotIn("feedback_url", finished)
        repeated = service.submit(
            RunRequest("pregunta válida", client_request_id="request-1"),
            user_id="evaluator",
            admin=True,
        )
        self.assertEqual(repeated["run_id"], created["run_id"])
        with self.assertRaises(IdempotencyConflictError):
            service.submit(
                RunRequest("otra pregunta", client_request_id="request-1"),
                user_id="evaluator",
                admin=True,
            )

    def test_scheduler_logs_the_private_cause_of_a_public_failure(self):
        class FailingExecutor(FakeExecutor):
            def execute(self, request, *, on_progress=None):
                try:
                    raise RuntimeError("private provider detail")
                except RuntimeError as exc:
                    raise PublicExecutionError(
                        "provider_unavailable", "El proveedor no está disponible."
                    ) from exc

        with mock.patch("human_eval.scheduler.LOGGER.exception") as log_exception:
            service, _ = self.make_service(FailingExecutor())
            created = service.submit(
                RunRequest("pregunta válida"), user_id="evaluator", admin=True
            )
            finished = wait_for_terminal(service, created["run_id"])

        self.assertEqual(finished["status"], "failed")
        self.assertEqual(finished["error"]["code"], "provider_unavailable")
        log_exception.assert_called_once_with(
            "human-evaluation run %s failed with %s",
            created["run_id"],
            "provider_unavailable",
        )

    def test_scheduler_prepares_executor_before_snapshotting_provenance(self):
        class PreparingExecutor(FakeExecutor):
            def __init__(self):
                self.prepared = False

            def prepare(self):
                self.prepared = True

            def provenance(self):
                provenance = dict(PROVENANCE)
                provenance["vector_used"] = self.prepared
                return provenance

        service, executor = self.make_service(PreparingExecutor())
        self.assertTrue(executor.prepared)
        created = service.submit(
            RunRequest("pregunta híbrida"), user_id="evaluator", admin=True
        )
        finished = wait_for_terminal(service, created["run_id"])
        self.assertTrue(finished["provenance"]["vector_used"])

    def test_only_one_active_run_per_evaluator(self):
        service, executor = self.make_service(BlockingExecutor())
        service.submit(
            RunRequest("primera pregunta"), user_id="evaluator", admin=True
        )
        self.assertTrue(executor.started.wait(timeout=1))
        with self.assertRaises(ActiveRunError):
            service.submit(
                RunRequest("segunda pregunta"), user_id="evaluator", admin=True
            )
        executor.release.set()

    def test_queue_full_rejection_is_logged(self):
        service, executor = self.make_service(BlockingExecutor(), queue_capacity=1)
        service.submit(
            RunRequest("primera", client_request_id="q1"),
            user_id="u1",
            admin=True,
        )
        self.assertTrue(executor.started.wait(timeout=3))
        service.submit(
            RunRequest("segunda", client_request_id="q2"),
            user_id="u2",
            admin=True,
        )
        with self.assertLogs("human_eval.service", level="WARNING") as captured:
            with self.assertRaises(QueueFullError):
                service.submit(
                    RunRequest("tercera", client_request_id="q3"),
                    user_id="u3",
                    admin=True,
                )
        self.assertIn("queue full", captured.output[0])
        self.assertIn("capacity=1", captured.output[0])
        executor.release.set()

    def test_active_run_rejection_is_logged_without_user_id(self):
        service, executor = self.make_service(BlockingExecutor(), queue_capacity=2)
        service.submit(
            RunRequest("primera", client_request_id="active-1"),
            user_id="sensitive-user-id",
            admin=True,
        )
        self.assertTrue(executor.started.wait(timeout=3))
        with self.assertLogs("human_eval.service", level="WARNING") as captured:
            with self.assertRaises(ActiveRunError):
                service.submit(
                    RunRequest("segunda", client_request_id="active-2"),
                    user_id="sensitive-user-id",
                    admin=True,
                )
        message = captured.output[0]
        self.assertIn("active run exists", message)
        self.assertIn("depth=0", message)
        self.assertIn("capacity=2", message)
        self.assertNotIn("sensitive-user-id", message)
        executor.release.set()


class AirAppTestCase(unittest.TestCase):
    daily_question_limit = 50

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "evaluation.sqlite"
        store = EvaluationStore(self.db_path)
        self.executor = FakeExecutor()
        # Web service admits and reads; a background scheduler executes.
        self.scheduler, self.scheduler_thread = start_scheduler(
            store, self.executor
        )
        self.service = EvaluationService(store)
        self.settings = WebSettings(
            host="127.0.0.1",
            port=0,
            db_path=store.path,
            session_secret="test-session-secret-that-is-at-least-32-bytes",
            daily_question_limit=self.daily_question_limit,
        )
        self.app = create_app(
            self.service,
            self.settings,
            self.executor.provenance,
            auth_backend=FakeAuthBackend(),
        )
        self.client_context = TestClient(self.app)
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        stop_scheduler(self.scheduler, self.scheduler_thread)
        self.tempdir.cleanup()

    @staticmethod
    def hidden(response, name: str) -> str:
        match = re.search(rf'name="{re.escape(name)}" value="([^"]+)"', response.text)
        if not match:
            raise AssertionError(f"missing hidden field {name}")
        return match.group(1)

    def as_user(self, user_id: str = "alice", *, admin: bool = False):
        self.client.headers["x-eval-user"] = user_id
        if admin:
            self.client.headers["x-eval-role"] = "admin"
        else:
            self.client.headers.pop("x-eval-role", None)

    def as_anonymous(self):
        self.client.headers.pop("x-eval-user", None)
        self.client.headers.pop("x-eval-role", None)

    def csrf(self) -> str:
        return self.hidden(self.client.get("/"), "csrf_token")

    def create_run(self, question: str = "pregunta desde Air") -> str:
        page = self.client.get("/")
        response = self.client.post(
            "/runs",
            data={
                "csrf_token": self.hidden(page, "csrf_token"),
                "client_request_id": self.hidden(page, "client_request_id"),
                "question": question,
                "as_of": "2026-08-17",
                "required_hops": "1",
            },
            follow_redirects=False,
        )
        self.assertEqual(response.status_code, 303, response.text[:500])
        return response.headers["location"].split("/")[-1]

    def submit_feedback(self, run_id: str, *, rating: str = "helpful"):
        page = self.client.get(f"/runs/{run_id}")
        return self.client.post(
            f"/runs/{run_id}/feedback",
            data={
                "csrf_token": self.hidden(page, "csrf_token"),
                "next": self.hidden(page, "next"),
                "rating": rating,
                "problem_types": [],
                "comment": "",
            },
            follow_redirects=False,
        )

    def review_published(self, run_id: str, *, rating: str = "helpful"):
        page = self.client.get(f"/answers/{run_id}")
        self.assertEqual(page.status_code, 200)
        return self.client.post(
            f"/runs/{run_id}/feedback",
            data={
                "csrf_token": self.hidden(page, "csrf_token"),
                "next": self.hidden(page, "next"),
                "rating": rating,
                "problem_types": [],
                "comment": "",
            },
            follow_redirects=False,
        )

    def seed_and_unlock(self, user_id: str) -> str:
        """Seed a published answer as admin and review it as ``user_id``.

        Leaves the client identified as ``user_id`` with the review gate
        satisfied for their next question.
        """
        self.as_user("root", admin=True)
        seed_id = self.create_run("respuesta semilla publicada")
        wait_for_terminal(self.service, seed_id)
        self.assertEqual(self.publish(seed_id).status_code, 303)
        self.as_user(user_id)
        self.assertEqual(self.review_published(seed_id).status_code, 303)
        return seed_id

    def publish(self, run_id: str):
        self.client.get("/")  # ensure a session csrf token exists
        return self.client.post(
            f"/admin/runs/{run_id}/publish",
            data={"csrf_token": self.csrf(), "next": "/admin/queue"},
            follow_redirects=False,
        )


class MarkdownRenderTests(unittest.TestCase):
    def test_preserves_newlines_and_paragraphs(self):
        rendered = render_markdown_html("Primera línea\nSegunda línea\n\nNuevo párrafo")
        self.assertIn("Primera línea<br>", rendered)
        self.assertEqual(rendered.count("<p>"), 2)
        self.assertIn("Nuevo párrafo", rendered)

    def test_renders_headings_and_emphasis(self):
        rendered = render_markdown_html(
            "# PODER EJECUTIVO\n\n## SECRETARIA DE SALUD\n\nTexto **negrita** y *cursiva*."
        )
        self.assertIn("<h1>PODER EJECUTIVO</h1>", rendered)
        self.assertIn("<h2>SECRETARIA DE SALUD</h2>", rendered)
        self.assertIn("<strong>negrita</strong>", rendered)
        self.assertIn("<em>cursiva</em>", rendered)

    def test_renders_lists_and_blockquotes(self):
        rendered = render_markdown_html("- primero\n- segundo\n\n> cita textual")
        self.assertIn("<li>primero</li>", rendered)
        self.assertIn("<li>segundo</li>", rendered)
        self.assertIn("<blockquote>", rendered)
        self.assertIn("cita textual", rendered)

    def test_renders_code_and_links(self):
        rendered = render_markdown_html("`código` y [dof](https://www.dof.gob.mx/nota)")
        self.assertIn("<code>código</code>", rendered)
        self.assertIn('<a href="https://www.dof.gob.mx/nota"', rendered)

    def test_escapes_and_sanitizes_unsafe_markup(self):
        rendered = render_markdown_html(
            '<script>alert("xss")</script>\n\n'
            "[enlace](javascript:alert(1))\n\n"
            '<img src="x" onerror="alert(1)">'
        )
        # Unsafe markup is rendered as inert escaped text: no live tags and
        # no dangerous hrefs remain (the literal source text stays visible).
        self.assertNotIn("<script", rendered)
        self.assertNotIn("<img", rendered)
        self.assertNotIn('href="javascript:', rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertIn("&lt;img", rendered)

    def test_preserves_inline_safe_markup(self):
        rendered = render_markdown_html("Texto con **realce** inline.")
        self.assertIn("<strong>realce</strong>", rendered)

    def test_empty_and_none_input(self):
        self.assertEqual(render_markdown_html(None), "")
        self.assertEqual(render_markdown_html(""), "")
        self.assertEqual(render_markdown_html("   \n  "), "")

    def test_plain_text_fallback_when_rendering_fails(self):
        with mock.patch(
            "human_eval.markdown_render._markdown_renderer",
            side_effect=RuntimeError("boom"),
        ):
            rendered = render_markdown_html("Texto <plano>\nsegunda línea")
        self.assertEqual(
            rendered, '<p class="chunk-text">Texto &lt;plano&gt;\nsegunda línea</p>'
        )


class ChunkHtmlTests(unittest.TestCase):
    def test_progress_chunk_renders_markdown_excerpt(self):
        rendered = _progress_chunk_html(
            {
                "chunk_id": 7,
                "document_id": 9,
                "path": "2024/acuerdo.md",
                "excerpt": "# ACUERDO\n\nPrimera línea\nSegunda línea",
            }
        )
        self.assertIn('class="markdown-body"', rendered)
        self.assertIn("<h1>ACUERDO</h1>", rendered)
        self.assertIn("Primera línea<br>", rendered)
        self.assertIn('data-chunk-id="7"', rendered)

    def test_progress_chunk_sanitizes_excerpt(self):
        rendered = _progress_chunk_html(
            {
                "chunk_id": 7,
                "document_id": 9,
                "excerpt": "<script>alert(1)</script>\n\n[x](javascript:alert(1))",
            }
        )
        self.assertNotIn("<script", rendered)
        self.assertNotIn('href="javascript:', rendered)

    def test_progress_chunk_truncation_marker(self):
        rendered = _progress_chunk_html(
            {
                "chunk_id": 7,
                "document_id": 9,
                "excerpt": "Pasaje largo.",
                "excerpt_truncated": True,
            }
        )
        self.assertIn('class="meta chunk-truncated"', rendered)
        self.assertIn("…", rendered)

    def test_progress_chunk_without_excerpt_stays_empty(self):
        rendered = _progress_chunk_html({"chunk_id": 7, "document_id": 9})
        self.assertNotIn("markdown-body", rendered)
        self.assertIn("Ruta no disponible", rendered)

    def test_attach_chunk_html_adds_sanitized_html(self):
        event = {
            "payload": {
                "chunks": [
                    {
                        "chunk_id": 1,
                        "excerpt": "# Título\n\n<script>alert(1)</script>",
                    },
                    {"chunk_id": 2, "snippet": "Texto **plano**."},
                    {"chunk_id": 3},
                    "not-a-dict",
                ]
            }
        }
        _attach_chunk_html(event)
        chunks = event["payload"]["chunks"]
        self.assertIn("<h1>Título</h1>", chunks[0]["excerpt_html"])
        self.assertNotIn("<script", chunks[0]["excerpt_html"])
        self.assertIn("<strong>plano</strong>", chunks[1]["excerpt_html"])
        self.assertNotIn("excerpt_html", chunks[2])

    def test_attach_chunk_html_preserves_existing_html(self):
        event = {"payload": {"chunks": [{"excerpt": "x", "excerpt_html": "<p>ok</p>"}]}}
        _attach_chunk_html(event)
        self.assertEqual(event["payload"]["chunks"][0]["excerpt_html"], "<p>ok</p>")

    def test_attach_chunk_html_ignores_non_dict_payload(self):
        _attach_chunk_html({"payload": None})
        _attach_chunk_html({})


class AirAppTests(AirAppTestCase):
    def test_login_create_poll_and_feedback(self):
        self.seed_and_unlock("alice")
        run_id = self.create_run()
        wait_for_terminal(self.service, run_id)
        page = self.client.get(f"/runs/{run_id}")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Respuesta: pregunta desde Air", page.text)
        self.assertIn("chunk 123", page.text)
        self.assertIn("Versión de código, índice, modelo", page.text)
        self.assertIn("Proceso de investigación", page.text)
        self.assertIn("Ver 2 pasos del proceso", page.text)
        self.assertIn("Pasaje verificable.", page.text)
        polled = self.client.get(f"/runs/{run_id}/status")
        self.assertIn(f'action="/runs/{run_id}/feedback"', polled.text)
        streamed = self.client.get(f"/runs/{run_id}/events?after=0")
        self.assertEqual(streamed.status_code, 200)
        self.assertEqual(
            streamed.headers["content-type"], "text/event-stream; charset=utf-8"
        )
        self.assertIn("event: progress", streamed.text)
        self.assertIn('"excerpt_html"', streamed.text)
        self.assertIn("event: terminal", streamed.text)
        self.assertNotIn("reasoning", streamed.text)
        resumed = self.client.get(
            f"/runs/{run_id}/events?after=0", headers={"Last-Event-ID": "1"}
        )
        self.assertNotIn("id: 1\n", resumed.text)
        self.assertIn("id: 2\n", resumed.text)
        response = self.submit_feedback(run_id, rating="partially_helpful")
        self.assertEqual(response.status_code, 303)
        feedback = self.service.store.feedback_for_run(run_id)
        self.assertEqual(feedback[0]["rating"], "partially_helpful")
        self.assertEqual(feedback[0]["user_id"], "alice")

    def test_finished_run_splits_queue_wait_from_processing(self):
        self.seed_and_unlock("alice")
        run_id = self.create_run()
        wait_for_terminal(self.service, run_id)
        page = self.client.get(f"/runs/{run_id}")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Espera en cola:", page.text)
        self.assertIn("Procesamiento:", page.text)

    def test_progress_timeline_shows_decisions_and_expandable_chunks(self):
        rendered = _progress_timeline(
            [
                {
                    "sequence": 1,
                    "event_type": "tool_completed",
                    "created_at": "2026-08-18T00:00:00Z",
                    "payload": {
                        "message": "Leyó un pasaje.",
                        "why": "Puede sostener una cita.",
                        "chunks": [
                            {
                                "chunk_id": 123,
                                "document_id": 45,
                                "path": "2026/documento.md",
                                "heading_path": ["Acuerdo", "Artículo 2"],
                                "excerpt": "Texto <verificable>.",
                            }
                        ],
                    },
                }
            ]
        )
        self.assertIn("Puede sostener una cita.", rendered)
        self.assertIn('class="chunk-link"', rendered)
        self.assertIn("Chunk 123 · documento 45", rendered)
        self.assertIn("Acuerdo › Artículo 2", rendered)
        self.assertIn('class="markdown-body"', rendered)
        self.assertIn("Texto &lt;verificable&gt;.", rendered)
        self.assertNotIn("Ver datos públicos", rendered)

    def test_anonymous_and_csrf_constraints(self):
        anonymous = self.client.get("/")
        self.assertEqual(anonymous.status_code, 200)
        self.assertIn("Entrar o crear cuenta", anonymous.text)
        response = self.client.post(
            "/runs",
            data={"question": "pregunta válida", "required_hops": "1"},
            follow_redirects=False,
        )
        self.assertEqual(response.status_code, 303)
        self.assertTrue(response.headers["location"].startswith("/login"))
        self.as_user("alice")
        rejected = self.client.post(
            "/runs",
            data={
                "csrf_token": "wrong",
                "client_request_id": "csrf-test",
                "question": "pregunta válida",
                "required_hops": "1",
            },
        )
        self.assertEqual(rejected.status_code, 403)

    def test_anonymous_sees_only_published_answers(self):
        self.seed_and_unlock("alice")
        run_id = self.create_run("pregunta aún privada")
        wait_for_terminal(self.service, run_id)
        self.as_anonymous()
        redirect = self.client.get(f"/runs/{run_id}", follow_redirects=False)
        self.assertEqual(redirect.status_code, 303)
        self.assertEqual(self.client.get(f"/answers/{run_id}").status_code, 404)
        self.assertNotIn("pregunta aún privada", self.client.get("/").text)

        self.as_user("root", admin=True)
        self.assertEqual(self.publish(run_id).status_code, 303)

        self.as_anonymous()
        home = self.client.get("/")
        self.assertIn("pregunta aún privada", home.text)
        page = self.client.get(f"/answers/{run_id}")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Respuesta: pregunta aún privada", page.text)
        self.assertNotIn("Guardar evaluación", page.text)
        stream = self.client.get(f"/runs/{run_id}/events")
        self.assertEqual(stream.status_code, 401)

    def test_users_cannot_read_each_others_private_runs(self):
        self.seed_and_unlock("alice")
        run_id = self.create_run("pregunta privada")
        self.as_user("bob")
        redirected = self.client.get(f"/runs/{run_id}", follow_redirects=False)
        self.assertEqual(redirected.status_code, 303)
        self.assertEqual(redirected.headers["location"], f"/answers/{run_id}")
        self.assertEqual(self.client.get(f"/answers/{run_id}").status_code, 404)
        stream = self.client.get(f"/runs/{run_id}/events")
        self.assertEqual(stream.status_code, 404)
        self.as_user("root", admin=True)
        self.assertEqual(self.client.get(f"/runs/{run_id}").status_code, 200)

    def test_review_gate_blocks_first_and_next_questions(self):
        self.as_user("alice")
        home = self.client.get("/")
        self.assertIn("Aún no hay respuestas publicadas", home.text)
        self.assertNotIn('name="client_request_id"', home.text)
        blocked = self.client.post(
            "/runs",
            data={
                "csrf_token": self.hidden(home, "csrf_token"),
                "client_request_id": "gate-test",
                "question": "pregunta bloqueada por la puerta",
                "required_hops": "1",
            },
        )
        self.assertEqual(blocked.status_code, 422)
        self.assertIn("Evalúa una respuesta publicada", blocked.text)

        self.as_user("root", admin=True)
        seed_id = self.create_run("respuesta semilla publicada")
        wait_for_terminal(self.service, seed_id)
        self.assertEqual(self.publish(seed_id).status_code, 303)

        self.as_user("alice")
        home = self.client.get("/")
        self.assertIn("Antes de hacer tu primera pregunta", home.text)
        self.assertEqual(self.review_published(seed_id).status_code, 303)
        self.assertIn('name="client_request_id"', self.client.get("/").text)

        run_id = self.create_run("primera pregunta de alice")
        wait_for_terminal(self.service, run_id)
        home = self.client.get("/")
        self.assertIn("Antes de hacer otra pregunta", home.text)
        # Reviewing her own answer also unlocks the next question.
        self.assertEqual(self.submit_feedback(run_id).status_code, 303)
        self.assertIn('name="client_request_id"', self.client.get("/").text)

    def test_any_user_can_evaluate_a_published_answer(self):
        self.seed_and_unlock("alice")
        run_id = self.create_run("respuesta que será pública")
        wait_for_terminal(self.service, run_id)
        self.as_user("root", admin=True)
        self.assertEqual(self.publish(run_id).status_code, 303)

        self.as_user("bob")
        page = self.client.get(f"/answers/{run_id}")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Guardar evaluación", page.text)
        response = self.client.post(
            f"/runs/{run_id}/feedback",
            data={
                "csrf_token": self.hidden(page, "csrf_token"),
                "next": self.hidden(page, "next"),
                "rating": "helpful",
                "problem_types": [],
                "comment": "Clara y con fuentes.",
            },
            follow_redirects=False,
        )
        self.assertEqual(response.status_code, 303)
        self.assertEqual(
            response.headers["location"],
            f"/answers/{run_id}?feedback=recorded",
        )
        feedback = self.service.store.feedback_for_run(run_id)
        self.assertEqual(feedback[0]["user_id"], "bob")

        # ...but not an unpublished one.
        self.as_user("root", admin=True)
        private_run = self.create_run("respuesta todavía privada")
        wait_for_terminal(self.service, private_run)
        self.as_user("bob")
        page = self.client.get("/")
        denied = self.client.post(
            f"/runs/{private_run}/feedback",
            data={
                "csrf_token": self.hidden(page, "csrf_token"),
                "next": f"/answers/{private_run}",
                "rating": "helpful",
            },
        )
        self.assertEqual(denied.status_code, 404)

    def test_moderation_queue_requires_admin_and_unpublish_hides(self):
        self.as_user("root", admin=True)
        run_id = self.create_run("respuesta para moderar")
        wait_for_terminal(self.service, run_id)

        self.as_anonymous()
        anon = self.client.get("/admin/queue", follow_redirects=False)
        self.assertEqual(anon.status_code, 303)
        self.as_user("bob")
        self.assertEqual(self.client.get("/admin/queue").status_code, 403)

        self.as_user("root", admin=True)
        queue = self.client.get("/admin/queue")
        self.assertEqual(queue.status_code, 200)
        self.assertIn("respuesta para moderar", queue.text)
        self.assertEqual(self.publish(run_id).status_code, 303)
        self.as_anonymous()
        self.assertEqual(self.client.get(f"/answers/{run_id}").status_code, 200)

        self.as_user("root", admin=True)
        self.client.get("/")
        unpublished = self.client.post(
            f"/admin/runs/{run_id}/unpublish",
            data={"csrf_token": self.csrf(), "next": "/admin/queue"},
            follow_redirects=False,
        )
        self.assertEqual(unpublished.status_code, 303)
        self.as_anonymous()
        self.assertEqual(self.client.get(f"/answers/{run_id}").status_code, 404)

    def test_admin_dashboard_and_delete_question(self):
        self.as_user("root", admin=True)
        run_id = self.create_run("pregunta que se eliminará")
        wait_for_terminal(self.service, run_id)
        self.assertEqual(self.publish(run_id).status_code, 303)

        # Anonymous visitors see the published answer but not the dashboard.
        self.as_anonymous()
        self.assertEqual(self.client.get(f"/answers/{run_id}").status_code, 200)
        anon = self.client.get("/admin", follow_redirects=False)
        self.assertEqual(anon.status_code, 303)

        # Non-admins cannot see the dashboard nor delete.
        self.as_user("bob")
        home = self.client.get("/")
        self.assertNotIn('href="/admin"', home.text)
        self.assertEqual(self.client.get("/admin").status_code, 403)
        denied = self.client.post(
            f"/admin/runs/{run_id}/delete",
            data={"csrf_token": self.csrf(), "next": "/admin"},
            follow_redirects=False,
        )
        self.assertEqual(denied.status_code, 403)
        self.assertIsNotNone(self.service.store.get_run(run_id))

        # Admins get a header link and the dashboard lists the run.
        self.as_user("root", admin=True)
        home = self.client.get("/")
        self.assertIn('<a href="/admin">Admin</a>', home.text)
        dashboard = self.client.get("/admin")
        self.assertEqual(dashboard.status_code, 200)
        self.assertIn("Panel de administración", dashboard.text)
        self.assertIn("pregunta que se eliminará", dashboard.text)
        self.assertIn("/admin/queue", dashboard.text)

        # CSRF is enforced.
        rejected = self.client.post(
            f"/admin/runs/{run_id}/delete",
            data={"csrf_token": "wrong", "next": "/admin"},
            follow_redirects=False,
        )
        self.assertEqual(rejected.status_code, 403)
        self.assertIsNotNone(self.service.store.get_run(run_id))

        # Deletion removes the run and its public answer.
        deleted = self.client.post(
            f"/admin/runs/{run_id}/delete",
            data={"csrf_token": self.csrf(), "next": "/admin"},
            follow_redirects=False,
        )
        self.assertEqual(deleted.status_code, 303)
        self.assertEqual(deleted.headers["location"], "/admin")
        self.assertIsNone(self.service.store.get_run(run_id))
        self.as_anonymous()
        self.assertEqual(self.client.get(f"/answers/{run_id}").status_code, 404)

    def test_delete_unknown_or_missing_run(self):
        self.as_user("root", admin=True)
        missing = self.client.post(
            "/admin/runs/no-existe/delete",
            data={"csrf_token": self.csrf(), "next": "/admin"},
            follow_redirects=False,
        )
        self.assertEqual(missing.status_code, 404)

    def test_delete_redirects_away_from_removed_run_page(self):
        self.as_user("root", admin=True)
        run_id = self.create_run("pregunta efímera")
        wait_for_terminal(self.service, run_id)
        response = self.client.post(
            f"/admin/runs/{run_id}/delete",
            data={"csrf_token": self.csrf(), "next": f"/runs/{run_id}"},
            follow_redirects=False,
        )
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/admin")

    def test_unknown_page_renders_styled_404(self):
        response = self.client.get("/pagina-inexistente")
        self.assertEqual(response.status_code, 404)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("Página no encontrada", response.text)
        self.assertIn("Error 404", response.text)
        self.assertIn("Agente del Diario", response.text)
        self.assertIn('href="/"', response.text)

    def test_unknown_answer_renders_styled_404(self):
        response = self.client.get("/answers/no-existe")
        self.assertEqual(response.status_code, 404)
        self.assertIn("Respuesta no encontrada", response.text)
        self.assertIn("Agente del Diario", response.text)
        self.assertIn('href="/"', response.text)

    def test_unknown_run_renders_styled_404_for_admin(self):
        # Non-owners are redirected to /answers/; only owners and admins
        # reach the run page itself.
        self.as_user("root", admin=True)
        response = self.client.get("/runs/no-existe")
        self.assertEqual(response.status_code, 404)
        self.assertIn("Ejecución no encontrada", response.text)
        self.assertIn("Agente del Diario", response.text)
        self.assertIn('<a href="/admin">Admin</a>', response.text)

    def test_unknown_api_route_returns_json_404(self):
        response = self.client.get("/api/v1/inexistente")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Not Found"})

    def test_method_not_allowed_keeps_plain_text_response(self):
        response = self.client.post("/api/v1/health")
        self.assertEqual(response.status_code, 405)
        self.assertNotIn("text/html", response.headers["content-type"])

    def test_health_and_capabilities_are_public_but_reveal_no_paths(self):
        health = self.client.get("/api/v1/health")
        self.assertEqual(health.json(), {"status": "ok"})
        capabilities = self.client.get("/api/v1/capabilities")
        self.assertEqual(capabilities.status_code, 200)
        self.assertEqual(capabilities.json()["retrieval_mode"], "lexical")
        self.assertEqual(
            capabilities.json()["limits"]["questions_per_day"],
            self.daily_question_limit,
        )
        self.assertNotIn(str(self.db_path), capabilities.text)


class LoginPageTests(AirAppTestCase):
    def test_login_page_uses_app_layout_with_dev_fallback(self):
        response = self.client.get("/login")
        self.assertEqual(response.status_code, 200)
        # The app shell (header, styles, footer), not a bare fragment.
        self.assertIn("Agente del Diario Oficial", response.text)
        self.assertIn("Las cuentas se gestionan con Clerk", response.text)
        self.assertIn("X-Eval-User", response.text)
        self.assertNotIn('id="sign-in"', response.text)
        # The modal enhancer ships on every page as progressive enhancement.
        self.assertIn("openSignIn", response.text)

    def test_login_page_mounts_provider_widget_when_configured(self):
        settings = WebSettings(
            host="127.0.0.1",
            port=0,
            db_path=self.db_path,
            session_secret="test-session-secret-that-is-at-least-32-bytes",
        )
        store = EvaluationStore(self.db_path)
        service = EvaluationService(store)
        app = create_app(
            service,
            settings,
            self.executor.provenance,
            auth_backend=FakeAuthBackend(),
            login_scripts=lambda target: f'<script data-mount="{target}"></script>',
        )
        with TestClient(app) as client:
            response = client.get("/login?next=/runs/abc")
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="sign-in"', response.text)
        self.assertIn('data-mount="/runs/abc"', response.text)

    def test_login_next_is_sanitized_against_open_redirects(self):
        self.as_user("alice")
        response = self.client.get(
            "/login?next=https://evil.example", follow_redirects=False
        )
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/")

        external_paths = (
            "///evil.example",
            "/\\evil.example",
            "/safe\nLocation: https://evil.example",
        )
        for external_path in external_paths:
            response = self.client.get(
                "/login", params={"next": external_path}, follow_redirects=False
            )
            self.assertEqual(response.status_code, 303)
            self.assertEqual(response.headers["location"], "/")

    def test_login_next_rejects_browser_normalized_external_paths(self):
        self.assertEqual(_sanitize_login_next("///evil.example"), "/")
        self.assertEqual(_sanitize_login_next("/\\evil.example"), "/")
        self.assertEqual(
            _sanitize_login_next("/safe\nLocation: https://evil.example"), "/"
        )

    def test_signed_in_user_is_redirected_to_next(self):
        self.as_user("alice")
        response = self.client.get("/login?next=/runs/abc", follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/runs/abc")


class ClerkScriptsTests(unittest.TestCase):
    def test_sync_script_resets_reload_guard_when_states_match(self):
        env = {
            "CLERK_PUBLISHABLE_KEY": "pk_test_dummy",
            "CLERK_SECRET_KEY": "sk_test_dummy",
        }
        with mock.patch.dict(os.environ, env):
            import airclerk  # noqa: F401 — validates Clerk env at import

            from human_eval.clerk_auth import clerk_page_scripts

            anonymous = clerk_page_scripts(None)
            signed_in = clerk_page_scripts(User(id="u1", role="user"))
        # The one-shot reload guard must clear once states match again,
        # otherwise only the first sign-in of a tab session ever resyncs.
        self.assertIn("sessionStorage.removeItem", anonymous)
        self.assertIn("serverHasUser = false", anonymous)
        self.assertIn("serverHasUser = true", signed_in)
        self.assertIn("pk_test_dummy", anonymous)

    def test_login_script_escapes_html_script_terminators(self):
        env = {
            "CLERK_PUBLISHABLE_KEY": "pk_test_dummy",
            "CLERK_SECRET_KEY": "sk_test_dummy",
        }
        with mock.patch.dict(os.environ, env):
            import airclerk  # noqa: F401 — validates Clerk env at import

            from human_eval.clerk_auth import clerk_login_scripts

            script = clerk_login_scripts("/</script><script>alert(1)</script>")

        self.assertNotIn('"/</script><script>', script)
        self.assertIn(r"\u003c/script\u003e", script)


class DailyQuotaTests(AirAppTestCase):
    daily_question_limit = 1

    def test_second_question_within_24h_is_rejected_but_admin_is_exempt(self):
        self.seed_and_unlock("alice")
        run_id = self.create_run("única pregunta del día")
        wait_for_terminal(self.service, run_id)
        # Reviewing her own answer clears the gate, leaving only the quota.
        self.assertEqual(self.submit_feedback(run_id).status_code, 303)

        blocked = self.client.post(
            "/runs",
            data={
                "csrf_token": self.csrf(),
                "client_request_id": "quota-test",
                "question": "otra pregunta el mismo día",
                "required_hops": "1",
            },
        )
        self.assertEqual(blocked.status_code, 422)
        self.assertIn("Ya enviaste tu pregunta", blocked.text)

        self.as_user("root", admin=True)
        admin_run = self.create_run("pregunta de administración")
        self.assertTrue(admin_run)


if __name__ == "__main__":
    unittest.main()
