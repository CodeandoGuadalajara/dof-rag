from __future__ import annotations

import re
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path

from starlette.testclient import TestClient

from human_eval.agent_executor import _public_result
from human_eval.app import WebSettings, create_app
from human_eval.contracts import ContractError, FeedbackRequest, RunRequest
from human_eval.service import (
    ActiveRunError,
    EvaluationService,
    IdempotencyConflictError,
)
from human_eval.store import EvaluationStore

PROVENANCE = {
    "code_revision": "abc123",
    "code_dirty": False,
    "corpus_version": "corpus-v1",
    "chunker_version": "chunks-v1",
    "vector_available": False,
    "vector_index_version": None,
    "provider": "fake",
    "model": "fake-model",
    "configuration": {
        "retrieval_mode": "lexical",
        "max_model_turns": 8,
        "max_tool_calls": 8,
    },
}


class FakeExecutor:
    def execute(self, request: RunRequest):
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

    def execute(self, request: RunRequest):
        self.started.set()
        if not self.release.wait(timeout=3):
            raise RuntimeError("test timed out")
        return super().execute(request)


def wait_for_terminal(service: EvaluationService, run_id: str) -> dict:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        run = service.public_run(run_id)
        if run["status"] in {"succeeded", "failed"}:
            return run
        time.sleep(0.01)
    raise AssertionError("run did not reach a terminal state")


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
            request, evaluator_hash="evaluator", provenance=PROVENANCE
        )
        self.assertTrue(created)
        self.store.append_event(run["run_id"], "started")
        self.store.append_event(run["run_id"], "succeeded", {"answer": {}})
        feedback = self.store.add_feedback(
            run["run_id"],
            FeedbackRequest("not_helpful", ("incorrect_answer",), "No coincide."),
            evaluator_hash="evaluator",
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

    def test_client_request_id_is_idempotent_per_evaluator(self):
        request = RunRequest("pregunta válida", client_request_id="same-request")
        first, first_created = self.store.create_run(
            request, evaluator_hash="one", provenance=PROVENANCE
        )
        second, second_created = self.store.create_run(
            request, evaluator_hash="one", provenance=PROVENANCE
        )
        third, third_created = self.store.create_run(
            request, evaluator_hash="two", provenance=PROVENANCE
        )
        self.assertTrue(first_created)
        self.assertFalse(second_created)
        self.assertTrue(third_created)
        self.assertEqual(first["run_id"], second["run_id"])
        self.assertNotEqual(first["run_id"], third["run_id"])


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = EvaluationStore(Path(self.tempdir.name) / "evaluation.sqlite")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_worker_persists_success_and_provenance(self):
        executor = FakeExecutor()
        service = EvaluationService(self.store, executor, executor.provenance)
        service.start()
        try:
            created = service.submit(
                RunRequest("pregunta válida", client_request_id="request-1"),
                evaluator_hash="evaluator",
            )
            finished = wait_for_terminal(service, created["run_id"])
            self.assertEqual(finished["status"], "succeeded")
            self.assertEqual(finished["result"]["answer"]["citation_ids"], [123])
            self.assertEqual(finished["provenance"]["code_revision"], "abc123")
            repeated = service.submit(
                RunRequest("pregunta válida", client_request_id="request-1"),
                evaluator_hash="evaluator",
            )
            self.assertEqual(repeated["run_id"], created["run_id"])
            with self.assertRaises(IdempotencyConflictError):
                service.submit(
                    RunRequest("otra pregunta", client_request_id="request-1"),
                    evaluator_hash="evaluator",
                )
        finally:
            service.close()

    def test_only_one_active_run_per_evaluator(self):
        executor = BlockingExecutor()
        service = EvaluationService(self.store, executor, executor.provenance)
        service.start()
        try:
            service.submit(RunRequest("primera pregunta"), evaluator_hash="evaluator")
            self.assertTrue(executor.started.wait(timeout=1))
            with self.assertRaises(ActiveRunError):
                service.submit(
                    RunRequest("segunda pregunta"), evaluator_hash="evaluator"
                )
            executor.release.set()
            service.queue.join()
        finally:
            executor.release.set()
            service.close()


class AirAppTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "evaluation.sqlite"
        store = EvaluationStore(self.db_path)
        self.executor = FakeExecutor()
        self.service = EvaluationService(store, self.executor, self.executor.provenance)
        settings = WebSettings(
            host="127.0.0.1",
            port=0,
            db_path=store.path,
            evaluator_tokens=("secret-token", "other-token"),
            session_secret="test-session-secret-that-is-at-least-32-bytes",
        )
        self.app = create_app(self.service, settings, self.executor.provenance)
        self.client_context = TestClient(self.app)
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.tempdir.cleanup()

    @staticmethod
    def hidden(response, name: str) -> str:
        match = re.search(rf'name="{re.escape(name)}" value="([^"]+)"', response.text)
        if not match:
            raise AssertionError(f"missing hidden field {name}")
        return match.group(1)

    def login(self, token: str = "secret-token"):
        page = self.client.get("/login")
        csrf = self.hidden(page, "csrf_token")
        return self.client.post(
            "/login",
            data={"csrf_token": csrf, "token": token},
            follow_redirects=False,
        )

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
        self.assertEqual(response.status_code, 303)
        return response.headers["location"].split("/")[-1]

    def test_login_create_poll_and_feedback(self):
        self.assertEqual(self.login().status_code, 303)
        run_id = self.create_run()
        wait_for_terminal(self.service, run_id)
        page = self.client.get(f"/runs/{run_id}")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Respuesta: pregunta desde Air", page.text)
        self.assertIn("chunk 123", page.text)
        self.assertIn("Versión de código, índice, modelo", page.text)
        polled = self.client.get(f"/runs/{run_id}/status")
        self.assertIn(f'action="/runs/{run_id}/feedback"', polled.text)
        response = self.client.post(
            f"/runs/{run_id}/feedback",
            data={
                "csrf_token": self.hidden(page, "csrf_token"),
                "rating": "partially_helpful",
                "problem_types": ["missing_evidence", "incomplete_coverage"],
                "comment": "Falta una fuente.",
            },
            follow_redirects=False,
        )
        self.assertEqual(response.status_code, 303)
        feedback = self.service.store.feedback_for_run(run_id)
        self.assertEqual(feedback[0]["rating"], "partially_helpful")
        self.assertEqual(
            feedback[0]["problem_types"],
            ["missing_evidence", "incomplete_coverage"],
        )

    def test_authentication_session_and_csrf_are_enforced(self):
        anonymous = self.client.get("/", follow_redirects=False)
        self.assertEqual(anonymous.status_code, 303)
        self.assertEqual(anonymous.headers["location"], "/login")
        self.assertEqual(self.login("wrong-token").status_code, 401)
        self.assertEqual(self.login().status_code, 303)
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

    def test_invitation_token_is_not_persisted_or_echoed_in_session(self):
        response = self.login()
        self.assertNotIn("secret-token", response.headers.get("set-cookie", ""))
        run_id = self.create_run("pregunta privada persistida")
        wait_for_terminal(self.service, run_id)
        self.assertNotIn(b"secret-token", self.db_path.read_bytes())

    def test_evaluators_cannot_read_each_others_runs(self):
        self.assertEqual(self.login().status_code, 303)
        run_id = self.create_run("pregunta privada")
        self.client.cookies.clear()
        self.assertEqual(self.login("other-token").status_code, 303)
        response = self.client.get(f"/runs/{run_id}")
        self.assertEqual(response.status_code, 404)

    def test_health_and_capabilities_are_public_but_reveal_no_paths(self):
        health = self.client.get("/api/v1/health")
        self.assertEqual(health.json(), {"status": "ok"})
        capabilities = self.client.get("/api/v1/capabilities")
        self.assertEqual(capabilities.status_code, 200)
        self.assertEqual(capabilities.json()["retrieval_mode"], "lexical")
        self.assertNotIn(str(self.db_path), capabilities.text)


if __name__ == "__main__":
    unittest.main()
