"""Tests for the singleton scheduler, execution lock, and split web/executor roles."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from human_eval.app import _queue_status_event, _status_fragment
from human_eval.contracts import RunRequest
from human_eval.scheduler import (
    RunScheduler,
    acquire_execution_lock,
    execution_lock_path,
)
from human_eval.service import (
    ActiveRunError,
    EvaluationService,
    IdempotencyConflictError,
    QueueFullError,
)
from human_eval.store import EvaluationStore
from tests.test_human_eval import (
    PROVENANCE,
    BlockingExecutor,
    FakeExecutor,
    start_scheduler,
    stop_scheduler,
    wait_for_terminal,
)


class ExecutionLockTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.db_path = Path(self.tempdir.name) / "evaluation.sqlite"

    def test_only_one_process_holds_the_execution_lock(self):
        fd = acquire_execution_lock(self.db_path)
        try:
            with self.assertRaises(RuntimeError) as caught:
                acquire_execution_lock(self.db_path)
            self.assertIn("execution lock", str(caught.exception))
        finally:
            os.close(fd)
        # The kernel releases the lock when the holder's fd closes.
        second = acquire_execution_lock(self.db_path)
        os.close(second)

    def test_database_symlink_shares_execution_lock(self):
        alias = self.db_path.with_name("alias.sqlite")
        alias.symlink_to(self.db_path.name)
        # Also resolve a dangling alias before SQLite creates the database.
        for exists in (False, True):
            with self.subTest(database_exists=exists):
                if exists:
                    self.db_path.touch()
                self.assertEqual(
                    execution_lock_path(alias), execution_lock_path(self.db_path)
                )
                fd = acquire_execution_lock(self.db_path)
                try:
                    with self.assertRaisesRegex(RuntimeError, "execution lock"):
                        acquire_execution_lock(alias)
                finally:
                    os.close(fd)
                second = acquire_execution_lock(alias)
                os.close(second)

    def test_lock_fd_is_not_inheritable_by_child_processes(self):
        fd = acquire_execution_lock(self.db_path)
        try:
            self.assertFalse(os.get_inheritable(fd))
        finally:
            os.close(fd)

    def test_lock_lives_next_to_the_database(self):
        self.assertEqual(
            execution_lock_path(self.db_path),
            Path(str(self.db_path) + ".lock"),
        )


class WebStartupTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.db_path = Path(self.tempdir.name) / "evaluation.sqlite"

    def test_web_startup_never_migrates(self):
        # A missing database must fail validation, not trigger a migration:
        # the scheduler (or seed, under the execution lock) is the only
        # component allowed to initialize or upgrade the schema.
        service = EvaluationService(EvaluationStore(self.db_path))
        with self.assertRaises(RuntimeError) as caught:
            service.start()
        self.assertIn("scheduler", str(caught.exception))
        self.assertFalse(self.db_path.exists())

    def test_web_startup_rejects_an_unknown_schema_version(self):
        store = EvaluationStore(self.db_path)
        store.initialize()
        with store._connect() as connection:
            connection.execute(
                "UPDATE schema_meta SET value = '99' WHERE key = 'schema_version'"
            )
        with self.assertRaises(RuntimeError):
            EvaluationService(store).start()

    def test_web_startup_accepts_a_migrated_database(self):
        EvaluationStore(self.db_path).initialize()
        service = EvaluationService(EvaluationStore(self.db_path))
        service.start()
        service.close()

    def test_web_startup_can_wait_for_scheduler_schema_preparation(self):
        store = mock.Mock()
        store.validate_schema.side_effect = [RuntimeError("not ready"), None]
        service = EvaluationService(store)
        with mock.patch("human_eval.service.time.sleep") as sleep:
            service.start(schema_wait_seconds=1)
        self.assertEqual(store.validate_schema.call_count, 2)
        sleep.assert_called_once()
        service.close()


class SchedulerRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.store = EvaluationStore(Path(self.tempdir.name) / "evaluation.sqlite")
        self.store.initialize()

    def test_startup_fails_orphaned_started_runs_before_claiming(self):
        orphaned, _ = self.store.create_run(
            RunRequest("interrumpida"), user_id="one", provenance=PROVENANCE
        )
        # Simulate a run left started by a dead scheduler process.
        self.store.append_event(orphaned["run_id"], "started")
        queued, _ = self.store.create_run(
            RunRequest("en cola"), user_id="two", provenance=PROVENANCE
        )

        scheduler = RunScheduler(self.store, FakeExecutor(), poll_seconds=0.02)
        self.assertEqual(scheduler.prepare(), 1)

        recovered = self.store.get_run(orphaned["run_id"])
        self.assertEqual(recovered["status"], "failed")
        self.assertEqual(recovered["error"]["code"], "service_restarted")
        # Recovery happens before any claiming begins.
        self.assertEqual(self.store.get_run(queued["run_id"])["status"], "queued")

        thread = threading.Thread(target=scheduler.run, daemon=True)
        thread.start()
        try:
            service = EvaluationService(self.store)
            finished = wait_for_terminal(service, queued["run_id"])
            self.assertEqual(finished["status"], "succeeded")
        finally:
            stop_scheduler(scheduler, thread)

    def test_claim_stamps_scheduler_provenance_with_started(self):
        service = EvaluationService(self.store)
        run, created = self.store.create_run(RunRequest("pregunta"), user_id="one")
        self.assertTrue(created)
        self.assertEqual(run["provenance"], {})

        claimed = self.store.claim_next_run(provenance=dict(PROVENANCE))
        self.assertEqual(claimed, run["run_id"])

        record = self.store.get_run(run["run_id"])
        self.assertEqual(record["status"], "running")
        self.assertEqual(record["provenance"], PROVENANCE)
        self.assertIsNotNone(record["started_at"])
        # The started run counts as occupied capacity for the queue UI.
        self.assertEqual(
            self.store.model_activity(1),
            {"active": 1, "capacity": 1, "available": 0},
        )
        # Web-side reads of a running run include the shared snapshot.
        service.public_run(run["run_id"], admin=True)
        with self.assertRaises(KeyError):
            # Not public and not owned by this user.
            service.public_run(run["run_id"], user_id="other")

    def test_idle_scheduler_does_not_snapshot_provenance(self):
        class CountingExecutor(FakeExecutor):
            def __init__(self):
                self.provenance_calls = 0

            def provenance(self):
                self.provenance_calls += 1
                return super().provenance()

        executor = CountingExecutor()
        scheduler, thread = start_scheduler(self.store, executor)
        try:
            time.sleep(0.08)
            self.assertEqual(executor.provenance_calls, 0)

            service = EvaluationService(self.store)
            service.start()
            run = service.submit(RunRequest("pregunta"), user_id="one", admin=True)
            self.assertEqual(
                wait_for_terminal(service, run["run_id"])["status"], "succeeded"
            )
            self.assertEqual(executor.provenance_calls, 1)
        finally:
            stop_scheduler(scheduler, thread)

    def test_stop_halts_new_claims_and_drains_in_flight(self):
        executor = BlockingExecutor()
        scheduler, thread = start_scheduler(
            self.store, executor, model_concurrency=1
        )
        service = EvaluationService(self.store)
        service.start()
        first = service.submit(RunRequest("primera"), user_id="one", admin=True)
        second = service.submit(RunRequest("segunda"), user_id="two", admin=True)
        self.assertTrue(executor.started.wait(1))

        scheduler.stop()  # what the SIGTERM/SIGINT handler calls
        # While the first run is still blocked, the second must never be
        # claimed, and the pool keeps waiting for the in-flight call (the
        # supervisor's stop timeout bounds this wait in production).
        time.sleep(0.1)
        self.assertEqual(self.store.get_run(second["run_id"])["status"], "queued")
        self.assertTrue(thread.is_alive())

        executor.release.set()
        thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        # In-flight work drained to a terminal state instead of orphaning.
        self.assertEqual(
            self.store.get_run(first["run_id"])["status"], "succeeded"
        )
        self.assertEqual(self.store.get_run(second["run_id"])["status"], "queued")

    def test_model_concurrency_never_exceeds_the_pool_limit(self):
        class OverlapTracker(FakeExecutor):
            def __init__(self):
                self.active = 0
                self.max_active = 0
                self.lock = threading.Lock()
                self.first_wave = threading.Barrier(concurrency)
                self.calls = 0

            def execute(self, request, *, on_progress=None):
                with self.lock:
                    self.active += 1
                    self.calls += 1
                    first_wave = self.calls <= concurrency
                    self.max_active = max(self.max_active, self.active)
                try:
                    if first_wave:
                        self.first_wave.wait(timeout=5)
                    return super().execute(request, on_progress=on_progress)
                finally:
                    with self.lock:
                        self.active -= 1

        for concurrency in (1, 2):
            with self.subTest(concurrency=concurrency):
                store = EvaluationStore(
                    Path(self.tempdir.name) / f"conc-{concurrency}.sqlite"
                )
                store.initialize()
                executor = OverlapTracker()
                scheduler, thread = start_scheduler(
                    store, executor, model_concurrency=concurrency
                )
                try:
                    service = EvaluationService(store)
                    service.start()
                    runs = [
                        service.submit(
                            RunRequest(f"pregunta {index}"),
                            user_id=f"user-{index}",
                            admin=True,
                        )
                        for index in range(3)
                    ]
                    for run in runs:
                        self.assertEqual(
                            wait_for_terminal(service, run["run_id"])["status"],
                            "succeeded",
                        )
                finally:
                    stop_scheduler(scheduler, thread)
                self.assertEqual(executor.max_active, concurrency)


class CrossProcessAdmissionTests(unittest.TestCase):
    """Independent web processes share one persistent admission state."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.store = EvaluationStore(Path(self.tempdir.name) / "evaluation.sqlite")
        self.store.initialize()

    def test_queue_capacity_is_shared_across_web_processes(self):
        first = EvaluationService(self.store, queue_capacity=1)
        second = EvaluationService(
            EvaluationStore(self.store.path), queue_capacity=1
        )
        first.start()
        second.start()
        first.submit(RunRequest("una"), user_id="one", admin=True)
        with self.assertRaises(QueueFullError):
            second.submit(RunRequest("dos"), user_id="two", admin=True)

    def test_active_run_limit_is_shared_across_web_processes(self):
        first = EvaluationService(self.store)
        second = EvaluationService(EvaluationStore(self.store.path))
        first.start()
        second.start()
        first.submit(RunRequest("una"), user_id="same-user", admin=True)
        with self.assertRaises(ActiveRunError):
            second.submit(RunRequest("dos"), user_id="same-user", admin=True)

    def test_idempotency_conflict_is_detected_across_web_processes(self):
        first = EvaluationService(self.store)
        second = EvaluationService(EvaluationStore(self.store.path))
        first.start()
        second.start()
        request = RunRequest("original", client_request_id="shared-key")
        created = first.submit(request, user_id="user", admin=True)
        # Let the first run finish so the active-run pre-check passes; the
        # loser's transactional payload check inside create_run must still
        # reject a reused key for a different payload even when its
        # preliminary lookup raced and missed.
        self.store.append_event(created["run_id"], "started")
        self.store.append_event(created["run_id"], "succeeded", {"answer": {}})
        with mock.patch.object(
            second.store, "find_idempotent_run", return_value=None
        ), self.assertRaises(IdempotencyConflictError):
            second.submit(
                RunRequest("distinta", client_request_id="shared-key"),
                user_id="user",
                admin=True,
            )

    def test_same_idempotency_retry_wins_over_admission_rejections(self):
        first = EvaluationService(self.store, queue_capacity=1)
        second = EvaluationService(
            EvaluationStore(self.store.path), queue_capacity=1
        )
        first.start()
        second.start()
        request = RunRequest("original", client_request_id="shared-key")
        created = first.submit(request, user_id="user", admin=True)

        # A preflight lookup could miss a concurrent winner and let the
        # active-run or queue checks reject this retry. Admission must instead
        # enter create_run's transaction, where idempotency is checked first.
        with mock.patch.object(
            second.store, "find_idempotent_run", return_value=None
        ) as preflight_lookup:
            repeated = second.submit(request, user_id="user", admin=True)

        preflight_lookup.assert_not_called()
        self.assertEqual(repeated["run_id"], created["run_id"])

    def test_same_idempotency_retry_wins_over_review_and_quota(self):
        first = EvaluationService(self.store)
        second = EvaluationService(EvaluationStore(self.store.path))
        first.start()
        second.start()
        request = RunRequest("original", client_request_id="shared-key")
        created = first.submit(request, user_id="user", admin=True)
        self.store.append_event(created["run_id"], "started")
        self.store.append_event(created["run_id"], "succeeded", {"answer": {}})

        # The successful submission consumed both the review gate and daily
        # quota. An identical retry must still return it instead of being
        # treated as a new question.
        with mock.patch.object(
            second.store, "find_idempotent_run", return_value=None
        ) as preflight_lookup:
            repeated = second.submit(request, user_id="user", admin=False)

        preflight_lookup.assert_not_called()
        self.assertEqual(repeated["run_id"], created["run_id"])


class QueuePresentationTests(unittest.TestCase):
    def test_near_zero_wait_has_consistent_html_and_stream_text(self):
        for wait in (0, 1, 59):
            with self.subTest(wait=wait):
                run = {
                    "created_at": "2026-09-05T00:00:00Z",
                    "run_id": "test",
                    "status": "queued",
                    "queue_position": 1,
                    "estimated_wait_seconds": wait,
                    "queue_snapshot": {"active": 0, "capacity": 4},
                }
                event = _queue_status_event(run)
                self.assertIn("Espera aproximada: menos de 1 min", event[1])
                html = _status_fragment(run)
                self.assertIn("Espera aproximada: menos de 1 min", html)

    def test_queue_event_requires_a_complete_snapshot(self):
        base = {
            "created_at": "2026-09-05T00:00:00Z",
            "run_id": "test",
            "status": "queued",
            "queue_position": 2,
            "estimated_wait_seconds": 480,
            "queue_snapshot": {"active": 1, "capacity": 1},
        }
        self.assertIsNotNone(_queue_status_event(base))
        incomplete = dict(base, queue_snapshot={})
        self.assertIsNone(_queue_status_event(incomplete))
        running = dict(base, status="running")
        self.assertIsNone(_queue_status_event(running))


class QueuePresentationAppTests(unittest.TestCase):
    """Queue visibility in the web UI while no scheduler is executing."""

    def setUp(self):
        import re

        from starlette.testclient import TestClient

        from human_eval.app import WebSettings, create_app
        from human_eval.auth import FakeAuthBackend

        self._re = re
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.store = EvaluationStore(Path(self.tempdir.name) / "evaluation.sqlite")
        self.store.initialize()
        self.service = EvaluationService(self.store, queue_capacity=1)
        settings = WebSettings(
            host="127.0.0.1",
            port=0,
            db_path=self.store.path,
            session_secret="test-session-secret-that-is-at-least-32-bytes",
            queue_capacity=1,
        )
        app = create_app(
            self.service,
            settings,
            lambda: dict(PROVENANCE),
            auth_backend=FakeAuthBackend(),
        )
        self.client_context = TestClient(app)
        self.client = self.client_context.__enter__()
        self.addCleanup(self.client_context.__exit__, None, None, None)

    def _hidden(self, response, name: str) -> str:
        match = self._re.search(rf'name="{name}" value="([^"]+)"', response.text)
        self.assertIsNotNone(match, f"missing hidden field {name}")
        return match.group(1)

    def _ask(self, question: str, user: str = "alice"):
        self.client.headers["x-eval-user"] = user
        # Admins bypass the review gate, so the question form is rendered.
        self.client.headers["x-eval-role"] = "admin"
        page = self.client.get("/")
        return self.client.post(
            "/runs",
            data={
                "csrf_token": self._hidden(page, "csrf_token"),
                "client_request_id": self._hidden(page, "client_request_id"),
                "question": question,
                "as_of": "",
                "required_hops": "1",
            },
            follow_redirects=False,
        )

    def test_full_queue_returns_retry_after(self):
        response = self._ask("primera pregunta", user="alice")
        self.assertEqual(response.status_code, 303)
        response = self._ask("segunda pregunta", user="bob")
        self.assertEqual(response.status_code, 503)
        self.assertIn("cola de preguntas", response.text)
        retry_after = int(response.headers["retry-after"])
        self.assertGreaterEqual(retry_after, 60)

    def test_status_page_shows_queue_position_and_capacity(self):
        response = self._ask("pregunta en cola")
        run_id = response.headers["location"].split("/")[-1]
        page = self.client.get(f"/runs/{run_id}")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Posición en la cola: 1", page.text)
        self.assertIn("0 de 1 slots ocupados", page.text)


class SchedulerFailureTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.store = EvaluationStore(Path(self.tempdir.name) / "evaluation.sqlite")
        self.store.initialize()

    def test_persistence_failure_stops_scheduler_for_supervised_restart(self):
        _, thread = start_scheduler(self.store, FakeExecutor())
        service = EvaluationService(self.store)
        service.start()
        # A terminal write that cannot persist strands the run in 'started';
        # only startup recovery repairs that, so the scheduler must stop and
        # let the supervisor (systemd Restart=always) restart it.
        with mock.patch.object(
            self.store,
            "append_event",
            side_effect=sqlite3.OperationalError("disk I/O error"),
        ):
            run = service.submit(RunRequest("pregunta"), user_id="one", admin=True)
            thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.store.get_run(run["run_id"])["status"], "running")

        replacement, replacement_thread = start_scheduler(self.store, FakeExecutor())
        try:
            recovered = self.store.get_run(run["run_id"])
            self.assertEqual(recovered["status"], "failed")
            self.assertEqual(recovered["error"]["code"], "service_restarted")
        finally:
            stop_scheduler(replacement, replacement_thread)

    def test_stop_requested_during_prepare_is_honored(self):
        scheduler = RunScheduler(self.store, FakeExecutor(), poll_seconds=0.02)
        scheduler.prepare()
        service = EvaluationService(self.store)
        service.start()
        run = service.submit(RunRequest("pregunta"), user_id="one", admin=True)
        # SIGTERM can arrive while prepare() (embedding-server startup) is
        # still running; run() must not erase that stop request.
        scheduler.stop()
        scheduler.run()
        self.assertEqual(self.store.get_run(run["run_id"])["status"], "queued")

    def test_stop_during_provenance_prevents_the_claim(self):
        class SlowProvenanceExecutor(FakeExecutor):
            def __init__(self):
                self.inside = threading.Event()
                self.release = threading.Event()

            def provenance(self):
                self.inside.set()
                if not self.release.wait(3):
                    raise RuntimeError("test provenance barrier timed out")
                return super().provenance()

        executor = SlowProvenanceExecutor()
        scheduler, thread = start_scheduler(self.store, executor)
        service = EvaluationService(self.store)
        service.start()
        run = service.submit(RunRequest("pregunta"), user_id="one", admin=True)
        self.assertTrue(executor.inside.wait(1))
        # SIGTERM arrives while provenance runs its git and index probes.
        scheduler.stop()
        executor.release.set()
        thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.store.get_run(run["run_id"])["status"], "queued")
