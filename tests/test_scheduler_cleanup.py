"""Regression tests for scheduler cleanup, duration sampling, and seed paths."""

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from human_eval.contracts import RunRequest
from human_eval.scheduler import RunScheduler
from human_eval.store import EvaluationStore
from scripts import seed_human_eval_v4_hybrid as seed


class SchedulerCleanupTests(unittest.TestCase):
    def test_signal_stop_does_not_arm_fatal_watchdog(self):
        scheduler = RunScheduler(mock.Mock(), mock.Mock())
        scheduler.stop()
        with mock.patch("human_eval.scheduler.threading.Timer") as timer:
            scheduler.run()
        timer.assert_not_called()
        scheduler.executor.close.assert_called_once()

    def test_fatal_errors_force_process_exit_with_hung_work(self):
        for failure in ("poll", "persistence"):
            with self.subTest(failure=failure):
                program = textwrap.dedent('''
                    import threading
                    from unittest import mock
                    from human_eval.scheduler import RunScheduler

                    scheduler = RunScheduler(
                        mock.Mock(), mock.Mock(), fatal_shutdown_seconds=0.1
                    )
                    def poll():
                        scheduler._pool.submit(threading.Event().wait)
                        if FAILURE == "poll":
                            raise RuntimeError("poll failed")
                        from concurrent.futures import Future
                        failed = Future()
                        failed.set_exception(RuntimeError("persistence failed"))
                        scheduler._in_flight.add(failed)
                        return RunScheduler.poll_once(scheduler)
                    scheduler.poll_once = poll
                    scheduler.run()
                ''').replace("FAILURE", repr(failure))
                result = subprocess.run(
                    [sys.executable, "-c", program],
                    capture_output=True, text=True, timeout=10, check=False,
                )
                self.assertEqual(result.returncode, 1, result.stderr)

    def test_poll_errors_drain_work_before_closing_executor(self):
        for source in ("queue_depth", "provenance", "claim_next_run"):
            with self.subTest(source=source):
                store = mock.Mock()
                store.queue_depth.return_value = 1
                executor = mock.Mock()
                target = executor if source == "provenance" else store
                getattr(target, source).side_effect = RuntimeError(source)
                scheduler = RunScheduler(store, executor)
                completed = []
                original_poll = scheduler.poll_once

                def poll(scheduler=scheduler, completed=completed,
                         original_poll=original_poll):
                    scheduler._pool.submit(lambda completed=completed: completed.append(True))
                    return original_poll()

                closed_after = []
                executor.close.side_effect = lambda completed=completed, closed_after=closed_after: closed_after.extend(completed)
                with (mock.patch.object(scheduler, "poll_once", side_effect=poll),
                      self.assertRaisesRegex(RuntimeError, source)):
                    scheduler.run()
                executor.close.assert_called_once()
                self.assertEqual(closed_after, [True])
                self.assertIsNone(scheduler._pool)


class WebWorkerStartupTests(unittest.TestCase):
    def test_multiple_workers_use_factory_and_propagate_root(self):
        from human_eval import app

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch("sys.argv", ["app", "--workers", "2", "--repo-root", directory]),
                mock.patch.object(app, "uvicorn") as uvicorn,
                mock.patch.object(app, "WebSettings") as settings,
                mock.patch.object(app, "build_default_app") as build,
            ):
                settings.from_env.return_value = mock.Mock(host="127.0.0.1", port=8000)
                self.assertEqual(app.main(), 0)
                settings.from_env.assert_called_once_with(root)
                self.assertEqual(os.environ["DOF_APP_REPO_ROOT"], str(root))
                uvicorn.run.assert_called_once_with(
                    "human_eval.app:create_uvicorn_app", factory=True,
                    workers=2, host="127.0.0.1", port=8000, access_log=False,
                )
                build.assert_not_called()
                build.return_value = (mock.sentinel.app, mock.sentinel.settings)
                self.assertIs(app.create_uvicorn_app(), mock.sentinel.app)
                build.assert_called_once_with(root)


class DurationSamplingTests(unittest.TestCase):
    def test_recovery_is_excluded_before_sample_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EvaluationStore(Path(directory) / "eval.sqlite")
            store.initialize()
            for index, terminal in enumerate(("succeeded", "failed", "recovery")):
                run, _ = store.create_run(RunRequest("question"), user_id=str(index))
                run_id = run["run_id"]
                store.append_event(run_id, "started")
                if terminal == "recovery":
                    store.fail_interrupted_runs()
                else:
                    store.append_event(run_id, terminal, {})
                with store._connect() as connection:
                    connection.execute(
                        "UPDATE run_events SET created_at = ? WHERE run_id = ? "
                        "AND event_type = 'started'",
                        ("2026-01-01T00:00:00+00:00", run_id),
                    )
                    connection.execute(
                        "UPDATE run_events SET created_at = ? WHERE run_id = ? "
                        "AND event_type IN ('succeeded', 'failed')",
                        (f"2026-01-0{index + 1}T00:00:10+00:00", run_id),
                    )
            self.assertEqual(store.recent_durations(limit=2), [86410.0, 10.0])


class SeedDatabaseTests(unittest.TestCase):
    def test_provenance_failure_does_not_persist_seed_work(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EvaluationStore(Path(directory) / "eval.sqlite")
            store.initialize()
            executor = mock.Mock()
            executor.provenance.side_effect = RuntimeError("probe failed")
            item = {"id": "probe-test", "question": "question"}
            # Repeated attempts must retry the probe, not return 'pending'.
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "probe failed"):
                    seed.seed_live_run(store, executor, item, publish=True)
                self.assertIsNone(
                    store.find_idempotent_run(
                        seed.SEED_USER, "eval-v4-hybrid:probe-test"
                    )
                )
                self.assertEqual(store.queue_depth(), 0)
            executor.execute.assert_not_called()

    def test_database_precedence_matches_lock_and_store(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for cli, env, expected in (
                (None, None, root / "var/human_evaluation.sqlite"),
                (None, str(root / "env.sqlite"), root / "env.sqlite"),
                (str(root / "cli.sqlite"), str(root / "env.sqlite"), root / "cli.sqlite"),
            ):
                with self.subTest(cli=cli, env=env):
                    argv = ["seed", "--repo-root", str(root)]
                    if cli:
                        argv.extend(["--db", cli])
                    environment = {"DOF_HUMAN_EVAL_DB": env} if env else {}
                    with (
                        mock.patch.dict(os.environ, environment, clear=True),
                        mock.patch("sys.argv", argv),
                        mock.patch.object(seed, "load_queries", return_value=[]),
                        mock.patch.object(seed, "EvaluationStore") as store,
                        mock.patch.object(seed, "acquire_execution_lock", return_value=123) as lock,
                        mock.patch.object(seed, "AgentExecutorConfig"),
                        mock.patch.object(seed, "AgentRunExecutor"),
                        mock.patch.object(seed.os, "close"),
                        mock.patch("builtins.print"),
                    ):
                        self.assertEqual(seed.main(), 0)
                    store.assert_called_once_with(expected)
                    lock.assert_called_once_with(expected)
