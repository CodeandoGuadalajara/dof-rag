"""Singleton scheduler process: the only component that executes runs.

Architecture invariants:

- Web workers only create ``queued`` runs and serve UI/API requests.
- Exactly one scheduler holds the OS-level execution lock (``flock``); the
  kernel releases it if the process dies, and a second scheduler or seed
  process fails immediately instead of violating the singleton.
- The scheduler owns ``AgentRunExecutor`` and the embedding server.
- Claiming stamps scheduler provenance and ``started`` in one transaction.
- An in-process bounded pool enforces ``DOF_MODEL_CONCURRENCY``.
- Scheduler startup fails every previously ``started`` run: only the
  scheduler starts runs, so a started run at startup is provably orphaned.

Shutdown contract: SIGTERM/SIGINT stop new claims and let in-flight work
finish; the supervisor (systemd ``TimeoutStopSec`` + ``KillMode``) bounds the
wait and kills the whole control group, including embedding-server children.
Runs still ``started`` after a hard kill are recovered at the next startup.
"""

from __future__ import annotations

import argparse
import fcntl
import logging
import os
import signal
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol

from .contracts import RunRequest
from .service import ProgressCallback, PublicExecutionError
from .store import EvaluationStore

LOGGER = logging.getLogger(__name__)

DEFAULT_POLL_SECONDS = 0.5


class RunExecutor(Protocol):
    def execute(
        self,
        request: RunRequest,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> dict[str, Any]: ...

    def provenance(self) -> dict[str, Any]: ...


def execution_lock_path(db_path: str | Path) -> Path:
    """Place the lock next to the canonical database, including symlink aliases.

    Hard-link aliases and retargeting symlinks while running are unsupported.
    """
    return Path(str(Path(db_path).resolve()) + ".lock")


def acquire_execution_lock(db_path: str | Path) -> int:
    """Take the exclusive scheduler/seed lock; return the open fd.

    The fd is explicitly non-inheritable so executor child processes (the
    embedding server) cannot retain the lock after this process dies.
    """
    path = execution_lock_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.set_inheritable(fd, False)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        raise RuntimeError(
            f"another scheduler or seed process holds the execution lock "
            f"({path}); stop it before starting a new one"
        ) from exc
    return fd


def execute_claimed_run(
    store: EvaluationStore, executor: RunExecutor, run_id: str
) -> None:
    """Execute one scheduler-claimed run and persist its terminal state."""
    request = store.get_request(run_id)
    if request is None:
        LOGGER.warning("claimed run %s vanished before execution", run_id)
        return
    try:
        result = executor.execute(
            request,
            on_progress=lambda event_type, payload: store.append_progress(
                run_id, event_type, payload
            ),
        )
    except PublicExecutionError as exc:
        if exc.__cause__ is not None:
            LOGGER.exception(
                "human-evaluation run %s failed with %s", run_id, exc.code
            )
        payload = {"code": exc.code, "message": str(exc)}
        if exc.used_tokens is not None:
            payload["usage"] = {"total_tokens": exc.used_tokens}
        store.append_event(run_id, "failed", payload)
    except Exception:
        LOGGER.exception("human-evaluation run %s failed", run_id)
        store.append_event(
            run_id,
            "failed",
            {
                "code": "internal_error",
                "message": "La ejecución no pudo completarse.",
            },
        )
    else:
        store.append_event(run_id, "succeeded", result)


class RunScheduler:
    """Poll the persistent queue and execute runs on a bounded local pool."""

    def __init__(
        self,
        store: EvaluationStore,
        executor: RunExecutor,
        *,
        model_concurrency: int = 1,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        fatal_shutdown_seconds: float = 30.0,
    ):
        if model_concurrency < 1:
            raise ValueError("model_concurrency must be positive")
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        if fatal_shutdown_seconds <= 0:
            raise ValueError("fatal_shutdown_seconds must be positive")
        self.fatal_shutdown_seconds = fatal_shutdown_seconds
        self.store = store
        self.executor = executor
        self.model_concurrency = model_concurrency
        self.poll_seconds = poll_seconds
        self._stopping = threading.Event()
        self._fatal: BaseException | None = None
        self._in_flight: set[Any] = set()
        self._pool: ThreadPoolExecutor | None = None

    def prepare(self) -> int:
        """Initialize the schema and recover orphaned runs. Lock holder only."""
        self.store.initialize()
        recovered = self.store.fail_interrupted_runs()
        if recovered:
            LOGGER.warning("recovered %s interrupted executions", recovered)
        prepare = getattr(self.executor, "prepare", None)
        if callable(prepare):
            prepare()
        return recovered

    def poll_once(self) -> int:
        """Claim and dispatch runs up to the pool's free capacity."""
        assert self._pool is not None, "poll_once requires a running pool"
        done = {future for future in self._in_flight if future.done()}
        for future in done:
            # execute_claimed_run handles executor failures; anything here is
            # a store error after the run ended, leaving it recoverable as
            # 'started' at the next scheduler startup.
            if future.exception() is not None:
                LOGGER.error(
                    "run execution raised unexpectedly",
                    exc_info=(
                        type(future.exception()),
                        future.exception(),
                        future.exception().__traceback__,
                    ),
                )
                if self._fatal is None:
                    self._fatal = future.exception()
        self._in_flight -= done
        if self._fatal is not None:
            # A persistence failure stranded its run in 'started' and only
            # startup recovery can repair that, so stop claiming and let the
            # supervisor restart this process (systemd Restart=always).
            self._stopping.set()
            return 0
        if self._stopping.is_set():
            return 0
        # Poll SQLite first so an idle scheduler does not run the much more
        # expensive git and index provenance probes every poll interval. New
        # arrivals after this snapshot wait at most one interval.
        available = min(
            self.model_concurrency - len(self._in_flight),
            self.store.queue_depth(),
        )
        claimed = 0
        for _ in range(available):
            if self._stopping.is_set():
                break
            provenance = self.executor.provenance()
            if self._stopping.is_set():
                # provenance() runs git and index probes; a stop requested
                # during that work must still prevent the claim.
                break
            run_id = self.store.claim_next_run(provenance=provenance)
            if run_id is None:
                break
            claimed += 1
            self._in_flight.add(
                self._pool.submit(execute_claimed_run, self.store, self.executor, run_id)
            )
        return claimed

    def run(self) -> None:
        """Supervise the queue until ``stop`` or a fatal persistence error.

        Single-use: an early stop request (for example SIGTERM during a slow
        ``prepare``) is honored instead of cleared.
        """
        watchdog: threading.Timer | None = None

        def force_exit() -> None:
            # ThreadPoolExecutor threads also block normal interpreter exit.
            # Skip Python cleanup so systemd sees failure and kills remaining
            # children in the unit's cgroup before restarting. Direct launches
            # need an external supervisor for child cleanup on this hard path.
            os._exit(1)

        try:
            with ThreadPoolExecutor(
                max_workers=self.model_concurrency,
                thread_name_prefix="dof-human-eval-scheduler",
            ) as self._pool:
                try:
                    while not self._stopping.is_set():
                        if self.poll_once() == 0:
                            self._stopping.wait(self.poll_seconds)
                except BaseException as exc:
                    self._fatal = exc
                    raise
                finally:
                    if self._fatal is not None:
                        # Arm before the pool context starts waiting. Unlike a
                        # signal-driven stop, an internal failure does not start
                        # systemd's TimeoutStopSec countdown.
                        watchdog = threading.Timer(
                            self.fatal_shutdown_seconds, force_exit
                        )
                        watchdog.daemon = True
                        watchdog.start()
        finally:
            self._pool = None
            try:
                close = getattr(self.executor, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        LOGGER.exception("executor shutdown hook failed")
            finally:
                if watchdog is not None:
                    watchdog.cancel()
        if self._fatal is not None:
            raise RuntimeError(
                "scheduler stopped after an execution persistence failure"
            ) from self._fatal

    def stop(self) -> None:
        self._stopping.set()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parent.parent
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="evaluation database (default: DOF_HUMAN_EVAL_DB or var/human_evaluation.sqlite)",
    )
    args = parser.parse_args()

    from .agent_executor import AgentExecutorConfig, AgentRunExecutor

    root = args.repo_root.resolve()
    db_path = args.db or Path(
        os.environ.get("DOF_HUMAN_EVAL_DB", root / "var/human_evaluation.sqlite")
    )
    config = AgentExecutorConfig.from_env(root)

    lock_fd = acquire_execution_lock(db_path)
    store = EvaluationStore(db_path)
    executor = AgentRunExecutor(config)
    scheduler = RunScheduler(
        store,
        executor,
        model_concurrency=config.model_concurrency,
    )

    def handle_signal(signum: int, _frame: Any) -> None:
        LOGGER.info("received signal %s; stopping claims and draining", signum)
        scheduler.stop()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        scheduler.prepare()
        scheduler.run()
    finally:
        os.close(lock_fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
