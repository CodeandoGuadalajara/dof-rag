"""Queue and lifecycle management independent of the HTTP transport."""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from typing import Any, Protocol

from .contracts import FeedbackRequest, RunRequest
from .store import EvaluationStore

LOGGER = logging.getLogger(__name__)


class RunExecutor(Protocol):
    def execute(self, request: RunRequest) -> dict[str, Any]: ...


class PublicExecutionError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class QueueFullError(RuntimeError):
    pass


class ActiveRunError(RuntimeError):
    pass


class IdempotencyConflictError(RuntimeError):
    pass


class EvaluationService:
    def __init__(
        self,
        store: EvaluationStore,
        executor: RunExecutor,
        provenance_factory: Callable[[], dict[str, Any]],
        *,
        queue_capacity: int = 20,
    ):
        self.store = store
        self.executor = executor
        self.provenance_factory = provenance_factory
        self.queue: queue.Queue[str | None] = queue.Queue(maxsize=queue_capacity)
        self.worker = threading.Thread(
            target=self._worker_loop, name="dof-human-eval-worker", daemon=True
        )
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self.store.initialize()
        for run_id, state in self.store.unfinished_runs():
            if state == "started":
                self.store.append_event(
                    run_id,
                    "failed",
                    {
                        "code": "service_restarted",
                        "message": "La ejecución se interrumpió al reiniciar el servicio.",
                    },
                )
            else:
                try:
                    self.queue.put_nowait(run_id)
                except queue.Full:
                    self.store.append_event(
                        run_id,
                        "failed",
                        {"code": "queue_full", "message": "La cola local está llena."},
                    )
        self.worker.start()
        self._started = True

    def close(self) -> None:
        if not self._started:
            return
        self.queue.put(None)
        self.worker.join(timeout=5)
        self._started = False

    def submit(self, request: RunRequest, *, evaluator_hash: str) -> dict[str, Any]:
        if not self._started:
            raise RuntimeError("service has not started")
        existing = self.idempotent_run(request, evaluator_hash=evaluator_hash)
        if existing is not None:
            return existing
        if self.store.has_active_run(evaluator_hash):
            raise ActiveRunError("evaluator already has an active run")
        if self.queue.full():
            raise QueueFullError("execution queue is full")
        run, created = self.store.create_run(
            request,
            evaluator_hash=evaluator_hash,
            provenance=self.provenance_factory(),
        )
        if created:
            try:
                self.queue.put_nowait(run["run_id"])
            except queue.Full:
                self.store.append_event(
                    run["run_id"],
                    "failed",
                    {"code": "queue_full", "message": "La cola local está llena."},
                )
                raise QueueFullError("execution queue is full")
        return self.public_run(run["run_id"], evaluator_hash=evaluator_hash)

    def idempotent_run(
        self, request: RunRequest, *, evaluator_hash: str
    ) -> dict[str, Any] | None:
        existing = self.store.find_idempotent_run(
            evaluator_hash, request.client_request_id
        )
        if existing is None:
            return None
        if any(
            (
                existing["question"] != request.question,
                existing["as_of"] != request.as_of,
                existing["required_hops"] != request.required_hops,
            )
        ):
            raise IdempotencyConflictError(
                "client_request_id was already used for a different request"
            )
        return self.public_run(existing["run_id"], evaluator_hash=evaluator_hash)

    def public_run(
        self, run_id: str, *, evaluator_hash: str | None = None
    ) -> dict[str, Any]:
        if evaluator_hash is not None and not self.store.run_belongs_to(
            run_id, evaluator_hash
        ):
            raise KeyError(run_id)
        run = self.store.get_run(run_id)
        if run is None:
            raise KeyError(run_id)
        run["status_url"] = f"/api/v1/runs/{run_id}"
        run["feedback_url"] = f"/api/v1/runs/{run_id}/feedback"
        return run

    def submit_feedback(
        self,
        run_id: str,
        request: FeedbackRequest,
        *,
        evaluator_hash: str,
    ) -> dict[str, Any]:
        return self.store.add_feedback(run_id, request, evaluator_hash=evaluator_hash)

    def _worker_loop(self) -> None:
        while True:
            run_id = self.queue.get()
            try:
                if run_id is None:
                    return
                request = self.store.get_request(run_id)
                if request is None:
                    continue
                self.store.append_event(run_id, "started")
                try:
                    result = self.executor.execute(request)
                except PublicExecutionError as exc:
                    self.store.append_event(
                        run_id,
                        "failed",
                        {"code": exc.code, "message": str(exc)},
                    )
                except Exception:
                    LOGGER.exception("human-evaluation run %s failed", run_id)
                    self.store.append_event(
                        run_id,
                        "failed",
                        {
                            "code": "internal_error",
                            "message": "La ejecución no pudo completarse.",
                        },
                    )
                else:
                    self.store.append_event(run_id, "succeeded", result)
            finally:
                self.queue.task_done()
