"""Append-oriented SQLite persistence for runs, events, and feedback."""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from .contracts import FeedbackRequest, RunRequest, utc_now

SCHEMA_VERSION = "2"
TERMINAL_STATES = frozenset({"succeeded", "failed"})
EVENT_STATES = frozenset({"queued", "started", *TERMINAL_STATES})
PROGRESS_EVENT_TYPES = frozenset(
    {
        "agent_started",
        "model_turn_started",
        "tool_started",
        "tool_completed",
        "answer_revision_requested",
        "verification_completed",
    }
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    question TEXT NOT NULL,
    as_of TEXT,
    required_hops INTEGER NOT NULL CHECK (required_hops BETWEEN 1 AND 5),
    evaluator_hash TEXT NOT NULL,
    client_request_id TEXT,
    provenance_json TEXT NOT NULL,
    UNIQUE (evaluator_hash, client_request_id)
);
CREATE TABLE IF NOT EXISTS run_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    sequence INTEGER NOT NULL,
    event_type TEXT NOT NULL CHECK (
        event_type IN ('queued', 'started', 'succeeded', 'failed')
    ),
    created_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    UNIQUE (run_id, sequence)
);
CREATE INDEX IF NOT EXISTS run_events_run_id ON run_events(run_id, sequence);
CREATE TABLE IF NOT EXISTS run_progress (
    progress_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    sequence INTEGER NOT NULL,
    event_type TEXT NOT NULL CHECK (
        event_type IN (
            'agent_started', 'model_turn_started', 'tool_started',
            'tool_completed', 'answer_revision_requested',
            'verification_completed'
        )
    ),
    created_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    UNIQUE (run_id, sequence)
);
CREATE INDEX IF NOT EXISTS run_progress_run_id
ON run_progress(run_id, sequence);
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    created_at TEXT NOT NULL,
    evaluator_hash TEXT NOT NULL,
    rating TEXT NOT NULL CHECK (
        rating IN ('helpful', 'partially_helpful', 'not_helpful')
    ),
    problem_types_json TEXT NOT NULL,
    comment TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS feedback_run_id ON feedback(run_id, created_at);
"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class EvaluationStore:
    """Use a fresh connection per operation so HTTP and worker threads are safe."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_meta ("
                "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            current = connection.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()
            if current and current[0] not in {"1", SCHEMA_VERSION}:
                raise RuntimeError(
                    f"unsupported evaluation schema {current[0]!r}; expected {SCHEMA_VERSION}"
                )
            connection.executescript(SCHEMA)
            connection.execute(
                "INSERT OR IGNORE INTO schema_meta(key, value) VALUES (?, ?)",
                ("schema_version", SCHEMA_VERSION),
            )
            if current and current[0] == "1":
                connection.execute(
                    "UPDATE schema_meta SET value = ? WHERE key = 'schema_version'",
                    (SCHEMA_VERSION,),
                )
            connection.execute("PRAGMA optimize")

    def create_run(
        self,
        request: RunRequest,
        *,
        evaluator_hash: str,
        provenance: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        created_at = utc_now()
        run_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if request.client_request_id:
                existing = connection.execute(
                    "SELECT run_id FROM runs WHERE evaluator_hash = ? "
                    "AND client_request_id = ?",
                    (evaluator_hash, request.client_request_id),
                ).fetchone()
                if existing:
                    connection.commit()
                    found = self.get_run(existing[0])
                    assert found is not None
                    return found, False
            connection.execute(
                "INSERT INTO runs(run_id, created_at, question, as_of, required_hops, "
                "evaluator_hash, client_request_id, provenance_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    created_at,
                    request.question,
                    request.as_of,
                    request.required_hops,
                    evaluator_hash,
                    request.client_request_id,
                    _json(provenance),
                ),
            )
            connection.execute(
                "INSERT INTO run_events(run_id, sequence, event_type, created_at, payload_json) "
                "VALUES (?, 1, 'queued', ?, '{}')",
                (run_id, created_at),
            )
        found = self.get_run(run_id)
        assert found is not None
        return found, True

    def find_idempotent_run(
        self, evaluator_hash: str, client_request_id: str | None
    ) -> dict[str, Any] | None:
        if client_request_id is None:
            return None
        with self._connect() as connection:
            row = connection.execute(
                "SELECT run_id FROM runs WHERE evaluator_hash = ? "
                "AND client_request_id = ?",
                (evaluator_hash, client_request_id),
            ).fetchone()
        return self.get_run(row[0]) if row else None

    def has_active_run(self, evaluator_hash: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM runs r JOIN run_events e ON e.run_id = r.run_id "
                "WHERE r.evaluator_hash = ? AND e.sequence = "
                "(SELECT MAX(e2.sequence) FROM run_events e2 WHERE e2.run_id = r.run_id) "
                "AND e.event_type IN ('queued', 'started') LIMIT 1",
                (evaluator_hash,),
            ).fetchone()
        return row is not None

    def run_belongs_to(self, run_id: str, evaluator_hash: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM runs WHERE run_id = ? AND evaluator_hash = ?",
                (run_id, evaluator_hash),
            ).fetchone()
        return row is not None

    def runs_for_evaluator(
        self, evaluator_hash: str, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Return recent persisted runs without exposing another evaluator's data."""
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT run_id FROM runs WHERE evaluator_hash = ? "
                "ORDER BY created_at DESC, run_id DESC LIMIT ?",
                (evaluator_hash, limit),
            ).fetchall()
        runs = [self.get_run(row[0]) for row in rows]
        return [run for run in runs if run is not None]

    def check_health(self) -> bool:
        try:
            with self._connect() as connection:
                return connection.execute("SELECT 1").fetchone()[0] == 1
        except sqlite3.Error:
            return False

    def append_event(
        self, run_id: str, event_type: str, payload: dict[str, Any] | None = None
    ) -> None:
        if event_type not in EVENT_STATES or event_type == "queued":
            raise ValueError(f"invalid appended event type: {event_type}")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT event_type, sequence FROM run_events WHERE run_id = ? "
                "ORDER BY sequence DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            if current is None:
                raise KeyError(run_id)
            allowed = {
                "queued": {"started", "failed"},
                "started": TERMINAL_STATES,
                "succeeded": set(),
                "failed": set(),
            }
            if event_type not in allowed[current["event_type"]]:
                raise ValueError(
                    f"invalid run transition {current['event_type']} -> {event_type}"
                )
            connection.execute(
                "INSERT INTO run_events(run_id, sequence, event_type, created_at, payload_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    run_id,
                    int(current["sequence"]) + 1,
                    event_type,
                    utc_now(),
                    _json(payload or {}),
                ),
            )

    def append_progress(
        self, run_id: str, event_type: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        if event_type not in PROGRESS_EVENT_TYPES:
            raise ValueError(f"invalid progress event type: {event_type}")
        created_at = utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            exists = connection.execute(
                "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if exists is None:
                raise KeyError(run_id)
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM run_progress "
                "WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            sequence = int(row[0])
            connection.execute(
                "INSERT INTO run_progress(run_id, sequence, event_type, created_at, "
                "payload_json) VALUES (?, ?, ?, ?, ?)",
                (run_id, sequence, event_type, created_at, _json(payload)),
            )
        return {
            "sequence": sequence,
            "event_type": event_type,
            "created_at": created_at,
            "payload": payload,
        }

    def progress_for_run(
        self, run_id: str, *, after: int = 0, limit: int = 200
    ) -> list[dict[str, Any]]:
        if after < 0:
            raise ValueError("after must be non-negative")
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT sequence, event_type, created_at, payload_json "
                "FROM run_progress WHERE run_id = ? AND sequence > ? "
                "ORDER BY sequence LIMIT ?",
                (run_id, after, limit),
            ).fetchall()
        return [
            {
                "sequence": int(row["sequence"]),
                "event_type": row["event_type"],
                "created_at": row["created_at"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]

    def get_request(self, run_id: str) -> RunRequest | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT question, as_of, required_hops, client_request_id "
                "FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return RunRequest(row[0], row[1], int(row[2]), row[3])

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            run = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if run is None:
                return None
            events = connection.execute(
                "SELECT event_type, created_at, payload_json FROM run_events "
                "WHERE run_id = ? ORDER BY sequence",
                (run_id,),
            ).fetchall()
        latest = events[-1]
        status = (
            "running" if latest["event_type"] == "started" else latest["event_type"]
        )
        response: dict[str, Any] = {
            "run_id": run["run_id"],
            "status": status,
            "question": run["question"],
            "as_of": run["as_of"],
            "required_hops": int(run["required_hops"]),
            "created_at": run["created_at"],
            "started_at": next(
                (
                    item["created_at"]
                    for item in events
                    if item["event_type"] == "started"
                ),
                None,
            ),
            "completed_at": (
                latest["created_at"]
                if latest["event_type"] in TERMINAL_STATES
                else None
            ),
            "provenance": json.loads(run["provenance_json"]),
        }
        payload = json.loads(latest["payload_json"])
        if latest["event_type"] == "succeeded":
            response["result"] = payload
        elif latest["event_type"] == "failed":
            response["error"] = payload
        else:
            response["retry_after_ms"] = 2000
        response["progress"] = self.progress_for_run(run_id)
        return response

    def add_feedback(
        self,
        run_id: str,
        request: FeedbackRequest,
        *,
        evaluator_hash: str,
    ) -> dict[str, Any]:
        feedback_id = str(uuid.uuid4())
        created_at = utc_now()
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM runs WHERE run_id = ? AND evaluator_hash = ?",
                (run_id, evaluator_hash),
            ).fetchone()
            if not exists:
                raise KeyError(run_id)
            connection.execute(
                "INSERT INTO feedback(feedback_id, run_id, created_at, evaluator_hash, "
                "rating, problem_types_json, comment) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    feedback_id,
                    run_id,
                    created_at,
                    evaluator_hash,
                    request.rating,
                    _json(list(request.problem_types)),
                    request.comment,
                ),
            )
        return {
            "feedback_id": feedback_id,
            "run_id": run_id,
            "created_at": created_at,
        }

    def unfinished_runs(self) -> list[tuple[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT r.run_id, e.event_type FROM runs r "
                "JOIN run_events e ON e.run_id = r.run_id "
                "WHERE e.sequence = (SELECT MAX(e2.sequence) FROM run_events e2 "
                "WHERE e2.run_id = r.run_id) AND e.event_type IN ('queued', 'started')"
            ).fetchall()
        return [(row[0], row[1]) for row in rows]

    def feedback_for_run(self, run_id: str) -> list[dict[str, Any]]:
        """Administrative/test helper; feedback is not exposed by the public API."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT feedback_id, created_at, rating, problem_types_json, comment "
                "FROM feedback WHERE run_id = ? ORDER BY created_at, feedback_id",
                (run_id,),
            ).fetchall()
        return [
            {
                "feedback_id": row[0],
                "created_at": row[1],
                "rating": row[2],
                "problem_types": json.loads(row[3]),
                "comment": row[4],
            }
            for row in rows
        ]
