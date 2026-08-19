"""Append-oriented SQLite persistence for runs, events, and feedback."""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from .contracts import FeedbackRequest, RunRequest, utc_now

SCHEMA_VERSION = "3"
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
    user_id TEXT NOT NULL,
    client_request_id TEXT,
    provenance_json TEXT NOT NULL,
    published_at TEXT,
    published_by TEXT,
    UNIQUE (user_id, client_request_id)
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
    user_id TEXT NOT NULL,
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
            if current and current[0] not in {"1", "2", SCHEMA_VERSION}:
                raise RuntimeError(
                    f"unsupported evaluation schema {current[0]!r}; expected {SCHEMA_VERSION}"
                )
            connection.executescript(SCHEMA)
            self._migrate_columns(connection)
            connection.execute(
                "INSERT OR IGNORE INTO schema_meta(key, value) VALUES (?, ?)",
                ("schema_version", SCHEMA_VERSION),
            )
            if current and current[0] != SCHEMA_VERSION:
                connection.execute(
                    "UPDATE schema_meta SET value = ? WHERE key = 'schema_version'",
                    (SCHEMA_VERSION,),
                )
            connection.execute("PRAGMA optimize")

    @staticmethod
    def _migrate_columns(connection: sqlite3.Connection) -> None:
        """Bring pre-v3 tables to the v3 shape (idempotent).

        v3 renamed evaluator_hash to user_id (Clerk user ids replace
        invitation-token hashes; legacy hash values are kept but can no
        longer log in) and added the publication columns.
        """
        run_columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)")}
        if "evaluator_hash" in run_columns:
            connection.execute(
                "ALTER TABLE runs RENAME COLUMN evaluator_hash TO user_id"
            )
        if "published_at" not in run_columns:
            connection.execute("ALTER TABLE runs ADD COLUMN published_at TEXT")
        if "published_by" not in run_columns:
            connection.execute("ALTER TABLE runs ADD COLUMN published_by TEXT")
        feedback_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(feedback)")
        }
        if "evaluator_hash" in feedback_columns:
            connection.execute(
                "ALTER TABLE feedback RENAME COLUMN evaluator_hash TO user_id"
            )

    def create_run(
        self,
        request: RunRequest,
        *,
        user_id: str,
        provenance: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        created_at = utc_now()
        run_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if request.client_request_id:
                existing = connection.execute(
                    "SELECT run_id FROM runs WHERE user_id = ? "
                    "AND client_request_id = ?",
                    (user_id, request.client_request_id),
                ).fetchone()
                if existing:
                    connection.commit()
                    found = self.get_run(existing[0])
                    assert found is not None
                    return found, False
            connection.execute(
                "INSERT INTO runs(run_id, created_at, question, as_of, required_hops, "
                "user_id, client_request_id, provenance_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    created_at,
                    request.question,
                    request.as_of,
                    request.required_hops,
                    user_id,
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
        self, user_id: str, client_request_id: str | None
    ) -> dict[str, Any] | None:
        if client_request_id is None:
            return None
        with self._connect() as connection:
            row = connection.execute(
                "SELECT run_id FROM runs WHERE user_id = ? AND client_request_id = ?",
                (user_id, client_request_id),
            ).fetchone()
        return self.get_run(row[0]) if row else None

    def has_active_run(self, user_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM runs r JOIN run_events e ON e.run_id = r.run_id "
                "WHERE r.user_id = ? AND e.sequence = "
                "(SELECT MAX(e2.sequence) FROM run_events e2 WHERE e2.run_id = r.run_id) "
                "AND e.event_type IN ('queued', 'started') LIMIT 1",
                (user_id,),
            ).fetchone()
        return row is not None

    def run_belongs_to(self, run_id: str, user_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM runs WHERE run_id = ? AND user_id = ?",
                (run_id, user_id),
            ).fetchone()
        return row is not None

    def runs_for_user(self, user_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent persisted runs without exposing another user's data."""
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT run_id FROM runs WHERE user_id = ? "
                "ORDER BY created_at DESC, run_id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        runs = [self.get_run(row[0]) for row in rows]
        return [run for run in runs if run is not None]

    def count_submissions_since(self, user_id: str, since: str) -> int:
        """Submissions in the window, regardless of outcome (quota basis)."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM runs WHERE user_id = ? AND created_at >= ?",
                (user_id, since),
            ).fetchone()
        return int(row[0])

    def has_review_since_last_submission(self, user_id: str) -> bool:
        """Whether the user evaluated any answer after their latest question.

        Users with no questions yet must have evaluated at least one answer
        (the gate also applies to the first question).
        """
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM feedback f WHERE f.user_id = ? "
                "AND f.created_at > COALESCE((SELECT MAX(r.created_at) "
                "FROM runs r WHERE r.user_id = ?), '') LIMIT 1",
                (user_id, user_id),
            ).fetchone()
        return row is not None

    def next_answer_to_review(self, user_id: str) -> dict[str, Any] | None:
        """An answer the user can review next: any published one, or their own
        succeeded runs (which only they can evaluate), least-reviewed first."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT r.run_id, r.question, r.published_at FROM runs r "
                "JOIN run_events e ON e.run_id = r.run_id AND e.sequence = "
                "(SELECT MAX(e2.sequence) FROM run_events e2 WHERE e2.run_id = r.run_id) "
                "WHERE e.event_type = 'succeeded' "
                "AND (r.published_at IS NOT NULL OR r.user_id = ?) "
                "AND NOT EXISTS (SELECT 1 FROM feedback f "
                "WHERE f.run_id = r.run_id AND f.user_id = ?) "
                "ORDER BY (r.published_at IS NULL) ASC, "
                "(SELECT COUNT(*) FROM feedback f2 WHERE f2.run_id = r.run_id) ASC, "
                "r.created_at ASC LIMIT 1",
                (user_id, user_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "run_id": row[0],
            "question": row[1],
            "published": row[2] is not None,
        }

    def has_feedback(self, run_id: str, user_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM feedback WHERE run_id = ? AND user_id = ? LIMIT 1",
                (run_id, user_id),
            ).fetchone()
        return row is not None

    def published_runs(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """Summaries of published, succeeded runs for the public listing."""
        if not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT r.run_id, r.question, r.created_at, r.published_at "
                "FROM runs r JOIN run_events e ON e.run_id = r.run_id "
                "AND e.sequence = (SELECT MAX(e2.sequence) FROM run_events e2 "
                "WHERE e2.run_id = r.run_id) "
                "WHERE r.published_at IS NOT NULL AND e.event_type = 'succeeded' "
                "ORDER BY r.published_at DESC, r.run_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "run_id": row[0],
                "question": row[1],
                "created_at": row[2],
                "published_at": row[3],
            }
            for row in rows
        ]

    def runs_for_moderation(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """Succeeded runs for the admin queue: unpublished first."""
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT r.run_id, r.question, r.created_at, r.published_at "
                "FROM runs r JOIN run_events e ON e.run_id = r.run_id "
                "AND e.sequence = (SELECT MAX(e2.sequence) FROM run_events e2 "
                "WHERE e2.run_id = r.run_id) "
                "WHERE e.event_type = 'succeeded' "
                "ORDER BY (r.published_at IS NULL) DESC, r.created_at DESC, "
                "r.run_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "run_id": row[0],
                "question": row[1],
                "created_at": row[2],
                "published_at": row[3],
            }
            for row in rows
        ]

    def publish_run(self, run_id: str, *, publisher_id: str) -> None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT e.event_type FROM run_events e WHERE e.run_id = ? "
                "ORDER BY e.sequence DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            if row is None:
                raise KeyError(run_id)
            if row[0] != "succeeded":
                raise ValueError("only succeeded runs can be published")
            connection.execute(
                "UPDATE runs SET published_at = ?, published_by = ? WHERE run_id = ?",
                (utc_now(), publisher_id, run_id),
            )

    def unpublish_run(self, run_id: str) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE runs SET published_at = NULL, published_by = NULL "
                "WHERE run_id = ?",
                (run_id,),
            )
            if cursor.rowcount == 0:
                raise KeyError(run_id)

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
            "published_at": run["published_at"],
            "published_by": run["published_by"],
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
        user_id: str,
    ) -> dict[str, Any]:
        feedback_id = str(uuid.uuid4())
        created_at = utc_now()
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if not exists:
                raise KeyError(run_id)
            connection.execute(
                "INSERT INTO feedback(feedback_id, run_id, created_at, user_id, "
                "rating, problem_types_json, comment) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    feedback_id,
                    run_id,
                    created_at,
                    user_id,
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
        """Administrative/test helper listing who evaluated the run and how."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT feedback_id, created_at, user_id, rating, problem_types_json, "
                "comment FROM feedback WHERE run_id = ? "
                "ORDER BY created_at, feedback_id",
                (run_id,),
            ).fetchall()
        return [
            {
                "feedback_id": row[0],
                "created_at": row[1],
                "user_id": row[2],
                "rating": row[3],
                "problem_types": json.loads(row[4]),
                "comment": row[5],
            }
            for row in rows
        ]
