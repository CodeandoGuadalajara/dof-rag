"""Append-oriented SQLite persistence for runs, events, and feedback."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from .contracts import FeedbackRequest, RunRequest, utc_now
from . import token_budget, chat_store

SCHEMA_VERSION = "5"
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

class ActiveRunConflict(RuntimeError):
    """The user already has a queued or running execution."""


class QueueCapacityConflict(RuntimeError):
    """The shared persistent queue has reached its configured capacity."""


class DailyQuotaConflict(RuntimeError):
    """The user has reached the configured rolling question limit."""


class ReviewRequiredConflict(RuntimeError):
    """The user must review an answer before submitting another question."""


class IdempotencyPayloadConflict(RuntimeError):
    """An idempotency key was reused for a different request payload."""


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

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open one connection for an operation and always close it."""
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            with connection:
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA foreign_keys = ON")
                connection.execute("PRAGMA busy_timeout = 30000")
                yield connection
        finally:
            connection.close()

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
            if current and current[0] not in {"1", "2", "3", "4", SCHEMA_VERSION}:
                raise RuntimeError(
                    f"unsupported evaluation schema {current[0]!r}; expected {SCHEMA_VERSION}"
                )
            connection.executescript(SCHEMA)
            self._migrate_columns(connection)
            token_budget.initialize(connection)
            chat_store.initialize(connection)
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

    def validate_schema(self) -> None:
        """Verify the database is initialized at the current schema version.

        Web processes never migrate: the scheduler (or the seed script, under
        the same execution lock) is the only component allowed to initialize
        or upgrade the database, so startup ordering never races a migration.
        """
        if not self.path.exists():
            raise RuntimeError(
                f"evaluation database {self.path} does not exist; "
                "start the scheduler once to initialize it"
            )
        with self._connect() as connection:
            try:
                row = connection.execute(
                    "SELECT value FROM schema_meta WHERE key = 'schema_version'"
                ).fetchone()
            except sqlite3.Error as exc:
                raise RuntimeError(
                    f"evaluation database {self.path} is not initialized"
                ) from exc
        if row is None or row[0] != SCHEMA_VERSION:
            raise RuntimeError(
                f"unsupported evaluation schema "
                f"{None if row is None else row[0]!r}; expected {SCHEMA_VERSION} "
                "(the scheduler migrates the database on startup)"
            )

    def create_run(
        self,
        request: RunRequest,
        *,
        user_id: str,
        provenance: dict[str, Any] | None = None,
        enforce_active_run: bool = False,
        require_review: bool = False,
        queue_capacity: int | None = None,
        daily_question_limit: int | None = None,
        daily_since: str | None = None,
        reserved_tokens: int | None = None,
        daily_token_limit: int = token_budget.DAILY_TOKEN_LIMIT,
    ) -> tuple[dict[str, Any], bool]:
        if daily_question_limit is not None and daily_question_limit < 1:
            raise ValueError("daily_question_limit must be positive")
        if daily_question_limit is not None and daily_since is None:
            raise ValueError("daily_since is required with daily_question_limit")
        if queue_capacity is not None and queue_capacity < 1:
            raise ValueError("queue_capacity must be positive")
        created_at = utc_now()
        run_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if request.client_request_id:
                existing = connection.execute(
                    "SELECT run_id, question, as_of, required_hops FROM runs "
                    "WHERE user_id = ? "
                    "AND client_request_id = ?",
                    (user_id, request.client_request_id),
                ).fetchone()
                if existing:
                    if any(
                        (
                            existing["question"] != request.question,
                            existing["as_of"] != request.as_of,
                            existing["required_hops"] != request.required_hops,
                        )
                    ):
                        raise IdempotencyPayloadConflict(request.client_request_id)
                    connection.commit()
                    found = self.get_run(existing["run_id"])
                    assert found is not None
                    return found, False
            if enforce_active_run:
                active = connection.execute(
                    "SELECT 1 FROM runs r JOIN run_events e ON e.run_id = r.run_id "
                    "WHERE r.user_id = ? AND e.sequence = "
                    "(SELECT MAX(e2.sequence) FROM run_events e2 "
                    "WHERE e2.run_id = r.run_id) "
                    "AND e.event_type IN ('queued', 'started') LIMIT 1",
                    (user_id,),
                ).fetchone()
                if active:
                    raise ActiveRunConflict(user_id)
            if require_review:
                reviewed = connection.execute(
                    "SELECT 1 FROM feedback f WHERE f.user_id = ? "
                    "AND f.created_at > COALESCE((SELECT MAX(r.created_at) "
                    "FROM runs r WHERE r.user_id = ? AND r.run_id NOT IN (SELECT run_id FROM chat_messages)), '') LIMIT 1",
                    (user_id, user_id),
                ).fetchone()
                if reviewed is None:
                    raise ReviewRequiredConflict(user_id)
            if daily_question_limit is not None:
                submissions = connection.execute(
                    "SELECT COUNT(*) FROM runs WHERE user_id = ? AND created_at >= ? "
                    "AND run_id NOT IN (SELECT run_id FROM chat_messages)",
                    (user_id, daily_since),
                ).fetchone()[0]
                if int(submissions) >= daily_question_limit:
                    raise DailyQuotaConflict(daily_question_limit)
            if queue_capacity is not None:
                queued = connection.execute(
                    "SELECT COUNT(*) FROM runs r JOIN run_events e ON e.run_id = r.run_id "
                    "WHERE e.sequence = (SELECT MAX(e2.sequence) FROM run_events e2 "
                    "WHERE e2.run_id = r.run_id) AND e.event_type = 'queued'"
                ).fetchone()[0]
                if int(queued) >= queue_capacity:
                    raise QueueCapacityConflict(queue_capacity)
            if reserved_tokens is not None:
                token_budget.reserve(
                    connection, run_id, user_id, reserved_tokens, limit=daily_token_limit
                )
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
                    _json(provenance or {}),
                ),
            )
            if reserved_tokens is not None:
                chat_store.add_message(connection, run_id, user_id, created_at, request.question)
            connection.execute(
                "INSERT INTO run_events(run_id, sequence, event_type, created_at, payload_json) "
                "VALUES (?, 1, 'queued', ?, '{}')",
                (run_id, created_at),
            )
        found = self.get_run(run_id)
        assert found is not None
        return found, True

    def chat_runs(self, user_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT m.run_id FROM chat_messages m JOIN chat_conversations c "
                "ON c.conversation_id = m.conversation_id "
                "WHERE c.user_id = ? AND m.role = 'user' "
                "ORDER BY m.created_at DESC LIMIT ?", (user_id, limit)
            ).fetchall()
        runs = []
        for row in reversed(rows):
            run = self.get_run(row[0])
            if run is not None:
                with self._connect() as connection:
                    messages = connection.execute(
                        "SELECT role, content, result_json FROM chat_messages WHERE run_id = ?", (row[0],)
                    ).fetchall()
                for message in messages:
                    if message['role'] == 'user':
                        run['question'] = message['content']
                    elif message['result_json'] is not None:
                        run['result' if run['status'] == 'succeeded' else 'error'] = json.loads(message['result_json'])
                runs.append(run)
        return runs

    def is_chat_run(self, run_id: str) -> bool:
        with self._connect() as connection:
            return connection.execute("SELECT 1 FROM chat_messages WHERE run_id = ?", (run_id,)).fetchone() is not None

    def token_balance(self, user_id: str, *, limit: int = token_budget.DAILY_TOKEN_LIMIT) -> int:
        with self._connect() as connection:
            return token_budget.balance(connection, user_id, limit=limit)

    def settle_tokens(self, run_id: str, used: int) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            token_budget.settle(connection, run_id, used)

    def queue_depth(self) -> int:
        """Runs currently waiting for the scheduler."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM runs r JOIN run_events e ON e.run_id = r.run_id "
                "WHERE e.sequence = (SELECT MAX(e2.sequence) FROM run_events e2 "
                "WHERE e2.run_id = r.run_id) AND e.event_type = 'queued'"
            ).fetchone()
        return int(row[0])

    def model_activity(self, capacity: int) -> dict[str, int]:
        """Active executions (latest event 'started') against the capacity.

        Only the scheduler starts runs, so the count of started runs is the
        exact in-flight number; ``capacity`` is the scheduler's configured
        model concurrency, mirrored in the web process for display only.
        """
        if capacity < 1:
            raise ValueError("capacity must be positive")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM runs r JOIN run_events e ON e.run_id = r.run_id "
                "WHERE e.sequence = (SELECT MAX(e2.sequence) FROM run_events e2 "
                "WHERE e2.run_id = r.run_id) AND e.event_type = 'started'"
            ).fetchone()
        active = int(row[0])
        return {
            "active": active,
            "capacity": capacity,
            "available": max(0, capacity - active),
        }

    def claim_next_run(self, *, provenance: dict[str, Any]) -> str | None:
        """Atomically start the oldest queued run, stamping its provenance.

        The singleton scheduler is the only caller, so no fencing is needed;
        the transaction still keeps the claim atomic against web readers.
        """
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            run = connection.execute(
                "SELECT r.run_id FROM runs r JOIN run_events e ON e.run_id = r.run_id "
                "WHERE e.sequence = (SELECT MAX(e2.sequence) FROM run_events e2 "
                "WHERE e2.run_id = r.run_id) AND e.event_type = 'queued' "
                "ORDER BY r.created_at, r.run_id LIMIT 1"
            ).fetchone()
            if run is None:
                return None
            run_id = str(run[0])
            self._start_run_in_connection(connection, run_id, provenance)
        return run_id

    def start_run(self, run_id: str, *, provenance: dict[str, Any]) -> bool:
        """Start one specific queued run (used by the seed script)."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            latest = connection.execute(
                "SELECT event_type FROM run_events WHERE run_id = ? "
                "ORDER BY sequence DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            if latest is None:
                raise KeyError(run_id)
            if latest[0] != "queued":
                return False
            self._start_run_in_connection(connection, run_id, provenance)
        return True

    @staticmethod
    def _start_run_in_connection(
        connection: sqlite3.Connection, run_id: str, provenance: dict[str, Any]
    ) -> None:
        """Stamp provenance and append 'started' in the caller's transaction."""
        connection.execute(
            "UPDATE runs SET provenance_json = ? WHERE run_id = ?",
            (_json(provenance), run_id),
        )
        current = connection.execute(
            "SELECT sequence FROM run_events WHERE run_id = ? "
            "ORDER BY sequence DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        if current is None:
            raise KeyError(run_id)
        connection.execute(
            "INSERT INTO run_events(run_id, sequence, event_type, created_at, payload_json) "
            "VALUES (?, ?, 'started', ?, '{}')",
            (run_id, int(current[0]) + 1, utc_now()),
        )

    def fail_interrupted_runs(self) -> int:
        """Fail every run left 'started' by a previous scheduler lifetime.

        Only the scheduler executes runs, so at scheduler startup (under the
        execution lock) any started run is provably orphaned.
        """
        recovered = 0
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            runs = connection.execute(
                "SELECT r.run_id, e.sequence FROM runs r JOIN run_events e "
                "ON e.run_id = r.run_id WHERE e.sequence = (SELECT MAX(e2.sequence) "
                "FROM run_events e2 WHERE e2.run_id = r.run_id) "
                "AND e.event_type = 'started'"
            ).fetchall()
            now = utc_now()
            for run in runs:
                chat_store.finish_message(connection, str(run[0]), 'failed', {
                    'code': 'service_restarted', 'message': 'La ejecución se interrumpió antes de terminar.',
                })
                # Unknown usage after a crash: conservatively charge the reservation.
                connection.execute(
                    "UPDATE token_reservations SET used = reserved "
                    "WHERE run_id = ? AND used IS NULL", (str(run[0]),)
                )
                connection.execute(
                    "INSERT INTO run_events(run_id, sequence, event_type, created_at, payload_json) "
                    "VALUES (?, ?, 'failed', ?, ?)",
                    (
                        str(run[0]),
                        int(run[1]) + 1,
                        now,
                        _json(
                            {
                                "code": "service_restarted",
                                "message": "La ejecución se interrumpió antes de terminar.",
                            }
                        ),
                    ),
                )
                recovered += 1
        return recovered

    def queue_position(self, run_id: str) -> int | None:
        """1-based FIFO position among queued runs; None when not queued."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT r.created_at FROM runs r WHERE r.run_id = ? AND "
                "(SELECT e.event_type FROM run_events e WHERE e.run_id = r.run_id "
                "ORDER BY e.sequence DESC LIMIT 1) = 'queued'",
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            ahead = connection.execute(
                "SELECT COUNT(*) FROM runs r WHERE "
                "(SELECT e.event_type FROM run_events e WHERE e.run_id = r.run_id "
                "ORDER BY e.sequence DESC LIMIT 1) = 'queued' AND "
                "(r.created_at < ? OR (r.created_at = ? AND r.run_id < ?))",
                (row["created_at"], row["created_at"], run_id),
            ).fetchone()
        return int(ahead[0]) + 1

    def recent_durations(self, *, limit: int = 10) -> list[float]:
        """Inference seconds (started -> terminal) of recent finished runs."""
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT started.created_at AS started_at, "
                "finished.created_at AS finished_at "
                "FROM run_events started "
                "JOIN run_events finished ON finished.run_id = started.run_id "
                "AND finished.sequence = started.sequence + 1 "
                "WHERE started.event_type = 'started' "
                "AND finished.event_type IN ('succeeded', 'failed') "
                "AND COALESCE(json_extract(finished.payload_json, '$.code'), '') "
                "!= 'service_restarted' "
                "ORDER BY finished.created_at DESC, finished.run_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        durations: list[float] = []
        for row in rows:
            try:
                began = datetime.fromisoformat(row["started_at"].replace("Z", "+00:00"))
                ended = datetime.fromisoformat(
                    row["finished_at"].replace("Z", "+00:00")
                )
            except (TypeError, ValueError):
                continue
            durations.append(max(0.0, (ended - began).total_seconds()))
        return durations

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
                "AND run_id NOT IN (SELECT run_id FROM chat_messages) "
                "ORDER BY created_at DESC, run_id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        runs = [self.get_run(row[0]) for row in rows]
        return [run for run in runs if run is not None]

    def count_submissions_since(self, user_id: str, since: str) -> int:
        """Submissions in the window, regardless of outcome (quota basis)."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM runs WHERE user_id = ? AND created_at >= ? "
                "AND run_id NOT IN (SELECT run_id FROM chat_messages)",
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
                "FROM runs r WHERE r.user_id = ? AND r.run_id NOT IN (SELECT run_id FROM chat_messages)), '') LIMIT 1",
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
                "AND r.run_id NOT IN (SELECT run_id FROM chat_messages) "
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
                "AND r.run_id NOT IN (SELECT run_id FROM chat_messages) "
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
                "AND r.run_id NOT IN (SELECT run_id FROM chat_messages) "
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
            if connection.execute("SELECT 1 FROM chat_messages WHERE run_id = ?", (run_id,)).fetchone():
                raise ValueError("chat messages cannot be published as evaluation answers")
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

    def admin_runs(self, *, limit: int = 200) -> list[dict[str, Any]]:
        """All runs for the admin dashboard, newest first, with latest status."""
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT r.run_id, r.question, r.created_at, r.published_at, "
                "e.event_type, r.user_id "
                "FROM runs r JOIN run_events e ON e.run_id = r.run_id "
                "AND e.sequence = (SELECT MAX(e2.sequence) FROM run_events e2 "
                "WHERE e2.run_id = r.run_id) "
                "WHERE r.run_id NOT IN (SELECT run_id FROM chat_messages) "
                "ORDER BY r.created_at DESC, r.run_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "run_id": row[0],
                "question": row[1],
                "created_at": row[2],
                "published_at": row[3],
                "status": "running" if row[4] == "started" else row[4],
                "user_id": row[5],
            }
            for row in rows
        ]

    def delete_run(self, run_id: str) -> None:
        """Delete a terminal run with its events, progress, and feedback.

        Active runs (queued/running) are refused: the worker may still be
        writing events for them, and the foreign keys to ``runs`` would make
        those writes fail.
        """
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            exists = connection.execute(
                "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if exists is None:
                raise KeyError(run_id)
            latest = connection.execute(
                "SELECT event_type FROM run_events WHERE run_id = ? "
                "ORDER BY sequence DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            if latest is not None and latest[0] in ("queued", "started"):
                raise ValueError("active runs cannot be deleted")
            for table in ("run_progress", "run_events", "feedback", "runs"):
                connection.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))

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
            if event_type in TERMINAL_STATES:
                reservation = connection.execute(
                    "SELECT reserved FROM token_reservations WHERE run_id = ?", (run_id,)
                ).fetchone()
                if reservation:
                    actual = (payload or {}).get("usage", {}).get("total_tokens")
                    charged = actual if isinstance(actual, int) and not isinstance(actual, bool) and actual >= 0 else reservation[0]
                    token_budget.settle(connection, run_id, charged)
                    if event_type == 'failed':
                        payload = dict(payload or {})
                        payload['partial'] = chat_store.saved_partial(connection, run_id)
                    chat_store.finish_message(connection, run_id, event_type, payload or {})
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
                "SELECT question, as_of, required_hops, client_request_id, user_id "
                "FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        with self._connect() as connection:
            reservation = connection.execute(
                "SELECT reserved FROM token_reservations WHERE run_id = ?", (run_id,)
            ).fetchone()
        history = []
        if reservation:
            for previous in self.chat_runs(row[4]):
                if previous["run_id"] == run_id:
                    break
                if previous["status"] == "succeeded":
                    history.extend([
                        {"role": "user", "content": previous["question"]},
                        {"role": "assistant", "content": previous["result"]["answer"]["text"]},
                    ])
        # ponytail: six prior exchanges; add conversation selection when needed.
        return RunRequest(row[0], row[1], int(row[2]), row[3],
                          reservation[0] if reservation else None, tuple(history[-12:]))

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

    def delete_seed_runs(self, *, user_prefix: str = "seed:") -> int:
        """Delete seed-owned runs together with events, progress, and feedback.

        The store is append-only for real users; ``seed:`` users are
        re-importable system fixtures, so reseeding may remove their runs.
        Any other prefix is refused. Returns the number of runs deleted.
        """
        if not user_prefix.startswith("seed:"):
            raise ValueError("only seed: system users can be deleted")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT run_id FROM runs WHERE substr(user_id, 1, ?) = ?",
                (len(user_prefix), user_prefix),
            ).fetchall()
            run_ids = [row[0] for row in rows]
            if run_ids:
                placeholders = ",".join("?" for _ in run_ids)
                for table in ("run_progress", "run_events", "feedback", "runs"):
                    connection.execute(
                        f"DELETE FROM {table} WHERE run_id IN ({placeholders})",
                        run_ids,
                    )
        return len(run_ids)

    def delete_seed_run(self, run_id: str, *, user_prefix: str = "seed:") -> bool:
        """Delete one seed-owned run, returning whether it existed."""
        if not user_prefix.startswith("seed:"):
            raise ValueError("only seed: system users can be deleted")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            exists = connection.execute(
                "SELECT 1 FROM runs WHERE run_id = ? AND substr(user_id, 1, ?) = ?",
                (run_id, len(user_prefix), user_prefix),
            ).fetchone()
            if exists is None:
                return False
            for table in ("run_progress", "run_events", "feedback", "runs"):
                connection.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
        return True

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
