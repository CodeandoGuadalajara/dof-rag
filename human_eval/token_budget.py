"""Transactional token reservations; shared by web workers and the scheduler."""

import sqlite3
from datetime import datetime, timedelta, timezone

# Matches the recipe's 262,144-token context capacity numerically;
# this is cumulative usage over 24h, not a model context-window setting.
DAILY_TOKEN_LIMIT = 262_144


class TokenBudgetExceeded(ValueError):
    pass


def initialize(connection: sqlite3.Connection) -> None:
    connection.execute("""CREATE TABLE IF NOT EXISTS token_reservations (
        run_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        reserved INTEGER NOT NULL CHECK (reserved > 0),
        used INTEGER CHECK (used >= 0 AND used <= reserved)
    )""")


def balance(connection: sqlite3.Connection, user_id: str, *, limit: int = DAILY_TOKEN_LIMIT) -> int:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    charged = connection.execute(
        "SELECT COALESCE(SUM(COALESCE(used, reserved)), 0) "
        "FROM token_reservations WHERE user_id = ? "
        "AND (used IS NULL OR created_at >= ?)",
        (user_id, cutoff),
    ).fetchone()[0]
    return max(0, limit - charged)


def reserve(connection: sqlite3.Connection, run_id: str, user_id: str, tokens: int,
            *, limit: int = DAILY_TOKEN_LIMIT) -> None:
    """Call inside the same BEGIN IMMEDIATE transaction that admits the run."""
    if not connection.in_transaction:
        raise RuntimeError("reservation requires an admission transaction")
    if tokens < 1 or limit < 1:
        raise ValueError("token amounts must be positive")
    existing = connection.execute(
        "SELECT user_id, reserved FROM token_reservations WHERE run_id = ?", (run_id,)
    ).fetchone()
    if existing:
        if tuple(existing) != (user_id, tokens):
            raise ValueError("reservation payload conflict")
        return
    if tokens > balance(connection, user_id, limit=limit):
        raise TokenBudgetExceeded("Insufficient tokens in the rolling 24-hour budget")
    connection.execute(
        "INSERT INTO token_reservations VALUES (?, ?, ?, ?, NULL)",
        (run_id, user_id, datetime.now(timezone.utc).isoformat(), tokens),
    )


def settle(connection: sqlite3.Connection, run_id: str, used: int) -> None:
    """Release unused tokens only when actual usage is known.

    Failed/interrupted runs without reliable usage keep their reservation.
    The executor must enforce the reserved ceiling before calling this function.
    """
    if used < 0:
        raise ValueError("usage must be non-negative")
    row = connection.execute(
        "SELECT reserved, used FROM token_reservations WHERE run_id = ?", (run_id,)
    ).fetchone()
    if row is None:
        raise KeyError(run_id)
    if used > row[0]:
        raise ValueError("executor exceeded its token reservation")
    if row[1] is not None and row[1] != used:
        raise ValueError("usage was already settled")
    connection.execute("UPDATE token_reservations SET used = ? WHERE run_id = ?", (used, run_id))
