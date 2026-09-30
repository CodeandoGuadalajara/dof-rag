"""Run with python -m unittest tests.test_token_budget."""
import sqlite3
import tempfile
import unittest
from pathlib import Path

from human_eval.token_budget import (
    DAILY_TOKEN_LIMIT,
    TokenBudgetExceeded,
    balance,
    initialize,
    reserve,
    settle,
)


class TokenBudgetTests(unittest.TestCase):
    def test_admission_reservation_is_atomic_and_retry_is_free(self):
        from human_eval.contracts import RunRequest
        from human_eval.store import EvaluationStore
        with tempfile.TemporaryDirectory() as directory:
            store = EvaluationStore(Path(directory) / "runs.sqlite")
            store.initialize()
            request = RunRequest("A question", client_request_id="retry")
            run, created = store.create_run(request, user_id="alice", reserved_tokens=DAILY_TOKEN_LIMIT - 20_000)
            self.assertTrue(created)
            same, created = store.create_run(request, user_id="alice", reserved_tokens=DAILY_TOKEN_LIMIT - 20_000)
            self.assertFalse(created)
            self.assertEqual(run["run_id"], same["run_id"])
            with self.assertRaises(TokenBudgetExceeded):
                store.create_run(RunRequest("Another question"), user_id="alice", reserved_tokens=30_000)
            self.assertEqual(len(store.chat_runs("alice")), 1)
            self.assertEqual(store.runs_for_user("alice"), [])
            self.assertEqual(store.token_balance("alice"), 20_000)
            store.settle_tokens(run["run_id"], 10_000)
            self.assertEqual(store.token_balance("alice"), DAILY_TOKEN_LIMIT - 10_000)

    def test_persisted_ceiling_and_terminal_settlement(self):
        from human_eval.contracts import RunRequest
        from human_eval.store import EvaluationStore
        with tempfile.TemporaryDirectory() as directory:
            store = EvaluationStore(Path(directory) / "runs.sqlite")
            store.initialize()
            run, _ = store.create_run(RunRequest("A question"), user_id="alice", reserved_tokens=30_000)
            run_id = run["run_id"]
            self.assertEqual(store.get_request(run_id).token_limit, 30_000)
            store.start_run(run_id, provenance={})
            store.append_event(run_id, "succeeded", {"usage": {"total_tokens": 1234}})
            self.assertEqual(store.token_balance("alice"), DAILY_TOKEN_LIMIT - 1234)
            failed, _ = store.create_run(RunRequest("Another question"), user_id="alice", reserved_tokens=10_000)
            store.append_event(failed["run_id"], "failed", {})
            self.assertEqual(store.token_balance("alice"), DAILY_TOKEN_LIMIT - 1234 - 10_000)
            interrupted, _ = store.create_run(RunRequest("Interrupted question"), user_id="alice", reserved_tokens=10_000)
            store.start_run(interrupted["run_id"], provenance={})
            store.fail_interrupted_runs()
            with store._connect() as connection:
                used = connection.execute("SELECT used FROM token_reservations WHERE run_id = ?", (interrupted["run_id"],)).fetchone()[0]
            self.assertEqual(used, 10_000)

    def test_excess_usage_does_not_block_terminal_event(self):
        from human_eval.contracts import RunRequest
        from human_eval.store import EvaluationStore
        with tempfile.TemporaryDirectory() as directory:
            store = EvaluationStore(Path(directory) / "runs.sqlite")
            store.initialize()
            for state in ("succeeded", "failed"):
                run, _ = store.create_run(
                    RunRequest("A question"), user_id=state, reserved_tokens=1000
                )
                run_id = run["run_id"]
                store.start_run(run_id, provenance={})
                store.append_event(run_id, state, {"usage": {"total_tokens": 1001}})
                self.assertEqual(store.get_run(run_id)["status"], state)
                self.assertEqual(store.token_balance(state), DAILY_TOKEN_LIMIT - 1000)
                with store._connect() as connection:
                    used = connection.execute(
                        "SELECT used FROM token_reservations WHERE run_id = ?", (run_id,)
                    ).fetchone()[0]
                self.assertEqual(used, 1000)

    def test_shared_reservations_settlement_and_rolling_window(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.sqlite"
            with sqlite3.connect(path) as first, sqlite3.connect(path) as second:
                initialize(first)
                first.commit()
                first.execute("BEGIN IMMEDIATE")
                reserve(first, "a", "alice", DAILY_TOKEN_LIMIT - 20_000)
                reserve(first, "a", "alice", DAILY_TOKEN_LIMIT - 20_000)  # Retry is free.
                first.commit()
                self.assertEqual(balance(second, "alice"), 20_000)
                self.assertEqual(balance(second, "bob"), DAILY_TOKEN_LIMIT)
                second.execute("BEGIN IMMEDIATE")
                with self.assertRaises(TokenBudgetExceeded):
                    reserve(second, "b", "alice", 30_000)
                second.rollback()
                with self.assertRaises(ValueError):
                    settle(first, "a", DAILY_TOKEN_LIMIT - 20_000 + 1)
                settle(first, "a", 10_000)
                first.commit()
                self.assertEqual(balance(second, "alice"), DAILY_TOKEN_LIMIT - 10_000)
                first.execute("UPDATE token_reservations SET created_at = '2000-01-01' WHERE run_id = 'a'")
                first.commit()
                self.assertEqual(balance(second, "alice"), DAILY_TOKEN_LIMIT)
                first.execute("BEGIN IMMEDIATE")
                reserve(first, "c", "alice", DAILY_TOKEN_LIMIT)
                first.execute("UPDATE token_reservations SET created_at = '2000-01-01' WHERE run_id = 'c'")
                first.commit()
                # Active reservations cannot expire while work is still running.
                self.assertEqual(balance(second, "alice"), 0)


if __name__ == "__main__":
    unittest.main()
