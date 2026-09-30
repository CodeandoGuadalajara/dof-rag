import unittest
from types import SimpleNamespace
from human_eval.budget_backend import BudgetBackend, TokenCeilingReached


class BudgetBackendTests(unittest.TestCase):
    def test_caps_output_accounts_all_turns_and_restores_backend(self):
        calls = []
        backend = SimpleNamespace(max_output_tokens=20)
        def create_turn(**kwargs):
            calls.append(backend.max_output_tokens)
            return SimpleNamespace(usage={"total_tokens": 10 + backend.max_output_tokens})
        backend.create_turn = create_turn
        guard = BudgetBackend(backend, 55, lambda **kwargs: 10)
        guard.create_turn()
        guard.create_turn()
        self.assertEqual(calls, [20, 15])
        self.assertEqual(guard.used, 55)
        self.assertEqual(backend.max_output_tokens, 20)
        with self.assertRaises(TokenCeilingReached) as stopped:
            guard.create_turn()
        self.assertEqual(stopped.exception.used, 55)
        self.assertEqual(len(calls), 2)

    def test_missing_usage_fails_closed(self):
        backend = SimpleNamespace(max_output_tokens=20,
            create_turn=lambda **kwargs: SimpleNamespace(usage={}))
        guard = BudgetBackend(backend, 50, lambda **kwargs: 10)
        with self.assertRaises(TokenCeilingReached):
            guard.create_turn()
        self.assertEqual(backend.max_output_tokens, 20)
        with self.assertRaises(TokenCeilingReached):
            guard.create_turn()


if __name__ == "__main__":
    unittest.main()
