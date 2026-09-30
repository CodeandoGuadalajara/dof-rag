"""Fail-closed per-run guard for backends with a verified input-token counter.

The counter must include instructions, tool/response schemas, history and provider
framing. Do not wire a character estimate here as if it were exact tokenization.
"""
from collections.abc import Callable
from typing import Any


class TokenCeilingReached(RuntimeError):
    def __init__(self, message: str, *, used: int | None = None):
        super().__init__(message)
        self.used = used


class BudgetBackend:
    def __init__(self, backend: Any, limit: int | None, count_input: Callable[..., int], *, context_limit: int = 262_144):
        if limit is not None and limit < 1:
            raise ValueError("token limit must be positive")
        self.backend = backend
        self.limit = limit
        self.context_limit = context_limit
        self.count_input = count_input
        self.used = 0
        self.blocked = False

    @property
    def model(self) -> str:
        return self.backend.model

    def create_turn(self, **kwargs: Any) -> Any:
        if self.blocked:
            raise TokenCeilingReached("token accounting is unavailable or exhausted")
        inputs = self.count_input(**kwargs)
        if not isinstance(inputs, int) or isinstance(inputs, bool) or inputs < 0:
            raise ValueError("input counter must return a non-negative integer")
        remaining = self.context_limit - inputs
        if self.limit is not None:
            remaining = min(remaining, self.limit - self.used - inputs)
        if remaining <= 0:
            raise TokenCeilingReached("not enough tokens for another model call", used=self.used)
        original = self.backend.max_output_tokens
        self.backend.max_output_tokens = min(original, remaining)
        # A failed call may have consumed tokens: don't retry with this guard.
        self.blocked = True
        try:
            turn = self.backend.create_turn(**kwargs)
        finally:
            self.backend.max_output_tokens = original
        usage = turn.usage
        total = usage.get("total_tokens")
        if not isinstance(total, int) or isinstance(total, bool) or total <= 0:
            raise TokenCeilingReached("provider did not report reliable token usage")
        self.used += total
        if self.limit is not None and self.used > self.limit:
            raise TokenCeilingReached("provider usage exceeded the counted ceiling")
        self.blocked = False
        return turn
