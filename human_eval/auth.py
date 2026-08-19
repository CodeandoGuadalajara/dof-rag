"""Authentication seam: the app depends on AuthBackend, never on a provider.

Production wires ClerkAuthBackend (see clerk_auth.py, imported lazily so
tests and offline development never require Clerk credentials). Tests inject
FakeAuthBackend and identify users through headers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from starlette.requests import Request

ROLE_ADMIN = "admin"
ROLE_USER = "user"


@dataclass(frozen=True)
class User:
    id: str
    role: str = ROLE_USER
    email: str | None = None

    @property
    def is_admin(self) -> bool:
        return self.role == ROLE_ADMIN


class AuthBackend(Protocol):
    """Resolve the current request to a user, or None when anonymous."""

    async def get_user(self, request: Request) -> User | None: ...


class FakeAuthBackend:
    """Header-driven backend for tests and local development without Clerk.

    Send ``X-Eval-User: <id>`` and optionally ``X-Eval-Role: admin``.
    Requests without the header are anonymous.
    """

    def __init__(self, users: dict[str, User] | None = None):
        self.users = dict(users or {})

    async def get_user(self, request: Request) -> User | None:
        user_id = request.headers.get("x-eval-user")
        if not user_id:
            return None
        known = self.users.get(user_id)
        if known is not None:
            return known
        role = request.headers.get("x-eval-role", ROLE_USER)
        return User(id=user_id, role=role)
