"""Clerk-backed authentication via AirClerk.

This is the only module that imports airclerk or the Clerk SDK. It is
imported lazily by ``app.build_default_app`` because airclerk validates its
environment variables at import time; tests and offline development use
``auth.FakeAuthBackend`` instead and never import this module.
"""

from __future__ import annotations

import asyncio
import html
import json
import time

import airclerk
import httpx
from clerk_backend_api import Clerk
from clerk_backend_api.security.types import AuthenticateRequestOptions
from starlette.requests import Request

from .auth import ROLE_ADMIN, ROLE_USER, User


def _json_script_value(value: str) -> str:
    """Serialize a value for a JavaScript string inside an HTML script tag."""
    return (
        json.dumps(value)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


class ClerkAuthBackend:
    """Verify Clerk session JWTs and map Clerk users to local User records.

    The Clerk role comes from ``public_metadata.role == "admin"`` (managed in
    the Clerk dashboard). Lookups are cached briefly per session token so a
    page load does not turn into repeated Clerk API round-trips.
    """

    def __init__(self, *, secret_key: str, cache_ttl: float = 60.0):
        self.secret_key = secret_key
        self.cache_ttl = cache_ttl
        self._cache: dict[str, tuple[float, User | None]] = {}

    async def get_user(self, request: Request) -> User | None:
        token = request.cookies.get("__session") or request.headers.get(
            "authorization", ""
        )
        if not token:
            return None
        cached = self._cache.get(token)
        if cached and cached[0] > time.monotonic():
            return cached[1]
        body = await request.body()
        user = await asyncio.to_thread(self._authenticate, request, body)
        if len(self._cache) > 1000:
            self._cache.clear()
        self._cache[token] = (time.monotonic() + self.cache_ttl, user)
        return user

    def _authenticate(self, request: Request, body: bytes) -> User | None:
        httpx_request = httpx.Request(
            method=request.method,
            url=str(request.url),
            headers=dict(request.headers),
            content=body,
        )
        origin = f"{request.url.scheme}://{request.url.netloc}"
        with Clerk(bearer_auth=self.secret_key) as clerk:
            state = clerk.authenticate_request(
                httpx_request,
                AuthenticateRequestOptions(authorized_parties=[origin]),
            )
            if not state.is_signed_in:
                return None
            user_id = getattr(state, "user_id", None) or state.payload.get("sub")
            clerk_user = clerk.users.get(user_id=user_id)
        metadata = getattr(clerk_user, "public_metadata", None) or {}
        role = metadata.get("role") if isinstance(metadata, dict) else None
        email = None
        emails = getattr(clerk_user, "email_addresses", None) or []
        if emails:
            email = getattr(emails[0], "email_address", None)
        return User(
            id=str(user_id),
            role=ROLE_ADMIN if role == ROLE_ADMIN else ROLE_USER,
            email=email,
        )


def clerk_page_scripts(user: User | None) -> str:
    """Render Clerk JS tags that keep client/server auth state in sync.

    After an OAuth round-trip (or modal sign-in), Clerk's client sets the
    first-party session cookie only once the page's JS runs, so the server
    may render the anonymous version of a page for a signed-in user (or
    vice versa). The inline script detects that mismatch and reloads once.
    Unlike airclerk.clerk_scripts, the loop-guard flag is cleared whenever
    the states match again, so a later sign-in/out in the same tab still
    triggers its resync reload.
    """
    src = html.escape(airclerk.settings.CLERK_JS_SRC, quote=True)
    key = html.escape(airclerk.settings.CLERK_PUBLISHABLE_KEY, quote=True)
    server_has_user = "true" if user is not None else "false"
    return (
        f'<script src="{src}" crossorigin="anonymous" '
        f'data-clerk-publishable-key="{key}"></script>'
        "<script>document.addEventListener('DOMContentLoaded', async () => {"
        "if (!window.Clerk) return;"
        "await window.Clerk.load();"
        f"const serverHasUser = {server_has_user};"
        "const clerkHasUser = !!window.Clerk.user;"
        "const reloadKey = 'clerk_auth_reloaded';"
        "if (serverHasUser === clerkHasUser) {"
        "sessionStorage.removeItem(reloadKey);"
        "return;"
        "}"
        "if (!sessionStorage.getItem(reloadKey)) {"
        "sessionStorage.setItem(reloadKey, '1');"
        "window.location.reload();"
        "}"
        "});</script>"
    )


def clerk_login_scripts(next_url: str) -> str:
    """Render Clerk JS plus the SignIn mount for the styled /login page.

    ``next_url`` must already be sanitized to a same-origin path; it is
    emitted through ``json.dumps`` so it is always a safe JS string literal.
    """
    src = html.escape(airclerk.settings.CLERK_JS_SRC, quote=True)
    key = html.escape(airclerk.settings.CLERK_PUBLISHABLE_KEY, quote=True)
    target = _json_script_value(next_url)
    return (
        f'<script src="{src}" crossorigin="anonymous" '
        f'data-clerk-publishable-key="{key}"></script>'
        "<script>document.addEventListener('DOMContentLoaded', async () => {"
        "if (!window.Clerk) return;"
        "await window.Clerk.load();"
        f"if (window.Clerk.user) {{ window.location.assign({target}); return; }}"
        "window.Clerk.mountSignIn(document.getElementById('sign-in'),"
        f" {{ redirectUrl: {target} }});"
        "});</script>"
    )
