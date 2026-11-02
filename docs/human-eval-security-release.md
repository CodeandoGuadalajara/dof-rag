# Public-release security audit: `human_eval/`

Date of audit: current tree (post-audit commit). Scope: authentication and
authorization, HTML/JS rendering, request handling, persistence, and
deployment of the human-evaluation web app and its scheduler. The audit was
performed by reading the current code and reproducing candidate findings
against it; earlier findings were treated as hypotheses.

This document is a **release-readiness assessment of the code**, not a
statement that the live deployment is production-ready. The operator
decisions below remain open.

## Verified fixes (with regression tests)

1. **Deleted runs left chat content and token reservations behind**
   (`store.py::delete_run`). Confirmed by reproduction: after deleting a chat
   run, `chat_messages`, `chat_conversations`, and `token_reservations` still
   held the question, answer and reservation. Fixed: deletion now removes
   chat messages and the reservation, and drops conversations only when they
   have no messages left. Tests:
   `test_delete_run_removes_chat_and_token_data`,
   `test_delete_run_keeps_nonempty_conversations`. Note: the pre-existing
   production database may still contain orphaned rows from earlier
   deletions (one-time cleanup is an operator decision, below).
2. **Body-size cap bypassed for chunked requests** (`app.py::_form`). The cap
   only inspected `Content-Length`; requests without it (chunked transfer)
   were parsed unbounded. Fixed: the actual body is read and measured before
   form parsing (Starlette caches it, so no double read). Test:
   `RequestLimitTests::test_chunked_body_over_limit_is_rejected`.
3. **Missing HSTS on HTTPS deployments**. Fixed: `security_headers` now adds
   `Strict-Transport-Security: max-age=31536000; includeSubDomains` only when
   `DOF_SECURE_COOKIE` is enabled (i.e. HTTPS-only deployments; enabling it
   on plain HTTP is meaningless and unsafe). Test: `SecurityHeaderTests`.
4. **Post-action redirect hardening** (`app.py::_safe_next`). The redirect
   helper used after feedback/moderation accepted any `/`-prefixed string
   (control characters, backslashes). It now applies the same rules as the
   login-flow sanitizer. Test:
   `test_safe_next_rejects_browser_normalized_external_paths`.
5. **Vulnerable authentication-path dependencies** (`uv.lock`). `pip-audit`
   flagged `pyjwt 2.13.0` (17 advisories), `urllib3 2.3.0`, `requests`,
   `cryptography`, `anyio`, `certifi`, `idna`, `dnspython`, `pyasn1`,
   `python-dotenv`, `setuptools`. These packages sit on the live request path
   (Clerk JWT verification, TLS, HTTP). They were upgraded in `uv.lock` to
   patched versions (pyjwt 2.15.1, urllib3 2.8.0, cryptography 50.0.2,
   requests 2.34.2, …). Re-audit shows **zero known advisories** for every
   package loaded by the web/scheduler processes; `clerk_auth` imports and
   the full test suite pass on the new lock.

   Remaining advisories (`pillow`, `tornado`, `accelerate`, `filelock`,
   `fsspec`, `soupsieve`, `jupyter-core`) belong to the offline
   corpus/embedding toolchain (`marker-pdf`, `surya-ocr`, `transformers`,
   `peft`, `ipykernel`). Verified by import inspection: none of these modules
   is imported by `human_eval` or `agent_tools`, and none is present in
   `sys.modules` after `import human_eval.app`. They are not exposed through
   the web app; upgrading them is left to the corpus-tooling owners because
   those pins are workload-sensitive.

## Reviewed and considered sound (no change needed)

- **Authorization**: `/runs`, `/chat`, feedback, SSE and status endpoints all
  resolve the user server-side and check `run_belongs_to`/admin before any
  read; other users' private runs return 404/redirect rather than content.
  `/answers/{id}` only serves published+terminal runs. Chat runs can never be
  published (`publish_run` refuses `kind='chat'`). Admin endpoints are
  role-gated on every method, not just hidden in the UI.
- **XSS/rendering**: all interpolated values are HTML-escaped; Markdown goes
  through `markdown-it` (raw HTML disabled) + `nh3` allowlist sanitization,
  and the streamed `excerpt_html` path sanitizes server-side before
  `innerHTML`. CSP is present; Clerk scripts use JSON-encoded, escaped
  values. Login `next` sanitization rejects `//`, backslashes, control
  characters and absolute URLs on both server and client.
- **CSRF**: every mutating form validates a per-session token with
  `hmac.compare_digest`; session cookie is `SameSite=lax`.
- **Request handling**: contracts reject unknown fields and validate types,
  lengths, dates, hop ranges and idempotency-key format; `as_of` is parsed as
  a real ISO date; conversation ids must be UUIDs owned by the caller
  (`chat_store.add_message` raises on cross-user conversation ids).
- **Persistence**: all queries are parameterized; admission decisions
  (idempotency, one active run, quota, review gate, queue capacity, token
  reservation) happen inside one `BEGIN IMMEDIATE` transaction; token
  settlement never exceeds the reservation; scheduler startup recovers
  orphaned runs. Execution singleton is enforced by a non-inheritable flock.
- **Deployment defaults**: loopback bind by default; `DOF_SESSION_SECRET` ≥ 32
  chars required; `TrustedHostMiddleware` enabled; Uvicorn access log off (no
  IP logging); model API keys never rendered (provenance contains only model
  names/configuration); failed runs expose generic messages, with details
  only in logs.

## Open weaknesses (documented, not fixed here)

- **`ClerkAuthBackend` caches authorization for 60 s**: a revoked admin role
  or signed-out session can remain effective for up to `cache_ttl`. Acceptable
  for a pilot; lower `cache_ttl` for stricter revocation.
- **No application-level rate limiting on `/login` or feedback**; expensive
  model work is bounded by the per-user active-run rule, the 24 h quota, the
  queue cap and the token budget. Edge-level rate limiting is recommended
  before public exposure.

## Secure deployment requirements

- Bind to loopback and front with HTTPS (reverse proxy or Tailscale with
  HTTPS); set `DOF_SECURE_COOKIE=true` (also enables HSTS), a random
  `DOF_SESSION_SECRET` (≥32 chars), and a restrictive `DOF_ALLOWED_HOSTS`.
- Never expose the model API (`0.0.0.0` llama/TensorFold servers) publicly;
  the web app must be the only public surface.
- Use the systemd units in `ops/systemd/` (now with `NoNewPrivileges`,
  `PrivateTmp`, `ProtectSystem=full`). After changing the units, an operator
  must run `scripts/install_human_eval_systemd.sh` and restart the services;
  that was deliberately **not** done as part of this audit.
- Run exactly one scheduler per database; web workers are read/admit only.
- Keep `.env` out of source control and out of logs; do not print secrets.
- Add edge rate limiting (login, feedback, SSE) and monitoring before
  opening registration publicly.

## Privacy and retention requirements

- Stored per user: Clerk user id and email (session display), questions,
  answers, evidence, tool traces, model reasoning text, feedback, token
  reservations. IP addresses are not logged (Uvicorn access log disabled;
  reverse proxy must be configured likewise if that promise is kept).
- Published answers (question + answer + evidence) are public; admins must
  review questions for personal data before publishing (UI already warns).
- Admin deletion now removes all rows tied to a run, including chat content
  and reservations. **There is no user-facing self-service deletion or
  retention limit**: accounts and their history persist until an admin
  deletes them. A public launch needs a published retention policy and a
  user data-deletion process (Clerk account deletion does not cascade into
  this database).
- One-time cleanup of pre-fix orphaned rows (chat messages / reservations
  whose run no longer exists) is an operator decision; no automated migration
  was added because it deletes data.

## Remaining release blockers (operator decisions required)

1. **Legal/privacy sign-off**: publish the retention and deletion policy and
   implement a user data-deletion process (see above). Optional SQL cleanup
   of pre-existing orphaned rows.
2. **Apply deployment changes**: reinstall/restart the hardened systemd
   units and set `DOF_SECURE_COOKIE=true` behind HTTPS on the real host.
3. **Dependency refresh on the host**: run `uv sync --frozen` so the patched
   `uv.lock` takes effect (the lock was upgraded and tested, not deployed).
4. **Edge controls**: HTTPS termination, rate limiting, and a decision on
   open registration vs. invitation-only for the public pilot.
5. **Offline toolchain advisories** (pillow/tornado/etc.): schedule upgrades
   with the corpus/embedding owners; not a web-exposure blocker but tracked
   debt.

## Verification performed

- `pytest tests/` — 279 passed, 1 skipped (includes 5 new regression tests).
- `ruff check human_eval/store.py human_eval/app.py tests/test_human_eval.py`
  — clean.
- `pip-audit` before/after the lock upgrade; residual advisories confirmed
  unreachable from the web process by import inspection.
- Reproduction script for the deletion leak (before/after) and chunked-body
  cap (regression test).

No live services were changed, no models were restarted, no benchmarks run,
and no public exposure was added as part of this audit.
