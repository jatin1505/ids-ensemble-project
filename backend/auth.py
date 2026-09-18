"""
backend/auth.py

Google OAuth verification + role-based session tokens for the dashboard.

Flow
----
1. Frontend loads Google Identity Services, gets an ID token when the
   user signs in with a Google account.
2. Frontend POSTs that ID token to POST /auth/google.
3. This module verifies the token really came from Google AND was issued
   for OUR OAuth client (GOOGLE_CLIENT_ID) -- skipping either check is a
   real vulnerability, not a formality. verify_google_id_token() below
   does both via google-auth's own verify_oauth2_token, which also
   checks signature and expiry -- never hand-roll this part.
4. The verified email is checked against ADMIN_EMAILS. Anyone else who
   signs in successfully gets "user" role -- NOT rejected outright.
   Whether unauthenticated visitors get any access at all is a product
   decision made in backend/websocket_manager.py, not here.
5. This module issues its OWN short-lived signed token (HS256 JWT)
   encoding {email, role, exp}. The frontend re-sends THIS token (not
   the Google one) on the WebSocket connection, since browsers cannot
   attach custom headers to a WS handshake -- it goes as a query param:
   ws://host/ws?token=<this JWT>.

Why a second token instead of re-verifying the Google ID token on every
WS connection: Google ID tokens are short-lived (~1hr) and verifying one
requires a network round-trip to Google. Our own JWT is verified locally
(signature check only) and encodes OUR role decision once, at sign-in
time, instead of re-deriving it per connection.

SECURITY NOTE this file cannot enforce by itself: verifying a token here
is necessary but not sufficient. The actual access control is in
backend/websocket_manager.py's broadcast() -- it must build a genuinely
different, smaller payload for non-admin connections, not just trust the
frontend to hide fields in the UI. A user who opens devtools and reads
raw WebSocket frames will see exactly what the backend chose to send
them, nothing less.

Environment variables required
-------------------------------
GOOGLE_CLIENT_ID   From Google Cloud Console. See the setup guide.
APP_JWT_SECRET     Any long random string -- used to sign OUR tokens.
                   Generate with:
                       python -c "import secrets; print(secrets.token_urlsafe(32))"
                   Never reuse this secret for anything else. Never commit it.
ADMIN_EMAILS       Comma-separated allow-list, e.g.
                       "you@gmail.com,teammate@gmail.com"
                   Case-insensitive; whitespace around entries is trimmed.

Dev-only bypass
----------------
dev_bypass_enabled() gates a route (registered in backend/main.py) that
mints a role token WITHOUT a real Google sign-in, for local iteration
before you have real Google credentials wired up. It requires BOTH
AUTH_MODE=mock and ENVIRONMENT=local to be true at once -- a single
flag flip in a deployed environment can't accidentally expose it.
NEVER set AUTH_MODE=mock anywhere other than your own local machine.
"""

import os
import time

import jwt
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
APP_JWT_SECRET = os.environ.get("APP_JWT_SECRET", "")
ADMIN_EMAILS = {
    e.strip().lower() for e in os.environ.get("ADMIN_EMAILS", "").split(",") if e.strip()
}
TOKEN_TTL_SECONDS = 8 * 60 * 60  # 8 hours -- comfortably covers one review/demo session


class AuthConfigError(RuntimeError):
    """Raised at import time if required secrets aren't set -- fails loud
    and immediately, rather than letting a misconfigured server start and
    fail confusingly on the first real request."""


if not APP_JWT_SECRET:
    raise AuthConfigError(
        "APP_JWT_SECRET is not set. Generate one with:\n"
        '    python -c "import secrets; print(secrets.token_urlsafe(32))"\n'
        "and put it in your .env -- never hardcode it in source."
    )


def verify_google_id_token(google_id_token_str: str) -> dict:
    """Returns the verified claims dict (includes 'email', 'email_verified',
    'name', etc.) on success. Raises ValueError on ANY invalid, forged,
    expired, or wrong-audience token -- callers must not swallow this."""
    if not GOOGLE_CLIENT_ID:
        raise AuthConfigError("GOOGLE_CLIENT_ID is not configured on the backend.")
    claims = id_token.verify_oauth2_token(
        google_id_token_str, google_requests.Request(), GOOGLE_CLIENT_ID
    )
    # verify_oauth2_token already checks signature, expiry, and audience
    # (must match OUR client ID) -- if it returns without raising, the
    # token is genuine and was issued for this app specifically.
    if not claims.get("email_verified", False):
        raise ValueError("Google account email is not verified")
    return claims


def role_for_email(email: str) -> str:
    return "admin" if email.strip().lower() in ADMIN_EMAILS else "user"


def issue_app_token(email: str, role: str) -> str:
    payload = {"email": email, "role": role, "exp": time.time() + TOKEN_TTL_SECONDS}
    return jwt.encode(payload, APP_JWT_SECRET, algorithm="HS256")


def verify_app_token(token: str) -> dict:
    """Raises jwt.PyJWTError (expired, bad signature, malformed) on any
    invalid token. Callers -- especially the WS handshake -- must treat
    any exception here as 'reject the connection', never 'default to a
    role'. There is no safe default role for a token that failed
    verification."""
    return jwt.decode(token, APP_JWT_SECRET, algorithms=["HS256"])


def dev_bypass_enabled() -> bool:
    """Both flags required on purpose -- see module docstring. Check this
    at ROUTE-REGISTRATION time in main.py (so the route doesn't exist at
    all in a normal run), not just inside the handler."""
    return os.environ.get("AUTH_MODE") == "mock" and os.environ.get("ENVIRONMENT") == "local"