
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