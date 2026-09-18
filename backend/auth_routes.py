"""
backend/auth_routes.py

HTTP endpoints for the sign-in flow. Included in backend/main.py the same
way websocket_manager.router already is:

    from backend.auth_routes import router as auth_router
    app.include_router(auth_router)

POST /auth/google
    Body: {"credential": "<Google ID token from the frontend's Sign-In button>"}
    Verifies it against Google, decides admin/user via the ADMIN_EMAILS
    allow-list, and returns OUR app-level token for the frontend to use
    on the WebSocket connection.

POST /auth/dev   (only registered when auth.dev_bypass_enabled() is True)
    Body: {"email": "someone@example.com"}
    Skips Google entirely -- mints a token using the SAME role logic
    (still checked against ADMIN_EMAILS), so you can build/test the
    admin/user split locally before real Google credentials exist. This
    route does not exist at all unless AUTH_MODE=mock AND
    ENVIRONMENT=local are both set -- see backend/auth.py.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend import auth

router = APIRouter()


class GoogleSignInRequest(BaseModel):
    credential: str


class DevSignInRequest(BaseModel):
    email: str


class SignInResponse(BaseModel):
    token: str
    email: str
    role: str


@router.post("/auth/google", response_model=SignInResponse)
async def auth_google(body: GoogleSignInRequest):
    try:
        claims = auth.verify_google_id_token(body.credential)
    except Exception as e:
        # Deliberately vague to the client (don't leak WHY verification
        # failed -- expired vs. forged vs. wrong audience are all just
        # "no" from the caller's perspective) -- but keep it server-side
        # loggable by re-raising as HTTPException with the real reason
        # only in the server log, not the response body, in a real deploy.
        raise HTTPException(status_code=401, detail="Google sign-in could not be verified")

    email = claims["email"]
    role = auth.role_for_email(email)
    token = auth.issue_app_token(email, role)
    return SignInResponse(token=token, email=email, role=role)


if auth.dev_bypass_enabled():
    @router.post("/auth/dev", response_model=SignInResponse)
    async def auth_dev(body: DevSignInRequest):
        role = auth.role_for_email(body.email)
        token = auth.issue_app_token(body.email, role)
        return SignInResponse(token=token, email=body.email, role=role)

    print(
        "[auth] WARNING: /auth/dev bypass route is ACTIVE "
        "(AUTH_MODE=mock, ENVIRONMENT=local). Never set these in a "
        "deployed environment."
    )