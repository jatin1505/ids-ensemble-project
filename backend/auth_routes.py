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
     print(f"[auth] Google token verification failed: {type(e).__name__}: {e}")
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