from fastapi import Header, HTTPException
from gotrue.errors import AuthApiError
from typing import Optional 
from database import supabase
import logging

logger = logging.getLogger(__name__)
async def optional_verify_token(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """for guest users with no token"""
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token or token.lower() in ("undefined", "null", ""):
        return None
    try: 
        return await verify_token(authorization)
    except HTTPException:
        return None

async def verify_token(authorization: str = Header(...)) -> str:
    try:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(401, "Invalid or missing Authorization header")

        if not supabase:
            print("DEBUG: No Supabase client available for token verification")
            raise HTTPException(401, "Authentication service unavailable")

        logger.debug("Verifying token: %.20s...", token)
        resp = supabase.auth.get_user(token)

        if not resp.user:
            raise HTTPException(401, "Invalid authentication credentials")
        user_id = resp.user.id
        if not user_id:
            raise HTTPException(401, "Could not extract user ID")

        logger.debug("Verified user_id: %s", user_id)
        return user_id
        
    except AuthApiError as e:
        print(f"DEBUG: AuthApiError: {e.message}")
        raise HTTPException(401, f"Authentication failed: {e.message}")
    except Exception as e:
        print(f"DEBUG: Exception in verify_token: {str(e)}")
        raise HTTPException(500, f"Internal server error: {str(e)}")

