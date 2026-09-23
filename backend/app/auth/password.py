"""密码和 Session token 的安全处理。"""

from __future__ import annotations

import hashlib

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

_PASSWORD_HASHER = PasswordHasher()


def hash_password(password: str) -> str:
    if not password:
        raise ValueError("密码不能为空")
    return _PASSWORD_HASHER.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    if not password or not password_hash:
        return False
    try:
        return _PASSWORD_HASHER.verify(password_hash, password)
    except (InvalidHashError, VerificationError, VerifyMismatchError):
        return False


def hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


__all__ = ["hash_password", "verify_password", "hash_session_token"]
