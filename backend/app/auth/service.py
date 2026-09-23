"""用户注册、登录和 HttpOnly Session 生命周期。"""

from __future__ import annotations

import secrets
from datetime import timedelta

from app.config import Settings
from app.core.models import User, UserSession, UserView, utc_now
from app.state import StateStore

from .password import hash_password, hash_session_token, verify_password


class AuthenticationError(ValueError):
    """认证失败，不暴露用户是否存在等额外信息。"""


class AuthService:
    def __init__(self, store: StateStore, settings: Settings) -> None:
        self.store = store
        self.settings = settings

    def register(self, username: str, password: str, *, email: str | None = None, display_name: str | None = None) -> User:
        normalized_username = _normalize_username(username)
        normalized_email = _normalize_email(email)
        if len(password) < 8:
            raise ValueError("密码至少需要 8 个字符")
        if self.store.get_user_by_username(normalized_username) is not None:
            raise ValueError("用户名已存在")
        if normalized_email and self.store.get_user_by_email(normalized_email) is not None:
            raise ValueError("邮箱已存在")
        user = User(
            username=normalized_username,
            email=normalized_email,
            password_hash=hash_password(password),
            display_name=(display_name or normalized_username).strip() or normalized_username,
        )
        self.store.save_user(user)
        return user

    def login(self, identifier: str, password: str) -> tuple[User, str]:
        value = identifier.strip()
        user = self.store.get_user_by_username(value.casefold()) or self.store.get_user_by_email(value.casefold())
        if user is None or not user.is_active or not verify_password(password, user.password_hash):
            raise AuthenticationError("用户名或密码错误")
        token = secrets.token_urlsafe(32)
        now = utc_now()
        session = UserSession(
            user_id=user.id,
            token_hash=hash_session_token(token),
            expires_at=now + timedelta(hours=self.settings.auth_session_ttl_hours),
            last_seen_at=now,
        )
        self.store.save_session(session)
        return user, token

    def authenticate_token(self, token: str | None) -> User | None:
        if not token:
            return None
        session = self.store.get_session(hash_session_token(token))
        if session is None or session.expires_at <= utc_now():
            if session is not None:
                self.store.delete_session(session.id)
            return None
        user = self.store.get_user(session.user_id)
        if user is None or not user.is_active:
            self.store.delete_session(session.id)
            return None
        self.store.touch_session(session.id, utc_now())
        return user

    def logout(self, token: str | None) -> None:
        if token:
            self.store.delete_session_by_token(hash_session_token(token))

    def update_user(self, user: User, *, display_name: str, email: str | None) -> User:
        normalized_email = _normalize_email(email)
        existing = self.store.get_user_by_email(normalized_email) if normalized_email else None
        if existing is not None and existing.id != user.id:
            raise ValueError("邮箱已存在")
        updated = user.model_copy(update={"display_name": display_name.strip() or user.username, "email": normalized_email, "updated_at": utc_now()})
        self.store.save_user(updated)
        return updated

    def bootstrap_if_configured(self) -> User | None:
        username = self.settings.bootstrap_user.strip()
        password = self.settings.bootstrap_password
        if not username or not password or self.store.count_users() > 0:
            return None
        return self.register(username, password, email=self.settings.bootstrap_email, display_name=self.settings.bootstrap_display_name or username)

    @staticmethod
    def view(user: User) -> UserView:
        return UserView.from_user(user)


def _normalize_username(value: str) -> str:
    normalized = value.strip().casefold()
    if len(normalized) < 3 or len(normalized) > 64 or any(char.isspace() for char in normalized):
        raise ValueError("用户名长度应为 3 到 64 个字符且不能包含空格")
    return normalized


def _normalize_email(value: str | None) -> str | None:
    normalized = value.strip().casefold() if value else None
    return normalized or None


__all__ = ["AuthenticationError", "AuthService"]
