"""UserProfile 的独立持久化门面。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.core.models import MeasurementSystem, ResponseStyle, UserProfile
from app.state import StateStore


class UserProfileService:
    """只管理用户明确配置的工作交互偏好。"""

    def __init__(self, store: StateStore) -> None:
        self.store = store

    def get_or_create(self, user_id: str) -> UserProfile:
        profile = self.store.get_user_profile(user_id)
        if profile is not None:
            return profile
        profile = UserProfile(user_id=user_id)
        self.store.save_user_profile(profile)
        return profile

    def get(self, user_id: str) -> UserProfile | None:
        return self.store.get_user_profile(user_id)

    def update(self, user_id: str, changes: dict[str, Any]) -> UserProfile:
        current = self.get_or_create(user_id)
        allowed = {"language", "response_style", "measurement_system", "preferred_output_format"}
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"不支持的用户偏好字段：{', '.join(sorted(unknown))}")
        normalized = dict(changes)
        if "language" in normalized:
            normalized["language"] = _language(normalized["language"])
        if "response_style" in normalized and normalized["response_style"] is not None:
            normalized["response_style"] = ResponseStyle(normalized["response_style"])
        if "measurement_system" in normalized and normalized["measurement_system"] is not None:
            normalized["measurement_system"] = MeasurementSystem(normalized["measurement_system"])
        if "preferred_output_format" in normalized:
            normalized["preferred_output_format"] = _output_format(normalized["preferred_output_format"])
        updated = current.model_copy(update={**normalized, "updated_at": datetime.now(UTC)})
        self.store.save_user_profile(updated)
        return updated


def _language(value: Any) -> str:
    normalized = str(value).strip()
    aliases = {"中文": "zh-CN", "zh": "zh-CN", "zh-cn": "zh-CN", "英文": "en-US", "english": "en-US", "en": "en-US", "en-us": "en-US"}
    if normalized.casefold() in {key.casefold() for key in aliases}:
        return aliases[next(key for key in aliases if key.casefold() == normalized.casefold())]
    if normalized not in {"zh-CN", "en-US"}:
        raise ValueError("语言只支持 zh-CN 或 en-US")
    return normalized


def _output_format(value: Any) -> str | None:
    if value is None or not str(value).strip():
        return None
    normalized = str(value).strip().upper().replace(".", "")
    aliases = {"GEOPACKAGE": "GeoPackage", "GPKG": "GeoPackage", "GEOJSON": "GeoJSON", "GEOTIFF": "GeoTIFF", "TIFF": "GeoTIFF", "CSV": "CSV"}
    if normalized not in aliases:
        raise ValueError("默认输出格式只支持 GeoPackage、GeoJSON、GeoTIFF 或 CSV")
    return aliases[normalized]


__all__ = ["UserProfileService"]
