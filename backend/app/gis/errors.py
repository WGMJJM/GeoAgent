"""GIS 计算的可分类错误。"""

from __future__ import annotations

from typing import Any

from app.core.models import ErrorCategory, ToolError


class GISFailure(Exception):
    """可由 FailureAnalyzer 决定恢复动作的 GIS 异常。"""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.error = ToolError(
            code=code,
            category=category,
            message=message,
            retryable=retryable,
            details=details or {},
        )


def as_tool_error(exc: Exception) -> ToolError:
    if isinstance(exc, GISFailure):
        return exc.error
    return ToolError(code="UNEXPECTED_ERROR", message=str(exc) or exc.__class__.__name__)

