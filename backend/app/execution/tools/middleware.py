"""Tool 中间件挂点。

第一版将审计集中在 ToolExecutor；这个协议为后续限流、审批和外部 Connector
增加拦截器保留稳定入口。
"""

from collections.abc import Awaitable, Callable
from typing import Any

Middleware = Callable[[str, dict[str, Any], Callable[[], Awaitable[Any]]], Awaitable[Any]]

