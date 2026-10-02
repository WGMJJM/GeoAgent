"""标准 JSON Schema 校验；保留原始定义，不按业务工具重写参数类型。"""

from __future__ import annotations

from typing import Any

from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for
from referencing import Registry
from referencing.exceptions import Unresolvable


def validate_arguments(value: Any, schema: dict[str, Any], path: str = "参数") -> str | None:
    try:
        validator_type = validator_for(schema)
        validator_type.check_schema(schema)
        # 支持本地 $defs/$ref，不通过远程 Schema 引用发起隐式外部访问。
        validator = validator_type(schema, registry=Registry())
        problem = next(validator.iter_errors(value), None)
    except (SchemaError, Unresolvable) as exc:
        return f"{path} Schema 无效或包含不可解析的引用：{type(exc).__name__}。"
    if problem is None:
        return None
    location = path + "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in problem.path)
    return f"{location}不符合工具 Schema：{problem.message}"


__all__ = ["validate_arguments"]
