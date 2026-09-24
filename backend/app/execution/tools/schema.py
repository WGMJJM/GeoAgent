"""工具参数的最小 JSON Schema 校验器。"""

from __future__ import annotations

from typing import Any


def validate_arguments(value: Any, schema: dict[str, Any], path: str = "参数") -> str | None:
    expected = schema.get("type")
    valid = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "null": lambda item: item is None,
    }
    if expected in valid and not valid[expected](value):
        return f"{path}类型错误，期望 {expected}。"
    if "enum" in schema and value not in schema["enum"]:
        return f"{path}不在允许值范围内。"
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        missing = [key for key in schema.get("required", []) if key not in value]
        if missing:
            return f"{path}缺少必需字段：{', '.join(missing)}。"
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(properties)
            if extra:
                return f"{path}包含未声明字段：{', '.join(sorted(extra))}。"
        for key, item in value.items():
            child_schema = properties.get(key)
            if isinstance(child_schema, dict):
                problem = validate_arguments(item, child_schema, f"{path}.{key}")
                if problem:
                    return problem
    if isinstance(value, str) and len(value) < schema.get("minLength", 0):
        return f"{path}不能为空。"
    if isinstance(value, str) and "maxLength" in schema and len(value) > schema["maxLength"]:
        return f"{path}超过允许的最大长度。"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return f"{path}小于允许的最小值。"
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            return f"{path}必须大于 {schema['exclusiveMinimum']}。"
        if "maximum" in schema and value > schema["maximum"]:
            return f"{path}超过允许的最大值。"
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            return f"{path}必须小于 {schema['exclusiveMaximum']}。"
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        if len(value) < schema.get("minItems", 0):
            return f"{path}少于允许的最少项目数。"
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            return f"{path}超过允许的最多项目数。"
        for index, item in enumerate(value):
            problem = validate_arguments(item, schema["items"], f"{path}[{index}]")
            if problem:
                return problem
    return None


__all__ = ["validate_arguments"]
