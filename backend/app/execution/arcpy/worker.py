"""运行在 ArcGIS Pro Python 环境中的 JSON Lines Worker。"""

from __future__ import annotations

import json
import sys
import traceback


def _parameter(parameter):
    datatype = parameter.datatype
    if isinstance(datatype, (list, tuple)):
        datatype = list(datatype)
    return {
        "name": parameter.name,
        "display_name": parameter.displayName,
        "direction": parameter.direction,
        "datatype": datatype,
        "parameter_type": parameter.parameterType,
        "multi_value": parameter.multiValue,
        "enabled": parameter.enabled,
        "dependencies": list(getattr(parameter, "parameterDependencies", []) or []),
        "filter_type": getattr(parameter.filter, "type", None),
        "filter_list": list(getattr(parameter.filter, "list", []) or []),
    }


def _decode(value, arcpy):
    if isinstance(value, dict) and value.get("__arcpy__") == "spatial_reference":
        reference = str(value.get("value") or "").strip()
        if reference.upper().startswith("EPSG:"):
            return arcpy.SpatialReference(int(reference.split(":", 1)[1]))
        if reference.isdigit():
            return arcpy.SpatialReference(int(reference))
        return arcpy.SpatialReference(reference)
    return value


def _dispatch(request, arcpy):
    action = request["action"]
    payload = request.get("payload") or {}
    install = arcpy.GetInstallInfo()
    version = str(install.get("Version") or install.get("RealVersion") or "unknown")
    if action == "ping":
        return {"version": version, "product": install.get("ProductName", "ArcGIS Pro")}
    if action == "catalog":
        return {"version": version, "tools": arcpy.ListTools()}
    if action == "describe":
        name = str(payload["name"])
        return {
            "name": name,
            "version": version,
            "usage": arcpy.Usage(name),
            "parameters": [_parameter(item) for item in arcpy.GetParameterInfo(name)],
        }
    if action == "execute":
        name = str(payload["name"])
        values = [_decode(item, arcpy) for item in payload.get("values", [])]
        arcpy.env.overwriteOutput = False
        arcpy.env.addOutputsToMap = False
        result = getattr(arcpy.gp, name)(*values)
        output_count = int(getattr(result, "outputCount", 0) or 0)
        outputs = [str(result.getOutput(index)) for index in range(output_count)]
        messages = str(result.getMessages()) if hasattr(result, "getMessages") else ""
        return {"outputs": outputs, "messages": messages}
    raise ValueError(f"未知 ArcPy Worker action：{action}")


def main():
    import arcpy

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        response = {"id": request.get("id")}
        try:
            response.update({"ok": True, "result": _dispatch(request, arcpy)})
        except Exception as exc:
            response.update(
                {
                    "ok": False,
                    "error": str(exc) or exc.__class__.__name__,
                    "error_type": exc.__class__.__name__,
                    "traceback": traceback.format_exc(limit=8),
                }
            )
        print(json.dumps(response, ensure_ascii=True, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
