"""ArcPy 轻量目录、按需 Schema 物化和本地执行。"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from threading import Event
from typing import Any
from uuid import uuid4

from app.core.models import ErrorCategory, ToolMetadata, new_id
from app.execution.tools.model import RegisteredTool, ToolContext
from app.execution.tools.provider import ToolProviderError
from app.gis.errors import GISFailure

from .schema import (
    ArcPyToolDefinition,
    ArcPyToolSpec,
    UnsupportedArcPyTool,
    build_tool_definition,
    catalog_description,
    public_tool_name,
)


class ArcPyWorkerError(ToolProviderError):
    pass


class ArcPyWorker:
    """单进程串行 ArcPy Worker；超时或取消时丢弃进程，后续请求再惰性重启。"""

    def __init__(self, executable: str | Path, *, timeout_seconds: int = 120) -> None:
        self.executable = Path(executable).expanduser().resolve()
        self.timeout_seconds = timeout_seconds
        self.worker_script = Path(__file__).with_name("worker.py").resolve()
        self._process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._stderr: list[str] = []
        self._lock = threading.RLock()

    def request(
        self,
        action: str,
        payload: dict[str, Any] | None = None,
        *,
        cancel_event: Event | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._ensure_started()
            assert self._process is not None and self._process.stdin is not None
            request_id = uuid4().hex
            message = {"id": request_id, "action": action, "payload": payload or {}}
            try:
                self._process.stdin.write(json.dumps(message, ensure_ascii=True, separators=(",", ":")) + "\n")
                self._process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                details = self._worker_details()
                self._stop()
                raise ArcPyWorkerError(f"ArcPy Worker 无法接收请求。{details}") from exc

            deadline = time.monotonic() + (timeout_seconds or self.timeout_seconds)
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    self._stop()
                    raise GISFailure("ARCPY_CANCELLED", "ArcPy 工具执行已取消。", category=ErrorCategory.EXECUTION)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._stop()
                    raise TimeoutError(f"ArcPy Worker 超过 {timeout_seconds or self.timeout_seconds}s 未返回。")
                try:
                    response = self._responses.get(timeout=min(0.1, remaining))
                except queue.Empty:
                    continue
                if response is None:
                    details = self._worker_details()
                    self._stop()
                    raise ArcPyWorkerError(f"ArcPy Worker 已退出。{details}")
                if response.get("id") != request_id:
                    continue
                if not response.get("ok"):
                    raise ArcPyWorkerError(str(response.get("error") or "ArcPy 执行失败"))
                result = response.get("result")
                return result if isinstance(result, dict) else {"value": result}

    def close(self) -> None:
        with self._lock:
            self._stop()

    def _ensure_started(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        if not self.executable.is_file():
            raise ArcPyWorkerError(f"找不到 ArcGIS Pro Python 启动器：{self.executable}")
        self._responses = queue.Queue()
        self._stderr = []
        command = self._command()
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=os.environ.copy(),
            shell=False,
        )
        assert self._process.stdout is not None and self._process.stderr is not None
        threading.Thread(target=self._read_stdout, args=(self._process,), daemon=True).start()
        threading.Thread(target=self._read_stderr, args=(self._process,), daemon=True).start()

    def _command(self) -> list[str]:
        if self.executable.suffix.casefold() in {".bat", ".cmd"}:
            command_line = subprocess.list2cmdline([str(self.executable), str(self.worker_script)])
            return [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/s", "/c", command_line]
        return [str(self.executable), str(self.worker_script)]

    def _read_stdout(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                self._responses.put(value)
        self._responses.put(None)

    def _read_stderr(self, process: subprocess.Popen[str]) -> None:
        assert process.stderr is not None
        for line in process.stderr:
            self._stderr.append(line.strip())
            if len(self._stderr) > 20:
                del self._stderr[0]

    def _worker_details(self) -> str:
        return f" Worker 日志：{' | '.join(self._stderr)}" if self._stderr else ""

    def _stop(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


class ArcPyProvider:
    """只把名称级目录常驻内存，命中时才读取并缓存完整参数。"""

    def __init__(
        self,
        executable: str | Path,
        cache_dir: str | Path,
        *,
        timeout_seconds: int = 120,
        worker: ArcPyWorker | None = None,
    ) -> None:
        self.executable = Path(executable).expanduser().resolve()
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.worker = worker or ArcPyWorker(self.executable, timeout_seconds=timeout_seconds)
        self.timeout_seconds = timeout_seconds
        self._version: str | None = None
        self._summaries: dict[str, ToolMetadata] | None = None
        self._actual_names: dict[str, str] = {}
        self._materialized: dict[str, RegisteredTool] = {}
        self._unsupported: set[str] = set()
        self._lock = threading.RLock()

    @property
    def available(self) -> bool:
        return self.executable.is_file()

    def summaries(self) -> tuple[ToolMetadata, ...]:
        with self._lock:
            self._ensure_catalog()
            assert self._summaries is not None
            return tuple(self._summaries.values())

    def materialize(self, name: str) -> RegisteredTool:
        with self._lock:
            if name in self._materialized:
                return self._materialized[name]
            if name in self._unsupported:
                raise UnsupportedArcPyTool(f"ArcPy 工具当前不支持：{name}")
            self._ensure_catalog()
            actual_name = self._actual_names.get(name)
            if actual_name is None:
                raise KeyError(name)
            raw = self._load_tool_cache(name)
            if raw is None:
                raw = self.worker.request("describe", {"name": actual_name}, timeout_seconds=self.timeout_seconds)
                self._write_tool_cache(name, raw)
            spec = ArcPyToolSpec.from_dict(raw, public_name=name)
            try:
                definition = build_tool_definition(spec)
            except UnsupportedArcPyTool:
                self._unsupported.add(name)
                raise

            def handler(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
                return self._execute(spec, definition, arguments, context)

            registered = RegisteredTool(definition.metadata, handler)
            self._materialized[name] = registered
            return registered

    def close(self) -> None:
        self.worker.close()

    def _ensure_catalog(self) -> None:
        if self._summaries is not None:
            return
        if not self.available:
            raise ArcPyWorkerError(f"找不到 ArcGIS Pro Python 启动器：{self.executable}")
        identity = self.worker.request("ping", timeout_seconds=self.timeout_seconds)
        self._version = str(identity.get("version") or "unknown")
        cached = self._read_json(self.cache_dir / "catalog.json")
        if not isinstance(cached, dict) or cached.get("version") != self._version or not isinstance(cached.get("tools"), list):
            cached = self.worker.request("catalog", timeout_seconds=self.timeout_seconds)
            self._write_json(self.cache_dir / "catalog.json", cached)
        names = sorted({str(item) for item in cached["tools"] if isinstance(item, str) and item.strip()})
        summaries: dict[str, ToolMetadata] = {}
        actual_names: dict[str, str] = {}
        for actual_name in names:
            public_name = public_tool_name(actual_name)
            if public_name in summaries:
                continue
            summaries[public_name] = ToolMetadata(
                name=public_name,
                description=catalog_description(actual_name),
                input_schema={"type": "object", "properties": {}, "additionalProperties": False},
                required_scopes=["dataset.read"],
                required_envs=["gis.arcpy", "workspace"],
                risk_level="WRITE",
                tags=["gis", "arcpy"],
            )
            actual_names[public_name] = actual_name
        self._summaries = summaries
        self._actual_names = actual_names

    def _execute(
        self,
        spec: ArcPyToolSpec,
        definition: ArcPyToolDefinition,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> dict[str, Any]:
        values: list[Any] = []
        generated: dict[str, Path] = {}
        source_ids: list[str] = []
        workspace = context.services["workspace"]
        registry = context.services["registry"]

        for parameter in spec.parameters:
            if parameter.name in definition.generated_outputs:
                suffix = definition.generated_outputs[parameter.name]
                filename = f"{spec.actual_name.rsplit('_', 1)[0].casefold()}_{new_id('out').split('_', 1)[1]}{suffix}"
                path = workspace.output_path(filename, intermediate=True)
                generated[parameter.name] = path
                values.append(str(path))
                continue
            kind = definition.input_kinds.get(parameter.name)
            if kind is None or parameter.name not in arguments:
                values.append(None)
                continue
            value = arguments[parameter.name]
            if kind == "dataset":
                value, ids = self._dataset_value(value, registry, context)
                source_ids.extend(ids)
            elif kind == "spatial_reference":
                value = {"__arcpy__": "spatial_reference", "value": value}
            elif parameter.multi_value and isinstance(value, list):
                value = ";".join(str(item) for item in value)
            values.append(value)
        while values and values[-1] is None:
            values.pop()

        try:
            result = self.worker.request(
                "execute",
                {"name": spec.actual_name, "values": values},
                cancel_event=context.cancel_event,
                timeout_seconds=self.timeout_seconds,
            )
        except ArcPyWorkerError as exc:
            raise GISFailure("ARCPY_EXECUTION_FAILED", str(exc), category=ErrorCategory.EXECUTION) from exc

        dataset_ids: list[str] = []
        registered_outputs: list[dict[str, Any]] = []
        for parameter_name, path in generated.items():
            if not path.exists():
                continue
            registered = registry.register_path(
                path,
                name=path.stem,
                run_id=context.run_id,
                source_dataset_ids=list(dict.fromkeys(source_ids)),
                operation=spec.public_name,
                parameters=arguments,
                tool_call_id=context.call_id,
            )
            dataset_ids.append(registered.id)
            registered_outputs.append(
                {"parameter": parameter_name, "dataset_id": registered.id, "path": registered.path}
            )
        messages = str(result.get("messages") or "")
        warnings = []
        if "\ufffd" in messages:
            messages = ""
            warnings.append("ArcPy 已完成执行，但本机 ArcGIS Pro 的本地化消息无法正确解码。")
        return {
            "output": {
                "tool": spec.public_name,
                "arcgis_tool": spec.actual_name,
                "outputs": registered_outputs,
                "raw_outputs": result.get("outputs", []),
                "messages": messages,
            },
            "datasets": dataset_ids,
            "warnings": warnings,
        }

    @staticmethod
    def _dataset_value(value: Any, registry: Any, context: ToolContext) -> tuple[Any, list[str]]:
        values = value if isinstance(value, list) else [value]
        paths: list[str] = []
        identifiers: list[str] = []
        for identifier in values:
            dataset = registry.resolve(str(identifier), user_id=context.services.get("user_id"))
            if dataset is None:
                raise GISFailure("MISSING_DATASET", f"未注册的数据集：{identifier}", category=ErrorCategory.DATA)
            paths.append(str(Path(dataset.path).expanduser().resolve()))
            identifiers.append(dataset.id)
        return (";".join(paths) if isinstance(value, list) else paths[0]), identifiers

    def _load_tool_cache(self, public_name: str) -> dict[str, Any] | None:
        value = self._read_json(self._tool_cache_path(public_name))
        if isinstance(value, dict) and value.get("version") == self._version:
            return value
        return None

    def _write_tool_cache(self, public_name: str, value: dict[str, Any]) -> None:
        self._write_json(self._tool_cache_path(public_name), value)

    def _tool_cache_path(self, public_name: str) -> Path:
        return self.cache_dir / "tools" / f"{public_name.removeprefix('arcpy.')}.json"

    @staticmethod
    def _read_json(path: Path) -> Any:
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        temporary.replace(path)


def discover_arcpy_executable(configured: str | Path | None = None) -> Path | None:
    if configured:
        candidate = Path(configured).expanduser().resolve()
        return candidate if candidate.is_file() else None
    if sys.platform != "win32":
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\ESRI\ArcGISPro") as key:
            install_dir, _ = winreg.QueryValueEx(key, "InstallDir")
    except OSError:
        return None
    candidate = Path(install_dir) / "bin" / "Python" / "Scripts" / "propy.bat"
    return candidate.resolve() if candidate.is_file() else None


__all__ = ["ArcPyProvider", "ArcPyWorker", "ArcPyWorkerError", "discover_arcpy_executable"]
