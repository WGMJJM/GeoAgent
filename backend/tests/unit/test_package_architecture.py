import os
import re
from pathlib import Path

from app.config import Settings


def test_backend_uses_the_single_loop_framework_and_has_no_legacy_runtime():
    backend = Path(__file__).resolve().parents[2]
    app = backend / "app"

    assert (app / "agent" / "loop.py").is_file()
    assert (app / "agent" / "context.py").is_file()
    assert (app / "entry" / "gateway.py").is_file()
    assert (app / "run" / "manager.py").is_file()
    assert (app / "run" / "checkpoints.py").is_file()

    for package in ("decision", "understanding", "planning", "runtime", "task", "knowledge", "context"):
        path = app / package
        assert not path.exists() or not list(path.glob("*.py"))

    for file in ("main_agent.py", "manager.py", "scheduler.py", "sub_agent.py"):
        assert not (app / "agent" / file).exists()
    for file in ("message_entry.py", "message_router.py", "dataset_resolver.py"):
        assert not (app / "entry" / file).exists()


def test_example_environment_options_have_settings_fields():
    example = Path(__file__).resolve().parents[2] / ".env.example"
    # 同时检查启用项及注释中的可选项，而不是让 extra="ignore" 掩盖无效配置。
    declared = set(
        re.findall(
            r"^\s*(?:#\s*)?(GEOAGENT_[A-Z0-9_]+)=",
            example.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
    )
    prefix = Settings.model_config["env_prefix"]
    supported = {f"{prefix}{name.upper()}" for name in Settings.model_fields}
    assert declared
    assert not (declared - supported), f"示例包含无效 Settings 选项：{sorted(declared - supported)}"


def test_example_environment_loads_without_local_overrides(monkeypatch):
    for key in tuple(os.environ):
        if key.startswith(Settings.model_config["env_prefix"]):
            monkeypatch.delenv(key)
    example = Path(__file__).resolve().parents[2] / ".env.example"
    settings = Settings(_env_file=example)
    assert settings.workspace == Path("../workspace")
    assert settings.max_agent_turns == 20
    assert settings.max_tool_calls == 40
    assert settings.max_subagents == 5
    assert settings.max_parallel_agents == 3
    assert settings.max_tokens == 3200
    assert settings.protocol_history_tokens == 25600
    assert settings.tool_result_compaction_ratio == 0.2
    assert settings.tool_context_tokens == 3200
    assert settings.tool_context_max_cards == 8
