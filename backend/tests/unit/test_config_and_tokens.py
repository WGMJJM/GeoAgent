import json
import os
import re
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from app.agent.skills import SkillCatalog, SkillStreamBuffer, parse_skill_request
from app.application import Application
from app.config import Settings
from app.core.tokens import DEFAULT_TOKENIZER_FILE
from app.models.config import ModelProfile
from app.models.providers.openai_compatible import OpenAICompatibleAdapter

DEEPSEEK_TOKENIZER_FILE = Path(__file__).resolve().parents[2] / "resources/tokenizers/deepseek-v4.1.json"


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
    assert settings.skills_directory == Path("backend/skills")
    assert settings.max_agent_turns == 20
    assert settings.max_tool_calls == 40
    assert settings.max_subagents == 5
    assert settings.max_parallel_agents == 3
    assert settings.max_tokens == 12800
    assert settings.model_input_tokens == 128000
    assert settings.summary_recent_messages == 16
    assert settings.summary_trigger_messages == 24
    assert settings.summary_trigger_tokens == 51200
    assert settings.summary_message_max_chars == 10000
    assert settings.emergency_recent_messages == 8
    assert settings.tool_result_recent_full == 16
    assert settings.conversation_tool_index_limit == Settings.model_fields["conversation_tool_index_limit"].default == 8
    assert settings.tool_result_emergency_fraction == 0.5
    assert settings.tool_context_tokens == Settings.model_fields["tool_context_tokens"].default == 25600
    assert settings.tool_context_max_cards == 8
    assert settings.tool_search_regex_results == 2
    assert settings.tool_search_chinese_results == 1
    assert settings.tool_search_english_results == 3
    assert settings.enable_arcpy is True
    assert settings.arcpy_cache == Path("state/arcpy")


@pytest.mark.asyncio
async def test_additional_model_profiles_extend_existing_profiles(tmp_path):
    settings = Settings(
        root=tmp_path,
        database=tmp_path / "state.sqlite3",
        workspace=tmp_path / "workspace",
        enable_arcpy=False,
        model_profiles='[{"id":"primary","label":"主模型","model":"primary","default":true}]',
        additional_model_profiles=json.dumps([{
            "id": "deepseek-flash",
            "label": "DeepSeek Flash",
            "model": "deepseek-flash",
            "tokenizer_file": str(DEEPSEEK_TOKENIZER_FILE),
        }]),
        model_reasoning_config='{"deepseek-flash":{"reasoning_efforts":["low","medium","high","xhigh","max"],"default_reasoning_effort":"medium"}}',
    )
    application = Application(settings)
    try:
        assert list(application.model_profiles) == ["primary", "deepseek-flash"]
        assert application.default_model_profile == "primary"
        status = application.model_status()
        assert [item["id"] for item in status["profiles"]] == ["primary", "deepseek-flash"]
        assert status["profiles"][1]["has_api_key"] is False
        assert status["profiles"][1]["reasoning_efforts"] == ["low", "medium", "high", "xhigh", "max"]
        assert status["profiles"][1]["default_reasoning_effort"] == "medium"
        sample = "坡度分析 DEM 重投影 EPSG:32650"
        assert application.get_model_adapter("primary").count_tokens(sample) != application.get_model_adapter("deepseek-flash").count_tokens(sample)
    finally:
        await application.close()


def test_model_profile_tokenizer_override_and_global_default(tmp_path):
    path = tmp_path / "custom.json"
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.enable_truncation(max_length=1)
    tokenizer.enable_padding(length=10)
    tokenizer.save(str(path))
    profile = ModelProfile(id="local", label="本地模型", model="example")
    config = profile.as_config(tokenizer_file=path)
    adapter = object.__new__(OpenAICompatibleAdapter)
    adapter.config = config
    assert adapter.count_tokens("hello world") == 2  # 不沿用文件的 padding/truncation。
    assert profile.model_copy(update={"tokenizer_file": DEFAULT_TOKENIZER_FILE}).as_config(tokenizer_file=path).tokenizer_file == DEFAULT_TOKENIZER_FILE


def test_skill_catalog_discovers_metadata_and_reads_only_approved_files(tmp_path):
    directory = tmp_path / "skills"
    skill = directory / "report"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text('---\nname: report\ndescription: >\n  按需组织分析证据\n---\n正文不提前加载', encoding="utf-8")
    (skill / "reference.md").write_text("参考资料", encoding="utf-8")
    outside = tmp_path / "private.txt"
    outside.write_text("目录外内容", encoding="utf-8")
    catalog = SkillCatalog(directory)
    assert "正文不提前加载" not in catalog.prompt_message()["content"]
    assert catalog.read("report")["content"].endswith("正文不提前加载")
    assert catalog.read("report", "reference.md")["content"] == "参考资料"
    for path in ("../../private.txt", str(outside)):
        with pytest.raises(ValueError, match="相对文件路径"):
            catalog.read("report", path)
    with pytest.raises(KeyError):
        catalog.read("unknown")
    assert SkillCatalog(tmp_path / "empty").prompt_message() is None


def test_skill_symlink_cannot_escape_approved_directory(tmp_path):
    directory = tmp_path / "skills" / "report"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text("---\nname: report\ndescription: 结果指导\n---\n正文", encoding="utf-8")
    outside = tmp_path / "private.txt"
    outside.write_text("目录外内容", encoding="utf-8")
    try:
        (directory / "escape.md").symlink_to(outside)
    except OSError as exc:
        if exc.winerror == 1314:
            pytest.skip("当前 Windows 账户没有创建符号链接的权限")
        raise
    with pytest.raises(ValueError):
        SkillCatalog(directory.parent).read("report", "escape.md")


def test_skill_control_request_and_streaming_do_not_capture_normal_answers():
    content = '{"action":"read_skill","name":"report"}'
    buffer = SkillStreamBuffer()
    assert all(buffer.feed(character) == "" for character in content)
    assert parse_skill_request(content).name == "report"
    for value in ('{"action":"read_skill","name":false}', '{"action":"read_skill"', '{"action":"read_skill","name":"report","code":"x"}'):
        with pytest.raises(ValueError):
            parse_skill_request(value)
    assert parse_skill_request('{"answer":"普通 JSON"}') is None
    assert parse_skill_request("普通回答") is None
    text = SkillStreamBuffer()
    assert text.feed("你好") == "你好"
    assert text.feed("，可以直接回答。") == "，可以直接回答。"
    ordinary = SkillStreamBuffer()
    assert ordinary.feed('{"answer":"普通 JSON"}') == ""
    assert ordinary.flush() == '{"answer":"普通 JSON"}'
