from pathlib import Path


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
