from concurrent.futures import ThreadPoolExecutor

from app.core.models import (
    Run,
    RunStatus,
    TokenUsage,
)
from app.state import StateStore


def test_token_usage_is_atomic_across_children_and_preserved_by_status_writes(tmp_path):
    store = StateStore(tmp_path / "usage.sqlite3")
    store.initialize()
    parent = Run(agent_id="main")
    store.save_run(parent)
    children = [Run(agent_id="child", parent_run_id=parent.id) for _ in range(4)]
    for child in children:
        store.save_run(child)
    delta = TokenUsage(local_input_tokens=20, local_output_tokens=3, reported_input_tokens=24,
                       reported_output_tokens=5, model_calls=1, reported_calls=1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        updates = list(pool.map(lambda index: store.add_run_token_usage(children[index % 4].id, delta), range(16)))
    assert all([run.id for run in update] == [children[index % 4].id, parent.id] for index, update in enumerate(updates))
    assert store.get_run(parent.id).token_usage == TokenUsage(local_input_tokens=320, local_output_tokens=48,
                                                            reported_input_tokens=384, reported_output_tokens=80,
                                                            model_calls=16, reported_calls=16)
    for child in children:
        assert store.get_run(child.id).token_usage.model_calls == 4
    # 一个执行中持有旧快照的状态写入不能将后续子任务计数覆盖掉。
    store.save_run(parent.model_copy(update={"status": RunStatus.COMPLETED}))
    assert store.get_run(parent.id).status is RunStatus.COMPLETED
    assert store.get_run(parent.id).token_usage.model_calls == 16
    untouched = Run(agent_id="old")
    store.save_run(untouched)
    assert store.get_run(untouched.id).token_usage is None
