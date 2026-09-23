import sqlite3
from datetime import UTC, datetime, timedelta

from app.core.models import (
    AgentResult,
    AgentResultStatus,
    ApprovalRequest,
    ApprovalStatus,
    Artifact,
    ArtifactKind,
    Dataset,
    DatasetKind,
    Message,
    RiskLevel,
    Run,
    RunStatus,
    Task,
)
from app.run.lifecycle import persist_result, record_approval_denied
from app.state import StateStore


def test_recent_messages_returns_latest_items_in_chronological_order(tmp_path):
    store = StateStore(tmp_path / "state.sqlite3")
    store.initialize()
    store.upsert_conversation("conv_recent", "最近消息", datetime.now(UTC).isoformat())
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for index in range(30):
        store.save_message(
            Message(
                id=f"msg_{index:02d}",
                conversation_id="conv_recent",
                role="user",
                content=f"消息 {index}",
                created_at=start + timedelta(minutes=index),
            )
        )

    messages = store.list_messages("conv_recent", limit=8)

    assert [item.content for item in messages] == [f"消息 {index}" for index in range(22, 30)]


def test_recent_runs_are_filtered_by_conversation_before_limit(tmp_path):
    store = StateStore(tmp_path / "state.sqlite3")
    store.initialize()
    store.upsert_conversation("conv_a", "A", datetime.now(UTC).isoformat())
    store.upsert_conversation("conv_b", "B", datetime.now(UTC).isoformat())
    store.save_run(Run(id="run_a_old", conversation_id="conv_a", agent_id="main", status=RunStatus.COMPLETED))
    for index in range(5):
        store.save_run(Run(id=f"run_b_{index}", conversation_id="conv_b", agent_id="main", status=RunStatus.COMPLETED))

    runs = store.list_runs_for_conversation("conv_a", limit=1)

    assert [item.id for item in runs] == ["run_a_old"]


def test_state_store_migrates_legacy_core_payload_columns(tmp_path):
    database = tmp_path / "legacy.sqlite3"
    task = Task(id="task_legacy", conversation_id="conv_legacy", goal="旧任务")
    run = Run(id="run_legacy", conversation_id="conv_legacy", task_id=task.id, agent_id="main", status=RunStatus.FAILED)
    dataset = Dataset(id="ds_legacy", name="旧数据", kind=DatasetKind.VECTOR, path="old.geojson", format="geojson", owner_user_id="user-1", created_by_run_id=run.id)
    artifact = Artifact(id="art_legacy", name="旧产物", kind=ArtifactKind.OTHER, owner_user_id="user-1", run_id=run.id)
    with sqlite3.connect(database) as db:
        db.executescript(
            """
            CREATE TABLE conversations (id TEXT PRIMARY KEY, title TEXT NOT NULL, user_id TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE tasks (id TEXT PRIMARY KEY, payload_json TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE runs (id TEXT PRIMARY KEY, payload_json TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE datasets (id TEXT PRIMARY KEY, payload_json TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE artifacts (id TEXT PRIMARY KEY, payload_json TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE tool_calls (id TEXT PRIMARY KEY, run_id TEXT, name TEXT NOT NULL, arguments_json TEXT NOT NULL, result_json TEXT, created_at TEXT NOT NULL);
            INSERT INTO conversations VALUES ('conv_legacy', '旧会话', 'user-1', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00');
            """
        )
        db.execute("INSERT INTO tasks VALUES (?,?,?)", (task.id, task.model_dump_json(), task.updated_at.isoformat()))
        db.execute("INSERT INTO runs VALUES (?,?,?)", (run.id, run.model_dump_json(), datetime.now(UTC).isoformat()))
        db.execute("INSERT INTO datasets VALUES (?,?,?)", (dataset.id, dataset.model_dump_json(), dataset.created_at.isoformat()))
        db.execute("INSERT INTO artifacts VALUES (?,?,?)", (artifact.id, artifact.model_dump_json(), artifact.created_at.isoformat()))

    store = StateStore(database)
    store.initialize()

    assert store.get_task(task.id) == task
    assert store.get_run(run.id) == run
    assert store.get_dataset_for_user(dataset.id, "user-1") == dataset
    assert store.get_artifact_for_user(artifact.id, "user-1") == artifact
    with store._connect() as db:
        assert {row[1] for row in db.execute("PRAGMA table_info(runs)").fetchall()} >= {"conversation_id", "status", "metadata_json"}
        assert {row[1] for row in db.execute("PRAGMA table_info(tasks)").fetchall()} >= {"conversation_id", "goal", "status"}


def test_delete_run_preserves_dataset_and_artifact_and_clears_provenance(tmp_path):
    store = StateStore(tmp_path / "state.sqlite3")
    store.initialize()
    run = Run(id="run_provenance", agent_id="main", status=RunStatus.COMPLETED)
    dataset = Dataset(id="ds_provenance", name="结果", kind=DatasetKind.VECTOR, path="result.geojson", format="geojson", created_by_run_id=run.id)
    artifact = Artifact(id="art_provenance", name="结果文件", kind=ArtifactKind.OTHER, run_id=run.id)
    store.save_run(run)
    store.save_dataset(dataset)
    store.save_artifact(artifact)

    assert store.delete_run(run.id) is True
    assert store.get_dataset(dataset.id).created_by_run_id is None
    assert store.get_artifact(artifact.id).run_id is None


def test_run_and_task_result_transition_is_atomic(tmp_path):
    store = StateStore(tmp_path / "state.sqlite3")
    store.initialize()
    task = Task(id="task_transition", conversation_id="conv_transition", goal="状态转换")
    run = Run(id="run_transition", conversation_id=task.conversation_id, task_id=task.id, agent_id="main", status=RunStatus.RUNNING)
    store.save_task(task)
    store.save_run(run)

    updated_run, updated_task = persist_result(
        store,
        run,
        task,
        AgentResult(agent_id="main", task_id=task.id, status=AgentResultStatus.SUCCESS, summary="完成", trace_id=run.id),
    )

    assert updated_run.status is RunStatus.COMPLETED
    assert updated_task is not None and updated_task.status.value == "SUCCEEDED"
    assert store.get_run(run.id).status is RunStatus.COMPLETED
    assert store.get_task(task.id).status.value == "SUCCEEDED"


def test_approval_denial_persists_with_run_transition(tmp_path):
    store = StateStore(tmp_path / "state.sqlite3")
    store.initialize()
    task = Task(id="task_approval", conversation_id="conv_approval", goal="审批任务", status="WAITING")
    run = Run(id="run_approval", conversation_id=task.conversation_id, task_id=task.id, agent_id="main", status=RunStatus.WAITING_APPROVAL)
    approval = ApprovalRequest(
        id="approval_atomic",
        user_id="user-1",
        conversation_id=task.conversation_id,
        task_id=task.id,
        source_run_id=run.id,
        tool_call_id="call_approval",
        tool_name="test.write",
        argument_fingerprint="fingerprint",
        risk_level=RiskLevel.WRITE,
        reason="需要审批",
        status=ApprovalStatus.DENIED,
    )
    store.save_task(task)
    store.save_run(run)

    record_approval_denied(store, approval)

    assert store.get_approval(approval.id).status is ApprovalStatus.DENIED
    assert store.get_run(run.id).status is RunStatus.WAITING_USER
    assert store.get_task(task.id).status.value == "WAITING"
