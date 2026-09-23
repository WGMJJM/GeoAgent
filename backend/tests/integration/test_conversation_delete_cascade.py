import asyncio
import json

import pytest

from app.core.models import (
    ApprovalRequest,
    Artifact,
    ArtifactKind,
    Checkpoint,
    ConversationMemory,
    Dataset,
    DatasetKind,
    Message,
    RiskLevel,
    Run,
    RunStatus,
    SubTask,
    Task,
    ToolCall,
    TraceEvent,
    WorkingMemory,
    utc_now,
)


@pytest.mark.parametrize(
    "status",
    [
        RunStatus.COMPLETED,
        RunStatus.WAITING_USER,
        RunStatus.WAITING_APPROVAL,
        RunStatus.RUNNING,
        RunStatus.WAITING_TOOL,
        RunStatus.WAITING_SUBAGENT,
        RunStatus.RETRYING,
    ],
)
def test_delete_conversation_accepts_completed_and_all_waiting_or_execution_states(application, status):
    conversation = application.conversations.create("可删除", user_id="user-delete")
    application.store.save_run(Run(conversation_id=conversation.id, agent_id="main", status=status))

    assert asyncio.run(application.conversations.delete(conversation.id, user_id="user-delete")) is True
    assert application.store.get_conversation(conversation.id) is None
    assert application.store.list_runs_for_conversation(conversation.id) == []


def test_delete_conversation_cascades_records_and_preserves_user_resources(application, tmp_path):
    conversation = application.conversations.create("级联清理", user_id="user-delete")
    task = Task(id="task_delete", goal="处理道路", conversation_id=conversation.id)
    parent = Run(id="run_delete_parent", task_id=task.id, conversation_id=conversation.id, agent_id="main", status=RunStatus.RUNNING)
    child = Run(id="run_delete_child", parent_run_id=parent.id, task_id=task.id, conversation_id=conversation.id, agent_id="subagent", status=RunStatus.WAITING_SUBAGENT)
    root_path = tmp_path / "roads.geojson"
    artifact_path = tmp_path / "result.geojson"
    root_path.write_text("root", encoding="utf-8")
    artifact_path.write_text("derived", encoding="utf-8")
    root_dataset = Dataset(id="dataset_root", name="roads.geojson", kind=DatasetKind.VECTOR, path=str(root_path), format="GeoJSON", owner_user_id="user-delete")
    derived_dataset = Dataset(
        id="dataset_derived",
        name="roads_buffer.geojson",
        kind=DatasetKind.VECTOR,
        path=str(artifact_path),
        format="GeoJSON",
        source_dataset_ids=[root_dataset.id],
        created_by_run_id=parent.id,
        owner_user_id="user-delete",
    )
    artifact = Artifact(
        id="artifact_delete",
        name="roads_buffer.geojson",
        kind=ArtifactKind.DATASET,
        path=str(artifact_path),
        dataset_id=derived_dataset.id,
        run_id=parent.id,
        owner_user_id="user-delete",
    )

    application.store.save_task(task)
    application.store.save_subtask(task.id, SubTask(id="subtask_delete", goal="生成缓冲区", description="测试子任务"))
    application.store.save_run(parent)
    application.store.save_run(child)
    application.store.save_message(Message(id="message_delete", conversation_id=conversation.id, role="user", content="删除这个对话", dataset_ids=[root_dataset.id]))
    application.store.save_working_memory(WorkingMemory(task_id=task.id, conversation_id=conversation.id, active_dataset_ids=[root_dataset.id]))
    application.store.save_conversation_memory(ConversationMemory(conversation_id=conversation.id, user_id="user-delete", summary="待删除摘要"))
    now = utc_now().isoformat()
    with application.store._connect() as db:
        db.execute(
            "INSERT INTO planning_sessions(id, user_id, conversation_id, payload_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("plan_delete", "user-delete", conversation.id, json.dumps({"goal": "道路缓冲区"}), now, now),
        )
        db.commit()
    application.store.save_checkpoint(Checkpoint(id="checkpoint_delete", run_id=parent.id, phase="tool"))
    application.store.save_tool_call(ToolCall(id="call_delete", run_id=parent.id, name="vector.buffer"))
    application.store.record_event(TraceEvent(id="event_delete", run_id=parent.id, event_type="ToolStarted", message="开始"))
    application.store.save_lineage(
        lineage_id="lineage_delete",
        run_id=parent.id,
        operation="buffer",
        input_dataset_ids=[root_dataset.id],
        output_dataset_id=derived_dataset.id,
        tool_call_id="call_delete",
        parameters={"distance": 500},
        created_at=utc_now().isoformat(),
    )
    application.store.save_approval(
        ApprovalRequest(
            id="approval_delete",
            user_id="user-delete",
            conversation_id=conversation.id,
            task_id=task.id,
            source_run_id=parent.id,
            tool_call_id="call_delete",
            tool_name="vector.buffer",
            argument_fingerprint="delete-test",
            risk_level=RiskLevel.WRITE,
            reason="测试清理",
        )
    )
    application.store.save_dataset(root_dataset)
    application.store.save_dataset(derived_dataset)
    application.store.save_artifact(artifact)

    assert asyncio.run(application.conversations.delete(conversation.id, user_id="user-delete")) is True

    assert application.store.get_conversation(conversation.id) is None
    assert application.store.list_messages(conversation.id) == []
    assert application.store.list_runs_for_conversation(conversation.id) == []
    assert application.store.get_task(task.id) is None
    assert application.store.get_subtask(task.id, "subtask_delete") is None
    assert application.store.get_working_memory(task.id) is None
    assert application.store.get_conversation_memory_for_user(conversation.id, "user-delete") is None
    assert application.store.list_approvals("user-delete") == []

    with application.store._connect() as db:
        for table in ("trace_events", "checkpoints", "tool_calls", "dataset_lineage", "approvals", "planning_sessions", "subtasks", "tasks", "working_memories", "messages", "runs", "conversation_memories"):
            assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0

    assert application.store.get_dataset(root_dataset.id) == root_dataset
    preserved_dataset = application.store.get_dataset(derived_dataset.id)
    assert preserved_dataset is not None and preserved_dataset.created_by_run_id is None
    preserved_artifact = application.store.get_artifact(artifact.id)
    assert preserved_artifact is not None and preserved_artifact.run_id is None
    assert root_path.exists()
    assert artifact_path.exists()
