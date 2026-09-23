import asyncio

from app.core.models import AgentRequest, Run, RunStatus
from app.entry.conversation_service import ConversationService, derive_conversation_title
from app.state import StateStore


class _InactiveRunManager:
    def __init__(self) -> None:
        self.cancelled_run_ids: list[str] = []

    async def cancel(self, run_id: str) -> bool:
        self.cancelled_run_ids.append(run_id)
        return True


def test_derive_conversation_title_normalizes_whitespace_and_limits_length():
    assert derive_conversation_title("  分析\n道路\t与人口  ") == "分析 道路 与人口"
    assert derive_conversation_title("这是一个超过标题长度限制的空间分析任务请求，需要继续处理") == "这是一个超过标题长度限制的空间分析任务请求，需要…"


def test_conversation_title_is_derived_only_from_first_user_message(tmp_path):
    store = StateStore(tmp_path / "state.sqlite3")
    store.initialize()
    service = ConversationService(store, _InactiveRunManager())
    service._save_user_message(AgentRequest(user_input="第一条消息：分析道路", conversation_id="conversation-title"))
    service._save_user_message(AgentRequest(user_input="第二条消息不应覆盖标题", conversation_id="conversation-title"))

    conversation = store.get_conversation("conversation-title")
    assert conversation is not None
    assert conversation.title == "第一条消息：分析道路"


def test_user_message_persists_deduplicated_dataset_and_attachment_ids(tmp_path):
    store = StateStore(tmp_path / "state.sqlite3")
    store.initialize()
    service = ConversationService(store, _InactiveRunManager())

    service._save_user_message(
        AgentRequest(
            user_input="检查道路",
            conversation_id="conversation-dataset-ids",
            dataset_ids=["roads", "roads"],
            attachment_ids=["dem", "roads"],
        )
    )

    assert store.list_messages("conversation-dataset-ids")[0].dataset_ids == ["roads", "dem"]


def test_conversation_delete_cancels_execution_run_before_removing_conversation(tmp_path):
    store = StateStore(tmp_path / "state.sqlite3")
    store.initialize()
    conversation = store.create_conversation("待删除")
    run = Run(conversation_id=conversation.id, task_id="task-delete-guard", agent_id="main", status=RunStatus.RUNNING)
    store.save_run(run)
    run_manager = _InactiveRunManager()

    assert asyncio.run(ConversationService(store, run_manager).delete(conversation.id)) is True
    assert run_manager.cancelled_run_ids == [run.id]
    assert store.get_conversation(conversation.id) is None


def test_conversation_delete_cancels_human_waiting_runs_and_removes_created_runs(tmp_path):
    store = StateStore(tmp_path / "state.sqlite3")
    store.initialize()
    conversation = store.create_conversation("等待中的对话")
    created = Run(conversation_id=conversation.id, task_id="task-created", agent_id="main", status=RunStatus.CREATED)
    waiting_user = Run(conversation_id=conversation.id, task_id="task-human-wait", agent_id="main", status=RunStatus.WAITING_USER)
    waiting_approval = Run(conversation_id=conversation.id, task_id="task-approval", agent_id="main", status=RunStatus.WAITING_APPROVAL)
    store.save_run(created)
    store.save_run(waiting_user)
    store.save_run(waiting_approval)
    run_manager = _InactiveRunManager()

    assert asyncio.run(ConversationService(store, run_manager).delete(conversation.id)) is True
    assert set(run_manager.cancelled_run_ids) == {waiting_user.id, waiting_approval.id}
    assert store.list_runs_for_conversation(conversation.id) == []
