import asyncio

from app.core.models import AgentRequest
from app.entry.conversation_service import ConversationService
from app.memory import ConversationMemoryService
from app.state import StateStore


class _InactiveRunManager:
    def __init__(self) -> None:
        self.cancelled_run_ids: list[str] = []

    async def cancel(self, run_id: str) -> bool:
        self.cancelled_run_ids.append(run_id)
        return True


def test_user_message_persists_deduplicated_dataset_and_attachment_ids(tmp_path):
    store = StateStore(tmp_path / "state.sqlite3")
    store.initialize()
    service = ConversationService(store, _InactiveRunManager(), memory=ConversationMemoryService(store))

    asyncio.run(service._save_user_message(
        AgentRequest(
            user_input="检查道路",
            conversation_id="conversation-dataset-ids",
            dataset_ids=["roads", "roads"],
            attachment_ids=["dem", "roads"],
        )
    ))

    assert store.list_messages("conversation-dataset-ids")[0].dataset_ids == ["roads", "dem"]
