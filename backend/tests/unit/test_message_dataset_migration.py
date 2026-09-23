import sqlite3

from app.core.models import Message
from app.state import StateStore


def test_message_dataset_ids_column_is_added_without_losing_legacy_messages(tmp_path):
    database = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database) as db:
        db.executescript(
            """
            CREATE TABLE conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                user_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                run_id TEXT,
                created_at TEXT NOT NULL
            );
            INSERT INTO conversations VALUES ('conv_legacy', '旧对话', NULL, '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00');
            INSERT INTO messages VALUES ('msg_legacy', 'conv_legacy', 'user', '旧消息', NULL, '2026-01-01T00:00:00+00:00');
            """
        )

    store = StateStore(database)
    store.initialize()
    store.save_message(Message(id="msg_new", conversation_id="conv_legacy", role="user", content="新消息", dataset_ids=["roads"]))

    messages = store.list_messages("conv_legacy")
    assert messages[0].dataset_ids == []
    assert messages[1].dataset_ids == ["roads"]
