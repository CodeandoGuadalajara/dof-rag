"""Conversation persistence; runs remain execution jobs, not chat history."""
import json
import uuid


def initialize(connection):
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS chat_conversations (
            conversation_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chat_messages (
            conversation_id TEXT NOT NULL REFERENCES chat_conversations(conversation_id),
            run_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            created_at TEXT NOT NULL,
            content TEXT NOT NULL DEFAULT '',
            result_json TEXT,
            PRIMARY KEY (run_id, role)
        );
        CREATE INDEX IF NOT EXISTS chat_messages_conversation ON chat_messages(conversation_id, created_at);
    """)
    # Migrate the first chat version without losing existing messages.
    rows = connection.execute(
        "SELECT r.run_id, r.user_id, r.created_at, r.question, e.event_type, e.payload_json "
        "FROM runs r JOIN token_reservations t ON t.run_id = r.run_id "
        "JOIN run_events e ON e.run_id = r.run_id AND e.sequence = "
        "(SELECT MAX(sequence) FROM run_events WHERE run_id = r.run_id)"
    ).fetchall()
    for row in rows:
        add_message(connection, row[0], row[1], row[2], row[3])
        if row[4] in {'succeeded', 'failed'}:
            finish_message(connection, row[0], row[4], json.loads(row[5]))
    connection.execute("UPDATE runs SET published_at = NULL, published_by = NULL WHERE run_id IN (SELECT run_id FROM chat_messages)")


def add_message(connection, run_id, user_id, created_at, question):
    connection.execute(
        "INSERT OR IGNORE INTO chat_conversations VALUES (?, ?, ?)",
        (str(uuid.uuid4()), user_id, created_at),
    )
    conversation_id = connection.execute(
        "SELECT conversation_id FROM chat_conversations WHERE user_id = ?", (user_id,)
    ).fetchone()[0]
    for role, text in [('user', question), ('assistant', '')]:
        connection.execute(
            "INSERT OR IGNORE INTO chat_messages(conversation_id, run_id, role, created_at, content) VALUES (?, ?, ?, ?, ?)",
            (conversation_id, run_id, role, created_at, text),
        )


def finish_message(connection, run_id, state, payload):
    text = payload.get('answer', {}).get('text', '') if state == 'succeeded' else payload.get('message', '')
    connection.execute(
        "UPDATE chat_messages SET content = ?, result_json = ? WHERE run_id = ? AND role = 'assistant'",
        (text, json.dumps(payload, ensure_ascii=False), run_id),
    )
