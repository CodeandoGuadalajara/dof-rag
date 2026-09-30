"""Conversation persistence; runs remain execution jobs, not chat history."""
import json
import uuid


def initialize(connection):
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS chat_conversations (
            conversation_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
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
    unique_user = any(
        index[2] and [column[2] for column in connection.execute(f'PRAGMA index_info("{index[1]}")')] == ['user_id']
        for index in connection.execute('PRAGMA index_list(chat_conversations)')
    )
    if unique_user:
        connection.executescript("""
            BEGIN IMMEDIATE;
            CREATE TABLE chat_conversations_new (
                conversation_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE chat_messages_new (
                conversation_id TEXT NOT NULL REFERENCES chat_conversations_new(conversation_id),
                run_id TEXT NOT NULL, role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                created_at TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', result_json TEXT,
                PRIMARY KEY (run_id, role)
            );
            INSERT INTO chat_conversations_new SELECT * FROM chat_conversations;
            INSERT INTO chat_messages_new SELECT * FROM chat_messages;
            DROP TABLE chat_messages;
            DROP TABLE chat_conversations;
            ALTER TABLE chat_conversations_new RENAME TO chat_conversations;
            ALTER TABLE chat_messages_new RENAME TO chat_messages;
            CREATE INDEX chat_messages_conversation ON chat_messages(conversation_id, created_at);
            COMMIT;
        """)
    # Migrate the first chat version without losing existing messages.
    rows = connection.execute(
        "SELECT r.run_id, r.user_id, r.created_at, r.question, e.event_type, e.payload_json "
        "FROM runs r "
        "JOIN run_events e ON e.run_id = r.run_id AND e.sequence = "
        "(SELECT MAX(sequence) FROM run_events WHERE run_id = r.run_id) "
        "WHERE r.kind = 'chat' OR r.run_id IN (SELECT run_id FROM token_reservations)"
    ).fetchall()
    for row in rows:
        add_message(connection, row[0], row[1], row[2], row[3])
        if row[4] in {'succeeded', 'failed'}:
            finish_message(connection, row[0], row[4], json.loads(row[5]))
    connection.execute("UPDATE runs SET published_at = NULL, published_by = NULL WHERE kind = 'chat' OR run_id IN (SELECT run_id FROM chat_messages)")


def add_message(connection, run_id, user_id, created_at, question, conversation_id=None):
    if connection.execute('SELECT 1 FROM chat_messages WHERE run_id = ?', (run_id,)).fetchone():
        return
    if conversation_id is None:
        previous = connection.execute(
            'SELECT conversation_id FROM chat_conversations WHERE user_id = ? ORDER BY created_at LIMIT 1', (user_id,)
        ).fetchone()
        conversation_id = previous[0] if previous else str(uuid.uuid4())
    owner = connection.execute('SELECT user_id FROM chat_conversations WHERE conversation_id = ?', (conversation_id,)).fetchone()
    if owner is not None and owner[0] != user_id:
        raise KeyError(conversation_id)
    connection.execute(
        'INSERT OR IGNORE INTO chat_conversations VALUES (?, ?, ?)', (conversation_id, user_id, created_at)
    )
    for role, text in [('user', question), ('assistant', '')]:
        connection.execute(
            "INSERT OR IGNORE INTO chat_messages(conversation_id, run_id, role, created_at, content) VALUES (?, ?, ?, ?, ?)",
            (conversation_id, run_id, role, created_at, text),
        )


def partial_from_progress(events):
    """Recover completed and interrupted turns from persisted stream batches."""
    turns, tools = {}, {}
    for event in events:
        payload = event.get('payload', {})
        field = payload.get('chat_delta')
        if field in {'content', 'reasoning_content'}:
            number = payload.get('turn', 1)
            turn = turns.setdefault(number, {'turn': number, 'content': '', 'reasoning_content': ''})
            turn[field] += payload.get('text', '')
        elif event.get('event_type') in {'tool_started', 'tool_completed'}:
            call_id = payload.get('call_id', str(event.get('sequence', '')))
            tool = tools.setdefault(call_id, {'name': payload.get('tool', ''), 'arguments': None})
            if 'arguments' in payload:
                tool['arguments'] = payload['arguments']
            if 'output' in payload:
                tool['output'] = payload['output']
    return {'turns': list(turns.values()), 'tools': list(tools.values())}


def saved_partial(connection, run_id):
    rows = connection.execute(
        'SELECT sequence, event_type, payload_json FROM run_progress WHERE run_id = ? ORDER BY sequence', (run_id,)
    ).fetchall()
    return partial_from_progress([
        {'sequence': row[0], 'event_type': row[1], 'payload': json.loads(row[2])} for row in rows
    ])


def finish_message(connection, run_id, state, payload):
    if state == 'failed' and 'partial' not in payload:
        payload = {**payload, 'partial': saved_partial(connection, run_id)}
    text = payload.get('answer', {}).get('text', '') if state == 'succeeded' else '\n\n'.join(
        turn.get('content', '') for turn in payload.get('partial', {}).get('turns', [])
    ) or payload.get('message', '')
    connection.execute(
        "UPDATE chat_messages SET content = ?, result_json = ? WHERE run_id = ? AND role = 'assistant'",
        (text, json.dumps(payload, ensure_ascii=False), run_id),
    )
