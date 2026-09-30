import tempfile
import unittest
from pathlib import Path
from human_eval.contracts import RunRequest
from human_eval.store import EvaluationStore


class ChatMigrationTests(unittest.TestCase):
    def test_v5_unique_user_constraint_is_removed_without_losing_history(self):
        import uuid
        with tempfile.TemporaryDirectory() as directory:
            store = EvaluationStore(Path(directory) / 'eval.sqlite')
            store.initialize()
            run, _ = store.create_run(RunRequest('Historial existente'), user_id='alice', reserved_tokens=1000)
            with store._connect() as connection:
                conversations = [tuple(row) for row in connection.execute('SELECT * FROM chat_conversations')]
                messages = [tuple(row) for row in connection.execute('SELECT * FROM chat_messages')]
                connection.execute('DROP TABLE chat_messages')
                connection.execute('DROP TABLE chat_conversations')
                connection.execute('CREATE TABLE chat_conversations (conversation_id TEXT PRIMARY KEY, user_id TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL)')
                connection.execute("CREATE TABLE chat_messages (conversation_id TEXT REFERENCES chat_conversations(conversation_id), run_id TEXT, role TEXT, created_at TEXT, content TEXT, result_json TEXT, PRIMARY KEY(run_id, role))")
                connection.executemany('INSERT INTO chat_conversations VALUES (?, ?, ?)', conversations)
                connection.executemany('INSERT INTO chat_messages VALUES (?, ?, ?, ?, ?, ?)', messages)
                connection.execute("UPDATE schema_meta SET value = '5' WHERE key = 'schema_version'")
            store.initialize()
            store.validate_schema()
            store.create_run(RunRequest('Nueva conversación'), user_id='alice', reserved_tokens=1000, conversation_id=str(uuid.uuid4()))
            self.assertEqual(len(store.chat_conversations('alice')), 2)
            self.assertEqual(store.chat_runs('alice')[0]['question'], 'Historial existente')
            with store._connect() as connection:
                self.assertEqual(connection.execute('PRAGMA foreign_key_check').fetchall(), [])
            store.initialize()
            self.assertEqual(len(store.chat_conversations('alice')), 2)

    def test_v4_chats_move_into_conversation_tables_without_losing_text(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EvaluationStore(Path(directory) / 'eval.sqlite')
            store.initialize()
            run, _ = store.create_run(RunRequest('Mensaje original'), user_id='alice', reserved_tokens=10_000)
            run_id = run['run_id']
            store.start_run(run_id, provenance={})
            result = {'answer': {'text': 'Respuesta original'}, 'usage': {'total_tokens': 100}}
            store.append_event(run_id, 'succeeded', result)
            with store._connect() as connection:
                connection.execute('DROP TABLE chat_messages')
                connection.execute('DROP TABLE chat_conversations')
                connection.execute("UPDATE schema_meta SET value = '4' WHERE key = 'schema_version'")
            store.initialize()
            store.validate_schema()
            self.assertEqual(store.chat_runs('alice')[0]['question'], 'Mensaje original')
            self.assertEqual(store.chat_runs('alice')[0]['result'], result)
            self.assertEqual(store.runs_for_user('alice'), [])
            with store._connect() as connection:
                self.assertEqual(connection.execute("SELECT content FROM chat_messages WHERE role = 'assistant'").fetchone()[0], 'Respuesta original')
            store.initialize()  # Migration is idempotent.
            self.assertEqual(len(store.chat_runs('alice')), 1)


if __name__ == '__main__':
    unittest.main()
