import tempfile
import unittest
from pathlib import Path
from human_eval.contracts import RunRequest
from human_eval.store import EvaluationStore


class ChatMigrationTests(unittest.TestCase):
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
