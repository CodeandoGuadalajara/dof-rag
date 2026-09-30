from tests.test_human_eval import AirAppTestCase, wait_for_terminal
from human_eval.contracts import RunRequest
from human_eval.token_budget import DAILY_TOKEN_LIMIT


class ConversationTests(AirAppTestCase):
    def setUp(self):
        super().setUp()
        execute = self.executor.execute
        def metered(*args, **kwargs):
            result = execute(*args, **kwargs)
            result['usage'] = {'total_tokens': 100}
            return result
        self.executor.execute = metered

    def send(self, url, question):
        page = self.client.get(url)
        key = self.hidden(page, 'client_request_id')
        response = self.client.post('/chat', data={
            'csrf_token': self.hidden(page, 'csrf_token'), 'client_request_id': key,
            'conversation_id': self.hidden(page, 'conversation_id'), 'question': question,
        }, follow_redirects=False)
        self.assertEqual(response.status_code, 303, response.text)
        run = self.service.store.find_idempotent_run(self.client.headers['x-eval-user'], key)
        wait_for_terminal(self.service, run['run_id'])
        return response.headers['location'], run['run_id']

    def test_new_selection_append_and_history_are_conversation_scoped(self):
        self.as_user('alice')
        alpha, first = self.send('/chat?new=1', 'Pregunta alfa')
        same, second = self.send(alpha, 'Continúa con alfa')
        self.assertEqual(alpha, same)
        beta, third = self.send('/chat?new=1', 'Pregunta beta')
        self.assertNotEqual(alpha, beta)
        self.assertEqual(len(self.service.store.chat_conversations('alice')), 2)
        self.assertEqual(self.service.store.get_request(third).history, ())
        self.assertEqual(self.service.store.get_request(second).history[0]['content'], 'Pregunta alfa')
        page = self.client.get(alpha)
        self.assertIn(beta, page.text)
        self.assertIn('+ Nueva conversación', page.text)
        messages = page.text.split('data-chat-messages')[1].split('class="chat-composer"')[0]
        self.assertIn('Continúa con alfa', messages)
        self.assertNotIn('Pregunta beta', messages)
        same, fourth = self.send(alpha, 'Otro mensaje en alfa')
        self.assertEqual(alpha, same)
        self.assertNotIn('Pregunta beta', str(self.service.store.get_request(fourth).history))

    def test_other_users_cannot_read_or_append_and_retries_cannot_switch_conversation(self):
        self.as_user('alice')
        url, run_id = self.send('/chat?new=1', 'Mensaje privado')
        conversation_id = self.service.store.conversation_for_run('alice', run_id)
        from human_eval.store import IdempotencyPayloadConflict
        import uuid
        with self.assertRaises(IdempotencyPayloadConflict):
            self.service.store.create_run(self.service.store.get_request(run_id), user_id='alice',
                reserved_tokens=100, conversation_id=str(uuid.uuid4()))
        self.as_user('bob')
        self.assertEqual(self.client.get(url).status_code, 404)
        page = self.client.get('/chat?new=1')
        rejected = self.client.post('/chat', data={
            'csrf_token': self.hidden(page, 'csrf_token'),
            'client_request_id': self.hidden(page, 'client_request_id'),
            'conversation_id': conversation_id, 'question': 'Intento de acceso',
        }, follow_redirects=False)
        self.assertEqual(rejected.status_code, 404)
        self.assertEqual(self.service.store.chat_conversations('bob'), [])
        self.assertEqual(self.service.store.token_balance('bob'), DAILY_TOKEN_LIMIT)

    def test_admin_can_chat_with_zero_balance_without_new_reservations(self):
        store = self.service.store
        old, _ = store.create_run(RunRequest('Cuota anterior'), user_id='root', reserved_tokens=DAILY_TOKEN_LIMIT)
        store.start_run(old['run_id'], provenance={})
        store.append_event(old['run_id'], 'succeeded', {'answer': {'text': 'Respuesta previa'}, 'usage': {'total_tokens': DAILY_TOKEN_LIMIT}})
        self.as_user('root', admin=True)
        self.assertEqual(store.token_balance('root'), 0)
        self.assertIn('Sin límite diario de tokens', self.client.get('/chat').text)
        url, run_id = self.send('/chat?new=1', 'Consulta de administrador')
        request = store.get_request(run_id)
        self.assertTrue(request.is_chat)
        self.assertIsNone(request.token_limit)
        self.send(url, 'Segunda consulta de administrador')
        self.assertEqual(store.token_balance('root'), 0)
        with store._connect() as connection:
            self.assertIsNone(connection.execute('SELECT 1 FROM token_reservations WHERE run_id = ?', (run_id,)).fetchone())
