from human_eval.token_budget import DAILY_TOKEN_LIMIT
from tests.test_human_eval import AirAppTestCase, wait_for_terminal


class ChatTests(AirAppTestCase):
    def test_budget_failure_keeps_streamed_text_thoughts_and_tools(self):
        from human_eval.service import PublicExecutionError
        def exhausted(request, *, on_progress=None):
            for kind, payload in [
                ('model_turn_started', {'chat_delta': 'reasoning_content', 'text': 'Pensamiento previo', 'turn': 1}),
                ('model_turn_started', {'chat_delta': 'content', 'text': 'Respuesta parcial <script>bad</script>', 'turn': 1}),
                ('tool_started', {'call_id': 'call-1', 'tool': 'search_documents', 'arguments': {'query': 'DOF'}}),
                ('tool_completed', {'call_id': 'call-1', 'tool': 'search_documents', 'output': {'ok': True}}),
            ]:
                on_progress(kind, payload)
            raise PublicExecutionError('token_budget_exhausted', 'La consulta alcanzó su presupuesto de tokens.', used_tokens=1234)
        self.executor.execute = exhausted
        self.as_user('alice')
        page = self.client.get('/chat')
        self.client.post('/chat', data={
            'csrf_token': self.hidden(page, 'csrf_token'),
            'client_request_id': self.hidden(page, 'client_request_id'), 'question': 'Pregunta larga',
        }, follow_redirects=False)
        run = self.service.store.chat_runs('alice')[0]
        wait_for_terminal(self.service, run['run_id'])
        page = self.client.get('/chat')
        self.assertIn('Respuesta parcial', page.text)
        self.assertIn('<pre class="chat-thinking">Pensamiento previo</pre>', page.text)
        self.assertIn('Texto parcial', page.text)
        self.assertIn('<details><summary>Llamada a herramienta · search_documents', page.text)
        self.assertNotIn('<script>bad</script>', page.text)
        self.assertEqual(self.service.store.token_balance('alice'), DAILY_TOKEN_LIMIT - 1234)
        with self.service.store._connect() as connection:
            content = connection.execute("SELECT content FROM chat_messages WHERE role = 'assistant'").fetchone()[0]
            self.assertIn('Respuesta parcial', content)
        # Reloading keeps the partial content, not only the warning.
        self.assertIn('Respuesta parcial', self.client.get('/chat').text)

    def test_private_chat_admission_history_budget_and_csrf(self):
        self.assertEqual(self.client.get('/chat', follow_redirects=False).status_code, 303)
        execute = self.executor.execute
        def metered(*args, **kwargs):
            result = execute(*args, **kwargs)
            result['usage'] = {'total_tokens': 1000}
            result['reasoning'] = [{'turn': 1, 'text': '<script>thought</script>', 'tokens': None}]
            result['trace'] = [{'name': 'search_documents', 'arguments': {'query': '<script>query</script>'}, 'output': {'ok': True}}]
            return result
        self.executor.execute = metered
        self.as_user('alice')
        page = self.client.get('/chat')
        self.assertIn(f'{DAILY_TOKEN_LIMIT:,}', page.text)
        payload = {'csrf_token': self.hidden(page, 'csrf_token'),
                   'client_request_id': self.hidden(page, 'client_request_id'),
                   'question': 'Pregunta privada'}
        invalid = self.client.post('/chat', data={**payload, 'csrf_token': 'bad'})
        self.assertEqual(invalid.status_code, 403)
        sent = self.client.post('/chat', data=payload, follow_redirects=False)
        self.assertEqual(sent.status_code, 303)
        run = self.service.store.chat_runs('alice')[0]
        wait_for_terminal(self.service, run['run_id'])
        self.assertEqual(self.service.store.get_request(run['run_id']).token_limit, DAILY_TOKEN_LIMIT)
        # Replay must not reserve again.
        self.client.post('/chat', data=payload, follow_redirects=False)
        self.assertEqual(len(self.service.store.chat_runs('alice')), 1)
        self.assertEqual(self.service.store.runs_for_user('alice'), [])
        self.assertEqual(self.service.store.runs_for_moderation(), [])
        self.assertEqual(self.service.store.admin_runs(), [])
        self.assertNotIn('Pregunta privada', self.client.get('/').text)
        self.assertEqual(self.service.store.count_submissions_since('alice', '2000-01-01'), 0)
        with self.assertRaises(ValueError):
            self.service.store.publish_run(run['run_id'], publisher_id='admin')
        with self.service.store._connect() as connection:
            self.assertEqual(connection.execute('SELECT COUNT(*) FROM chat_conversations').fetchone()[0], 1)
            self.assertEqual(connection.execute('SELECT COUNT(*) FROM chat_messages').fetchone()[0], 2)
            content = connection.execute("SELECT content FROM chat_messages WHERE role = 'user'").fetchone()[0]
            self.assertEqual(content, 'Pregunta privada')
        page = self.client.get('/chat')
        self.assertIn('<details><summary>Llamada a herramienta · search_documents', page.text)
        self.assertNotIn('<details open><summary>Llamada a herramienta', page.text)
        self.assertIn('<details open><summary>Pensamiento del modelo', page.text)
        self.assertIn('&lt;script&gt;query&lt;/script&gt;', page.text)
        self.assertNotIn('<script>thought</script>', page.text)
        self.assertIn('<pre class="chat-thinking">&lt;script&gt;thought&lt;/script&gt;</pre>', page.text)
        self.assertIn('.chat-assistant .chat-thinking { max-height:none; overflow:visible;', page.text)
        self.assertIn("reasoning.className = 'chat-thinking'", page.text)
        # Terminal refresh follows the bottom: the final answer must come last.
        answer_position = page.text.index('<strong>DOF</strong>')
        self.assertLess(page.text.index('<pre class="chat-thinking">'), answer_position)
        self.assertLess(page.text.index('<details><summary>Llamada a herramienta'), answer_position)
        second = {**payload, 'client_request_id': self.hidden(page, 'client_request_id'),
                  'question': '¿Y qué significa?'}
        self.client.post('/chat', data=second, follow_redirects=False)
        runs = self.service.store.chat_runs('alice')
        request = self.service.store.get_request(runs[-1]['run_id'])
        self.assertEqual(request.history[0]['content'], 'Pregunta privada')
        self.as_user('bob')
        self.assertNotIn('Pregunta privada', self.client.get('/chat').text)
