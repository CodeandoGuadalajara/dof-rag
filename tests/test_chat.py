from tests.test_human_eval import AirAppTestCase, wait_for_terminal


class ChatTests(AirAppTestCase):
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
        self.assertIn('50,000', page.text)
        payload = {'csrf_token': self.hidden(page, 'csrf_token'),
                   'client_request_id': self.hidden(page, 'client_request_id'),
                   'question': 'Pregunta privada'}
        invalid = self.client.post('/chat', data={**payload, 'csrf_token': 'bad'})
        self.assertEqual(invalid.status_code, 403)
        sent = self.client.post('/chat', data=payload, follow_redirects=False)
        self.assertEqual(sent.status_code, 303)
        run = self.service.store.chat_runs('alice')[0]
        wait_for_terminal(self.service, run['run_id'])
        self.assertEqual(self.service.store.get_request(run['run_id']).token_limit, 50_000)
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
        self.assertIn('<details open><summary>Llamada a herramienta · search_documents', page.text)
        self.assertIn('<details open><summary>Pensamiento del modelo', page.text)
        self.assertIn('&lt;script&gt;query&lt;/script&gt;', page.text)
        self.assertNotIn('<script>thought</script>', page.text)
        second = {**payload, 'client_request_id': self.hidden(page, 'client_request_id'),
                  'question': '¿Y qué significa?'}
        self.client.post('/chat', data=second, follow_redirects=False)
        runs = self.service.store.chat_runs('alice')
        request = self.service.store.get_request(runs[-1]['run_id'])
        self.assertEqual(request.history[0]['content'], 'Pregunta privada')
        self.as_user('bob')
        self.assertNotIn('Pregunta privada', self.client.get('/chat').text)
