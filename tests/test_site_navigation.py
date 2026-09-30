from tests.test_human_eval import AirAppTestCase


class SiteNavigationTests(AirAppTestCase):
    def test_composer_allows_drafting_during_generation(self):
        import re
        import threading
        started, release = threading.Event(), threading.Event()
        execute = self.executor.execute
        def slow(*args, **kwargs):
            started.set()
            release.wait(5)
            return execute(*args, **kwargs)
        self.executor.execute = slow
        self.as_user('alice')
        page = self.client.get('/chat')
        try:
            self.client.post('/chat', data={
                'csrf_token': self.hidden(page, 'csrf_token'),
                'client_request_id': self.hidden(page, 'client_request_id'),
                'question': 'Consulta pendiente',
            }, follow_redirects=False)
            self.assertTrue(started.wait(2))
            page = self.client.get('/chat')
            textarea = re.search(r'<textarea id="chat-question"[^>]*>', page.text)[0]
            self.assertNotIn('disabled', textarea)
            self.assertIn('<button type="submit" disabled>Enviar</button>', page.text)
            self.assertIn('Puedes preparar tu siguiente mensaje', page.text)
        finally:
            release.set()

    def test_menu_defaults_to_questions_and_admin_is_permission_scoped(self):
        home = self.client.get('/')
        self.assertIn('<nav class="site-nav" aria-label="Navegación principal">', home.text)
        self.assertIn('<a href="/" aria-current="page">Preguntas</a>', home.text)
        self.assertIn('<a href="/chat">Chat</a>', home.text)
        self.assertNotIn('href="/admin"', home.text)
        self.as_user('alice')
        chat = self.client.get('/chat')
        self.assertIn('<a href="/chat" aria-current="page">Chat</a>', chat.text)
        self.assertNotIn('href="/admin"', chat.text)
        self.assertIn('<body class="chat-page">', chat.text)
        self.assertIn('class="chat-messages" data-chat-messages', chat.text)
        self.assertIn('class="chat-composer" data-chat-form', chat.text)
        self.as_user('admin', admin=True)
        admin = self.client.get('/admin')
        self.assertIn('<a href="/admin" aria-current="page">Admin</a>', admin.text)
        self.assertIn('<a href="/admin">Admin</a>', self.client.get('/chat').text)
