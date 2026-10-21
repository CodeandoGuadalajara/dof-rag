"""Corpus reader stays independent of private conversation state and inference."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from starlette.testclient import TestClient

from agent_tools.models import EvidenceHit
from human_eval.app import WebSettings, create_app
from human_eval.auth import FakeAuthBackend
from human_eval.reader import passage_body
from human_eval.service import EvaluationService
from human_eval.store import EvaluationStore


def hit(chunk_id=7, **changes):
    values = dict(chunk_id=chunk_id, document_id=1, path="private-local-path",
                  publication_date="2025-01-23", section="Primera", chunk_index=chunk_id,
                  heading_path=["Artículo <1>"], text="Texto <script>alert(1)</script>",
                  score=0, source="read", title="Título <DOF>")
    return EvidenceHit(**(values | changes))


class PassageReaderTests(unittest.TestCase):
    def test_render_context_and_escape_untrusted_text(self):
        text = passage_body(7, [hit(6), hit(), hit(8)])
        self.assertEqual(text.count("Pasaje seleccionado"), 1)
        self.assertEqual(text.count("Contexto cercano"), 2)
        self.assertIn('/pasajes/8#pasaje-8', text)
        self.assertIn("&lt;script&gt;", text)
        self.assertNotIn("<script>", text)
        self.assertNotIn("private-local-path", text)
        self.assertIn("year=2025&amp;month=1&amp;day=23", text)
        self.assertIn("no el documento completo", text)
        self.assertIn("no determina la vigencia", text)

    def test_missing_or_invalid_date_does_not_invent_official_link(self):
        for value in (None, "invalid", "2025-02-30"):
            with self.subTest(value=value):
                text = passage_body(7, [hit(publication_date=value)])
                self.assertNotIn('href="https://', text)

    def test_public_route_and_missing_passage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evaluation.sqlite"
            reader = Mock(return_value=[hit()])
            store = EvaluationStore(path)
            store.initialize()
            service = EvaluationService(store)
            app = create_app(service, WebSettings(host="127.0.0.1", port=0, db_path=path,
                             session_secret="a-test-secret-with-at-least-32-characters"),
                             lambda: {}, auth_backend=FakeAuthBackend(), passage_reader=reader)
            with TestClient(app) as client:
                response = client.get('/pasajes/7')
                self.assertEqual(response.status_code, 200)
                self.assertIn("Pasaje seleccionado", response.text)
                self.assertEqual(response.headers['Cache-Control'], 'no-store')
                reader.assert_called_once_with(7)
                reader.side_effect = KeyError(999)
                self.assertEqual(client.get('/pasajes/999').status_code, 404)
                reader.reset_mock()
                self.assertEqual(client.get('/pasajes/0').status_code, 404)
                reader.assert_not_called()
                for error in (ValueError('hash verification: sensitive details'),
                              FileNotFoundError('/private/database')):
                    reader.side_effect = error
                    unavailable = client.get('/pasajes/7')
                    self.assertEqual(unavailable.status_code, 503)
                    self.assertNotIn(str(error), unavailable.text)
                    self.assertNotIn('Texto &lt;script&gt;', unavailable.text)
