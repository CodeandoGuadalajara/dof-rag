from __future__ import annotations

import subprocess
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from agent_tools.retrieval import LlamaQueryEmbedder
from corpus_store.embed import DIMS, pack_binary, start_server, stop_server


class EmbeddingServerTests(unittest.TestCase):
    def test_pack_binary_rejects_wrong_dimension(self):
        with self.assertRaisesRegex(ValueError, "embedding shape"):
            pack_binary(np.zeros(DIMS + 1, dtype=np.float32))

    def test_start_server_rejects_an_occupied_port_before_spawning(self):
        with (
            mock.patch(
                "corpus_store.embed._assert_port_available",
                side_effect=RuntimeError("occupied"),
            ),
            mock.patch("corpus_store.embed.subprocess.Popen") as popen,
            self.assertRaisesRegex(RuntimeError, "occupied"),
        ):
            start_server(Path("model.gguf"), ctx=8192, port=8086)
        popen.assert_not_called()

    def test_start_server_reaps_a_child_that_exits_during_startup(self):
        process = mock.Mock()
        process.poll.return_value = 9
        with (
            mock.patch("corpus_store.embed._assert_port_available"),
            mock.patch("corpus_store.embed.subprocess.Popen", return_value=process),
            self.assertRaisesRegex(RuntimeError, "code 9"),
        ):
            start_server(Path("model.gguf"), ctx=8192, port=8086)
        process.wait.assert_called_once_with()

    def test_stop_server_kills_a_child_that_ignores_terminate(self):
        process = mock.Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("llama-server", 0.1), 0]

        stop_server(process, timeout=0.1)

        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()

    def test_query_embedder_probes_and_closes_once(self):
        process = mock.Mock()
        process.poll.return_value = None
        with (
            mock.patch("corpus_store.embed.start_server", return_value=process),
            mock.patch(
                "corpus_store.embed.embed_batch",
                return_value=np.zeros((1, DIMS), dtype=np.float32),
            ) as embed_batch,
            mock.patch("corpus_store.embed.stop_server") as stop,
            mock.patch("agent_tools.retrieval.atexit.register") as register,
            mock.patch("agent_tools.retrieval.atexit.unregister") as unregister,
        ):
            embedder = LlamaQueryEmbedder(Path("model.gguf"))
            embedder.close()
            embedder.close()

        embed_batch.assert_called_once_with(["Query: probe"], 8086)
        register.assert_called_once()
        unregister.assert_called_once()
        stop.assert_called_once_with(process)

    def test_close_waits_for_an_in_flight_embedding(self):
        process = mock.Mock()
        process.poll.return_value = None
        query_started = threading.Event()
        release_query = threading.Event()
        close_started = threading.Event()
        errors: list[BaseException] = []

        def embed_batch(texts, port):
            if texts == ["Query: probe"]:
                return np.zeros((1, DIMS), dtype=np.float32)
            query_started.set()
            if not release_query.wait(timeout=1):
                raise TimeoutError("test did not release the embedding request")
            return np.zeros((1, DIMS), dtype=np.float32)

        with (
            mock.patch("corpus_store.embed.start_server", return_value=process),
            mock.patch("corpus_store.embed.embed_batch", side_effect=embed_batch),
            mock.patch("corpus_store.embed.stop_server") as stop,
            mock.patch("agent_tools.retrieval.atexit.register"),
            mock.patch("agent_tools.retrieval.atexit.unregister"),
        ):
            embedder = LlamaQueryEmbedder(Path("model.gguf"))

            def run_query():
                try:
                    embedder.embed_query("consulta")
                except BaseException as exc:  # noqa: BLE001 (capture any embedder failure)
                    errors.append(exc)

            def run_close():
                close_started.set()
                embedder.close()

            query_thread = threading.Thread(target=run_query)
            close_thread = threading.Thread(target=run_close)
            query_thread.start()
            self.assertTrue(query_started.wait(timeout=1))
            close_thread.start()
            self.assertTrue(close_started.wait(timeout=1))
            close_thread.join(timeout=0.05)

            self.assertTrue(close_thread.is_alive())
            stop.assert_not_called()

            release_query.set()
            query_thread.join(timeout=1)
            close_thread.join(timeout=1)

        self.assertFalse(query_thread.is_alive())
        self.assertFalse(close_thread.is_alive())
        self.assertEqual(errors, [])
        stop.assert_called_once_with(process)


if __name__ == "__main__":
    unittest.main()
