import plistlib
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

import convert_doc_to_md
import get_word_dof
from convert_doc_to_md import convert_single_doc
from corpus_store import chunk_index
from corpus_store.ingest import SCHEMA as CORPUS_SCHEMA
from get_word_dof import (
    has_valid_download,
    is_valid_sidof_listing,
    is_valid_word_payload,
)
from scripts import build_fts_full
from scripts.update_dof_daily import (
    build_manifest,
    choose_start_date,
    completed_through,
    conversion_command,
    last_jsonl_record,
)

WORD_BYTES = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"x" * 1016


class DailyUpdateTests(unittest.TestCase):
    def test_catches_up_from_day_after_watermark(self):
        self.assertEqual(
            choose_start_date(date(2026, 4, 24), date(2026, 8, 27), 7),
            date(2026, 4, 25),
        )

    def test_uses_overlap_when_watermark_is_current(self):
        self.assertEqual(
            choose_start_date(date(2026, 8, 27), date(2026, 8, 27), 7),
            date(2026, 8, 21),
        )

    def test_manifest_contains_only_requested_dates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            included = root / "2026/08/26082026/MAT/001_DOF_20260826_MAT_1.md"
            excluded = root / "2026/08/20082026/MAT/001_DOF_20260820_MAT_2.md"
            included.parent.mkdir(parents=True)
            excluded.parent.mkdir(parents=True)
            included.write_text("included", encoding="utf-8")
            excluded.write_text("excluded", encoding="utf-8")
            manifest = root / "manifest.jsonl"

            count = build_manifest(
                root, date(2026, 8, 24), date(2026, 8, 27), manifest
            )

            self.assertEqual(count, 1)
            self.assertIn("001_DOF_20260826_MAT_1.md", manifest.read_text())
            self.assertNotIn("001_DOF_20260820_MAT_2.md", manifest.read_text())

    def test_existing_note_id_survives_page_reordering(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            existing = directory / "005_DOF_20260826_MAT_12345.doc"
            existing.write_bytes(WORD_BYTES)
            self.assertTrue(has_valid_download(directory, "12345"))
            self.assertFalse(has_valid_download(directory, "99999"))

    def test_only_known_word_signatures_are_valid(self):
        self.assertTrue(is_valid_word_payload(WORD_BYTES))
        self.assertTrue(is_valid_word_payload(b"PK\x03\x04" + b"x" * 1024))
        self.assertFalse(
            is_valid_word_payload(b"<!DOCTYPE html><title>500</title>" + b"x" * 1024)
        )
        self.assertFalse(
            is_valid_word_payload(b'\xef\xbb\xbf{"error":true}' + b"x" * 1024)
        )

    def test_sidof_listing_accepts_dated_shell_with_either_notices_tab(self):
        html = (
            "<html><head><title>Diario Oficial de la Federación</title></head>"
            "<body><div id='resp-tab3'>26-08-2026</div></body></html>"
        )
        self.assertTrue(is_valid_sidof_listing(html, "26", "08", "2026"))

    def test_sidof_listing_rejects_pages_without_the_publication_shell(self):
        no_tabs = (
            "<html><head><title>Diario Oficial de la Federación</title></head>"
            "<body>26-08-2026</body></html>"
        )
        wrong_title = (
            "<html><head><title>Error</title></head>"
            "<body><div id='resp-tab3'>26-08-2026</div></body></html>"
        )
        wrong_date = (
            "<html><head><title>Diario Oficial de la Federación</title></head>"
            "<body><div id='resp-tab3'>25-08-2026</div></body></html>"
        )
        self.assertFalse(is_valid_sidof_listing(no_tabs, "26", "08", "2026"))
        self.assertFalse(is_valid_sidof_listing(wrong_title, "26", "08", "2026"))
        self.assertFalse(is_valid_sidof_listing(wrong_date, "26", "08", "2026"))

    def test_conversion_scan_is_limited_to_active_date_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "2026/04/24042026/MAT/001_DOF_old.doc"
            active = root / "2026/08/26082026/MAT/001_DOF_active.doc"
            future = root / "2026/08/28082026/MAT/001_DOF_future.doc"
            for path in (old, active, future):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"word")

            with mock.patch.object(convert_doc_to_md, "INPUT_DIR", root):
                files = convert_doc_to_md.find_doc_files(
                    [2026], date(2026, 8, 25), date(2026, 8, 27)
                )

            self.assertEqual(files, [active])

    def test_updater_passes_active_window_to_converter(self):
        command = conversion_command(2, date(2025, 12, 30), date(2026, 1, 2))
        self.assertEqual(
            command[command.index("--years") + 1:command.index("--start-date")],
            ["2025", "2026"],
        )
        self.assertEqual(command[command.index("--start-date") + 1], "2025-12-30")
        self.assertEqual(command[command.index("--end-date") + 1], "2026-01-02")

    def test_invalid_word_is_quarantined_for_redownload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            doc = root / "001_DOF_20260826_MAT_12345.doc"
            doc.write_bytes(b"<!DOCTYPE html><title>500</title>" + b"x" * 1024)

            result = convert_single_doc(doc, doc.with_suffix(".md"))

            self.assertEqual(result["status"], "invalid_download")
            self.assertFalse(doc.exists())
            self.assertTrue((root / f"{doc.name}.invalid").exists())
            self.assertFalse(has_valid_download(root, "12345"))

    def test_sidof_notices_are_checked_when_page_has_no_word_links(self):
        session = mock.Mock()
        response = mock.Mock()
        response.text = (
            "<html><head><title>DOF - Diario Oficial de la Federación</title></head>"
            "<body><div id='cuerpo_principal'>"
            "No hay datos para la fecha seleccionada</div></body></html>"
        )
        response.raise_for_status = lambda: None
        session.get.return_value = response

        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(
                get_word_dof, "process_sidof_notices", return_value=0
            ) as notices:
                downloaded = get_word_dof.process_dof_page(
                    session, "26/08/2026", "MAT", Path(temporary), sleep_delay=0
                )

        self.assertEqual(downloaded, 0)
        notices.assert_called_once()

    def test_malformed_dof_listing_is_an_error(self):
        session = mock.Mock()
        response = mock.Mock()
        response.text = "<html><title>500 error</title><body></body></html>"
        response.raise_for_status = lambda: None
        session.get.return_value = response

        with tempfile.TemporaryDirectory() as temporary:
            with (
                mock.patch.object(get_word_dof, "ERROR_COUNT", 0),
                mock.patch.object(
                    get_word_dof, "process_sidof_notices", return_value=0
                ),
            ):
                get_word_dof.process_dof_page(
                    session, "26/08/2026", "MAT", Path(temporary), sleep_delay=0
                )
                self.assertEqual(get_word_dof.ERROR_COUNT, 1)

    def test_malformed_sidof_listing_is_an_error(self):
        session = mock.Mock()
        response = mock.Mock()
        response.text = "<html><title>500 error</title><body></body></html>"
        response.raise_for_status = lambda: None
        session.get.return_value = response

        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(get_word_dof, "ERROR_COUNT", 0),
        ):
            get_word_dof.process_sidof_notices(
                session, "26", "08", "2026", "MAT", Path(temporary), sleep_delay=0
            )
            self.assertEqual(get_word_dof.ERROR_COUNT, 1)

    def test_chunk_store_rejects_versions_on_either_side_of_current(self):
        for version in ("aaa-older-version", "zzz-newer-version"):
            with self.subTest(version=version):
                chunks = sqlite3.connect(":memory:")
                chunks.executescript(chunk_index.SCHEMA)
                chunks.execute(
                    "INSERT INTO chunks (document_id, path, chunk_index, pattern,"
                    " start_offset, end_offset, spans_json, token_count, heading_path,"
                    " chunk_hash, chunker_version, corpus_version)"
                    " VALUES (1, 'x', 0, 'plain_text', 0, 1, '[]', 1, '[]',"
                    " ?, ?, 'v1')",
                    (b"0" * 32, version),
                )

                with self.assertRaisesRegex(RuntimeError, "Rebuild the chunk"):
                    chunk_index.require_current_chunker_version(chunks)

                chunks.close()

    def test_chunk_store_accepts_only_current_version(self):
        chunks = sqlite3.connect(":memory:")
        chunks.executescript(chunk_index.SCHEMA)
        chunks.execute(
            "INSERT INTO chunks (document_id, path, chunk_index, pattern,"
            " start_offset, end_offset, spans_json, token_count, heading_path,"
            " chunk_hash, chunker_version, corpus_version)"
            " VALUES (1, 'x', 0, 'plain_text', 0, 1, '[]', 1, '[]', ?, ?, 'v1')",
            (b"0" * 32, chunk_index.CHUNKER_VERSION),
        )

        chunk_index.require_current_chunker_version(chunks)

        chunks.close()

    def test_chunk_scan_starts_after_existing_checkpoint(self):
        corpus = sqlite3.connect(":memory:")
        corpus.executescript(CORPUS_SCHEMA)
        for path in ("one.md", "two.md", "three.md"):
            corpus.execute(
                "INSERT INTO documents(path, year, publication_date, section, markdown,"
                " byte_length, sha256, corpus_version)"
                " VALUES (?, 2026, '2026-08-26', 'MAT', ?, 4, ?, 'v1')",
                (path, path, b"0" * 32),
            )

        rows = list(chunk_index.iter_documents(corpus, after_document_id=2))

        self.assertEqual([(row[0], row[1]) for row in rows], [(3, "three.md")])
        corpus.close()

    def test_fts_appends_normal_and_segmented_documents(self):
        with tempfile.TemporaryDirectory() as temporary:
            db_path = Path(temporary) / "corpus.sqlite"
            conn = sqlite3.connect(db_path)
            conn.executescript(CORPUS_SCHEMA)
            conn.execute(
                "INSERT INTO documents(path, year, publication_date, section, markdown,"
                " byte_length, sha256, corpus_version)"
                " VALUES ('one.md', 2026, '2026-08-26', 'MAT',"
                " 'decreto inicial', 15, ?, 'v1')",
                (b"0" * 32,),
            )
            conn.commit()
            conn.close()

            self._run_fts(db_path)

            conn = sqlite3.connect(db_path)
            conn.execute(
                "INSERT INTO documents(path, year, publication_date, section, markdown,"
                " byte_length, sha256, corpus_version)"
                " VALUES ('two.md', 2026, '2026-08-27', 'MAT',"
                " 'decreto nuevo', 13, ?, 'v1')",
                (b"1" * 32,),
            )
            cursor = conn.execute(
                "INSERT INTO documents(path, year, publication_date, section, markdown,"
                " byte_length, sha256, corpus_version)"
                " VALUES ('three.md', 2026, '2026-08-27', 'MAT', '', 18, ?, 'v1')",
                (b"2" * 32,),
            )
            conn.execute(
                "INSERT INTO document_segments(document_id, segment_index, start_offset,"
                " end_offset, segment_text) VALUES (?, 0, 0, 18, 'aviso segmentado')",
                (cursor.lastrowid,),
            )
            conn.commit()
            conn.close()

            self._run_fts(db_path)

            conn = sqlite3.connect(db_path)
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM documents_fts_docsize").fetchone()[0],
                3,
            )
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM documents_fts"
                    " WHERE documents_fts MATCH 'segmentado'"
                ).fetchone()[0],
                1,
            )
            conn.close()

    def _run_fts(self, db_path: Path) -> None:
        argv = ["build_fts_full", "--corpus-db", str(db_path), "--batch", "2"]
        # The fixture is uncompressed; no native sqlite-zstd extension is needed.
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            build_fts_full, "connect", sqlite3.connect
        ):
            build_fts_full.main()

    @unittest.skipUnless(shutil.which("zsh"), "launchd installer requires zsh (macOS)")
    def test_launchd_renderer_preserves_special_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "rendered.plist"
            runner = root / "Application Support/DOF & RAG/run.sh"
            repository = root / "DOF & RAG/repo"
            subprocess.run(
                [
                    "zsh",
                    "scripts/install_dof_launchd.sh",
                    "--render-plist",
                    str(output),
                    str(runner),
                    str(repository),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            with output.open("rb") as stream:
                payload = plistlib.load(stream)

            self.assertEqual(payload["ProgramArguments"][-1], str(runner))
            self.assertEqual(
                payload["EnvironmentVariables"]["DOF_REPO_DIR"], str(repository)
            )
            self.assertEqual(
                payload["StandardOutPath"], str(repository / "logs/dof-daily.log")
            )

    def test_full_manifest_is_initial_contiguous_watermark(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                '{"publication_date": "2026-04-23"}\n'
                '{"publication_date": "2026-04-24"}\n',
                encoding="utf-8",
            )

            self.assertEqual(
                completed_through(root / "missing-state.json", manifest, root / "db"),
                date(2026, 4, 24),
            )
            self.assertEqual(
                last_jsonl_record(manifest), {"publication_date": "2026-04-24"}
            )


class PermanentlyMissingDownloadTest(unittest.TestCase):
    def _missing_response(self) -> mock.Mock:
        response = mock.Mock()
        response.content = (
            b"<!DOCTYPE html><html><title>El archivo no existe</title>"
            + b"x" * 1024
        )
        response.url = (
            "https://sidof.segob.gob.mx/notificacion/page/204/"
            "El%20archivo%20no%20existe"
        )
        response.headers = {"Content-Type": "text/html; charset=UTF-8"}
        response.text = "<html><body>El archivo no existe</body></html>"
        response.raise_for_status = lambda: None
        return response

    def test_permanent_missing_creates_tombstone_without_error(self):
        session = mock.Mock()
        session.get.return_value = self._missing_response()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "001_AVISO_20260825_MAT_5789248.doc"
            old_errors = get_word_dof.ERROR_COUNT
            try:
                get_word_dof.ERROR_COUNT = 0
                ok = get_word_dof._download_file(
                    session,
                    "https://sidof.segob.gob.mx/notas/getDoc/5789248",
                    target,
                )
                self.assertFalse(ok)
                self.assertEqual(get_word_dof.ERROR_COUNT, 0)
            finally:
                get_word_dof.ERROR_COUNT = old_errors
            self.assertFalse(target.exists())
            marker = root / "001_AVISO_20260825_MAT_5789248.doc.missing"
            self.assertTrue(marker.exists())
            self.assertTrue(get_word_dof.has_missing_marker(root, "5789248"))
            self.assertFalse(get_word_dof.has_missing_marker(root, "99999"))

    def test_invalid_payload_without_missing_page_still_errors(self):
        session = mock.Mock()
        response = mock.Mock()
        response.content = b"<html><title>500</title></html>" + b"x" * 1024
        response.url = "https://sidof.segob.gob.mx/notas/getDoc/5789248"
        response.headers = {"Content-Type": "text/html"}
        response.text = "<html><title>500</title></html>"
        response.raise_for_status = lambda: None
        session.get.return_value = response
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "001_AVISO_20260825_MAT_5789248.doc"
            old_errors = get_word_dof.ERROR_COUNT
            try:
                get_word_dof.ERROR_COUNT = 0
                ok = get_word_dof._download_file(
                    session,
                    "https://sidof.segob.gob.mx/notas/getDoc/5789248",
                    target,
                )
                self.assertFalse(ok)
                self.assertEqual(get_word_dof.ERROR_COUNT, 1)
            finally:
                get_word_dof.ERROR_COUNT = old_errors
            self.assertFalse(get_word_dof.has_missing_marker(root, "5789248"))


if __name__ == "__main__":
    unittest.main()
