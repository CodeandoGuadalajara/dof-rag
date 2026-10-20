"""Fase 3 (#88): match headings and brief excerpts on document candidates."""

import json
import sqlite3
import unittest

from agent_tools import retrieval
from agent_tools.models import (
    DocumentHit,
    DocumentSearchResult,
    RetrievalStrategy,
    SearchFilters,
)


def make_retriever():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE chunks (chunk_id INTEGER, document_id INTEGER, "
        "chunk_index INTEGER, heading_path TEXT)"
    )
    retriever = object.__new__(retrieval.DofRetriever)
    retriever.chunks = connection
    return retriever, connection


def add_chunk(connection, chunk_id, document_id, chunk_index, heading_path):
    connection.execute(
        "INSERT INTO chunks VALUES (?, ?, ?, ?)",
        (chunk_id, document_id, chunk_index, json.dumps(heading_path)),
    )
    connection.commit()


def stub_reconstruction(retriever, texts):
    """Map chunk_id -> (heading_path, text) without corpus access."""
    records = {}
    for chunk_id, (heading_path, _text) in texts.items():
        records[chunk_id] = retrieval._ChunkRecord(
            chunk_id=chunk_id,
            document_id=1,
            path="doc.md",
            chunk_index=chunk_id,
            pattern="body",
            spans_json="[]",
            chunk_hash=b"",
            heading_path=heading_path,
            token_count=10,
            publication_date="2025-01-01",
            section="MAT",
        )

    retriever._chunk_metadata = lambda chunk_ids: {
        chunk_id: records[chunk_id] for chunk_id in chunk_ids
    }
    retriever._reconstruct_records = lambda records_: [
        (record, texts[record.chunk_id][1]) for record in records_
    ]
    return records


class CandidateMatchEvidenceTests(unittest.TestCase):
    def test_headings_and_excerpt_match_query_terms(self):
        retriever, connection = make_retriever()
        add_chunk(connection, 10, 1, 0, ["Otra sección"])
        add_chunk(connection, 11, 1, 1, ["Programa de Apoyo al Empleo", "10. Apoyos"])
        add_chunk(connection, 12, 1, 2, ["Anexo"])
        stub_reconstruction(
            retriever,
            {
                11: (["Programa de Apoyo al Empleo", "10. Apoyos"],
                     "El apoyo económico mensual es de $1,600.00 M.N."),
                12: (["Anexo"], "sin coincidencia"),
            },
        )
        headings, excerpt = retriever._candidate_match_evidence(1, "apoyo empleo")
        self.assertEqual(headings, ["Programa de Apoyo al Empleo > 10. Apoyos"])
        self.assertIn("apoyo económico", excerpt)

    def test_heading_and_excerpt_limits(self):
        retriever, connection = make_retriever()
        headings = []
        for index in range(5):
            heading = f"Apoyo numeral {index}"
            add_chunk(connection, 20 + index, 1, index, [heading])
            headings.append(heading)
        long_text = "apoyo " + "x" * 1000
        stub_reconstruction(
            retriever,
            {
                20 + index: ([headings[index]], long_text)
                for index in range(3)
            },
        )
        result_headings, excerpt = retriever._candidate_match_evidence(1, "apoyo")
        self.assertEqual(len(result_headings), retrieval.MATCH_HEADING_LIMIT)
        self.assertLessEqual(len(excerpt), retrieval.EXCERPT_CHARS)

    def test_short_query_terms_return_nothing(self):
        retriever, _ = make_retriever()
        self.assertEqual(retriever._candidate_match_evidence(1, "a b"), ([], None))

    def test_no_heading_match_returns_nothing(self):
        retriever, connection = make_retriever()
        add_chunk(connection, 30, 1, 0, ["Instituciones"])
        called = []
        retriever._reconstruct_records = lambda records: called.append(records) or []
        retriever._chunk_metadata = lambda chunk_ids: {}
        headings, excerpt = retriever._candidate_match_evidence(1, "apoyo")
        self.assertEqual((headings, excerpt), ([], None))
        self.assertEqual(called, [])

    def test_candidate_fields_round_trip_through_tool_dict(self):
        result = DocumentSearchResult(
            query="apoyo",
            strategy=RetrievalStrategy.HYBRID,
            filters=SearchFilters(),
            documents=[
                DocumentHit(
                    1, "doc.md", "2025-01-01", "MAT", 1.0,
                    match_headings=["Apoyo"], match_excerpt="extracto breve",
                )
            ],
        )
        document = result.to_dict()["documents"][0]
        self.assertEqual(document["match_headings"], ["Apoyo"])
        self.assertEqual(document["match_excerpt"], "extracto breve")


if __name__ == "__main__":
    unittest.main()
