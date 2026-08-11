import unittest

from agent_tools.llm import _parse_json, answer_with_context
from agent_tools.models import EvidenceHit, SearchFilters, SearchResult
from agent_tools.retrieval import _bm25_chunk_scores, _fuse_documents, _rrf


class FakeClient:
    def complete(self, system, user, *, max_tokens=1200):
        return '{"answer":"ok","citations":[4,999],"premise_status":"supported"}'


class AgentToolsTests(unittest.TestCase):
    def test_weighted_fusion_prefers_lexical_when_weight_is_high(self):
        fused = _fuse_documents([(1, 10.0), (2, 1.0)], [(2, 0.9), (1, 0.1)], 0.75)
        self.assertEqual(fused[0][0], 1)

    def test_rrf_keeps_items_from_both_lists(self):
        self.assertEqual(_rrf([[1, 2], [3, 2]])[0], 2)

    def test_parse_json_allows_fenced_model_output(self):
        data = _parse_json('```json\n{"answer":"ok","citations":[4]}\n```')
        self.assertEqual(data, {"answer": "ok", "citations": [4]})

    def test_parse_json_returns_first_complete_object(self):
        data = _parse_json('before {"answer":"first"} after {"answer":"second"}')
        self.assertEqual(data, {"answer": "first"})

    def test_filters_validate_dates_and_normalize_section(self):
        filters = SearchFilters(
            date_from="2026-01-01", date_to="2026-01-31", section=" mat "
        )
        self.assertEqual(filters.section, "MAT")
        with self.assertRaises(ValueError):
            SearchFilters(date_from="2026-02-01", date_to="2026-01-01")

    def test_bounded_bm25_does_not_reward_irrelevant_length(self):
        scores = _bm25_chunk_scores(
            "agua", ["agua potable", "agua " + "relleno " * 100]
        )
        self.assertGreater(scores[0], scores[1])

    def test_answer_rejects_citations_outside_supplied_context(self):
        result = SearchResult(
            query="pregunta",
            as_of="2026-01-01",
            evidence=[
                EvidenceHit(
                    chunk_id=4,
                    document_id=2,
                    path="doc.md",
                    publication_date="2025-01-01",
                    section="MAT",
                    chunk_index=0,
                    heading_path=[],
                    text="evidencia",
                    score=1.0,
                    source="test",
                    rank=1,
                )
            ],
        )
        answer = answer_with_context(FakeClient(), "pregunta", result)
        self.assertEqual(answer.citations, [4])
        self.assertEqual(answer.invalid_citations, [999])
