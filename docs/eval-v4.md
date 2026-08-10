# DOF-RAG evidence evaluation v4

Eval v4 is a manually curated, corpus-backed pilot for evidence retrieval and
answer evaluation. It complements rather than replaces the 3,013-query v3
known-document retrieval benchmark.

## Scope

- 42 Spanish questions: six in each of seven categories.
- Frozen corpus: `dof-full-v1`, 657,867 documents, 1999-01-04 through
  2026-04-24.
- Frozen chunker: `dof-chunker-v1`.
- Every gold source has a corpus `document_id`, relative path, publication
  date, section, chunk id/index, and quoted supporting span.
- Multi-document questions identify every required evidence hop.
- Negative questions contain a corpus-backed correction rather than an empty
  answer.

The data is in `eval/dof_queries_v4.jsonl`; snapshot and methodology metadata
is in `eval/dof_queries_v4.meta.json`.

## Public review

The 42 questions, reference answers, and supporting spans are also available in
a [reader-friendly review edition](https://codeandoguadalajara.github.io/dof-rag-website/es/evals/v4).
Reviewers can discuss the methodology in
[GitHub Discussions](https://github.com/CodeandoGuadalajara/dof-rag/discussions)
or submit a specific correction through the repository's
[Eval v4 feedback form](https://github.com/CodeandoGuadalajara/dof-rag/issues/new?template=eval-v4-feedback.yml).

Feedback should identify the question id and distinguish among problems with
the question, reference answer, evidence, date boundary, or accepted sources.
One well-supported correction is useful; reviewers do not need to assess all
42 questions.

## Categories

| Category | Purpose |
|---|---|
| `single_passage` | One fact or definition supported by one passage |
| `list_enumeration` | Complete set of items, sometimes across passages |
| `temporal_transitorio` | Effective dates, delayed obligations, and transition rules |
| `cross_reference` | Resolve a cited article, numeral, or constitutional provision |
| `multi_document` | Synthesize evidence from at least two publications |
| `monitoring` | Retrieve by publication date, agency, and when relevant edition |
| `negative_false_premise` | Detect and correct a premise contradicted by the corpus |

## Validation

From the repository root:

```bash
uv run python scripts/validate_eval_v4.py
```

The validator checks:

1. exactly 42 unique questions and six per category;
2. the frozen corpus version, size, and date boundaries;
3. document id/path/date/section mappings;
4. chunk id/path/index mappings;
5. each quoted span against a freshly regenerated chunk;
6. answerability and multi-document invariants.

## Evaluation contract

V4 should not be reduced to one headline score. Report at least:

- document Recall@1/5/10/20;
- evidence-chunk Recall@1/5/10/20;
- all-hop recall for `multi_document`;
- MRR for the first relevant evidence chunk;
- context precision among the retrieved chunks;
- answer correctness and list completeness;
- citation precision and citation recall;
- premise-correction accuracy on the negative slice.

Report every metric by category. Tune fusion weights, reranking, and routing on
a future development split; keep a future expert-adjudicated test expansion
locked.

## Pilot limitations and next expansion

This version intentionally favors evidence integrity over breadth. It reuses a
small group of well-audited publications concerning labor, monetary indices,
water, national planning, and expropriation. Before treating v4 as a release
gate:

1. obtain independent legal/domain review of every answer and span;
2. add health, tax, customs, procurement, environmental, and social-program
   documents;
3. add more cross-document references and graded multi-gold relevance;
4. split by document family into development and locked test sets;
5. expand with production queries while preserving the frozen 42-question
   pilot for regression history.
