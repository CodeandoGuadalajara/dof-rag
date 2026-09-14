# Three local DOF agent stacks: full-suite results and answer review

Reviewed 2026-09-07. RadixArk Qwen3.8-27B NVFP4 + Pangoleen DFlash2 stack,
thinking off, one 42-question eval-v4 run. Production was not changed.

## Result

| Outcome | Questions |
|---|---:|
| Supported answer, no disqualifying issue found | **29/42 (69.0%)** |
| Partial, ambiguous, or correct core with citation/extra-claim problems | 8 |
| Incorrect requested answer | 1 |
| No usable final answer | 4 |

Separately, **34/42 (81.0%) have a correct supported core response**. Five of
these lose the stricter overall label because of additional wording, metadata,
location or citation problems. Another three responses have incomplete or
contradictory core content. Neither 34 nor 29 should be confused with the
mechanical completion count of 37/42.

## Method

Assistant-led review of all 42 final outcomes against the reference answers,
all 42 distinct cited source chunks read in full, and selected complete tool
sequences for suspected errors and failures. Checked exchange-rate arithmetic,
180-day elapsed-date arithmetic, and the sum of all 14 property areas. Every
final cited ID was also checked to have been read within that question's run.
A valid/read citation does not by itself establish semantic correctness.

This is not independent human legal adjudication, a fresh official-site audit,
or a verification of current legal force. Search-scoped statements that nothing
later was found are not credited as proof of exhaustive absence. Historical
chronology answers are credited for the requested dates, not a warranty that no
later amendment exists. The June 10 calculation is an elapsed-day computation;
legal deadline counting still requires the applicable procedural convention.

A non-gold citation can support an answer. Conversely, an extra true assertion
can still lack a supporting final citation. Minor typos and stylistic differences
are not penalized; misleading dates, source editions and property locations are.

The dataset has known corpus/quotation issues and this is one stochastic run,
not an estimate of population accuracy. Full per-question notes, final answers,
reference answers, stop reasons and original-result SHA256 are in
[`local-models-full-suite-2026-09-07.json`](../eval/results/local-models-full-suite-2026-09-07.json).
This public export includes all 126 outcomes and the 62 distinct cited passages
across the three stacks. Raw local results are unchanged. Raw tool traces,
launchers, environments, endpoints and provider error payloads are deliberately
excluded. No paid judge API was used.

## What failed or needs correction

| ID | Finding / correction |
|---|---|
| SP-002 | Wrong exchange-rate date: answers **10.9113**, obtained Aug 8. Requested Aug 9 value is **10.8386**, published Aug 10. Source 1341480 itself says Aug 8; correct source is 1342011. |
| LI-001 | Leaves workplaces with <=15 workers unanswered. Scope passage 4733252 excludes duties 5.2/5.3 for that group. It correctly avoids inventing a duty, but does not finish the question. |
| CR-003 | Correct compensation-only procedure in an alternate Oct 2025 decree, but omits the ten-business-day deadline and asserts an unsupported equivalent procedure for the El Cafetal agrarian decree. The question also needs a decree date/ID; alternate-source mismatch alone is not an error. |
| MD-002 | No final answer: 2025 UMA evidence was found late but never read/cited alongside 2026. |
| MD-004 | No final answer: spends turns on older expropriations, reaches target late, then invalid final JSON/turn limit. |
| MD-005 | No final answer: repeatedly searches **Programa**, not **Plan**, Nacional de Desarrollo and never establishes both approval decrees. |
| MO-002 | All requested values correct, but says the rate was obtained on publication day while also correctly distinguishing Aug 9 determination and Aug 10 publication. Remove the contradictory phrase. |
| MO-006 | Correct decree facts; invents “Materia Administrativa” from `MAT`. It means **matutina**, the morning edition. |
| NE-001 | Correctly rejects article 99 and counts five clauses/four transitories, but invents “Materia Legislativa” as the edition. Again, `MAT` means matutina. |
| NE-003 | No correction: invents 25-day search terms and restricts dates before the Dec 27 publication, then fails citation validation. |
| NE-004 | Gives the correct Feb 1 UMA commencement, but explicitly says the Jan 1 premise is not documented as false. Contradictory; it should reject that premise plainly. |
| NE-005 | Core correction and article 570 explanation correct. Extra article 90 inflation claim is supported by read neighbor 6632602, but that chunk is not cited. Add it or omit the extra claim. |
| NE-006 | Both false figures corrected and all 14 polygon IDs/areas match. However, appends Playa del Carmen to Tulum locations: the source's parenthesis identifies the folio registration location, not the municipality. Remove it from the municipality label. |

## Supported-answer counts by category

| Category | Strict supported |
|---|---:|
| Single passage | 5/6 |
| List enumeration | 5/6 |
| Temporal/transitory | 6/6 |
| Cross-reference | 5/6 |
| Multi-document | 3/6 |
| Monitoring | 4/6 |
| False premise | 1/6 |

The false-premise strict result does not mean five invented core answers: three
have correct corrections with extra-claim/citation issues, one is contradictory,
and one yields no answer. The two scoring axes make that distinction explicit.

## Comparison and interpretation

| Stack | Provisional supported answers | Mean recorded latency |
|---|---:|---:|
| Qwopus | 24/42 | 36.6 s |
| Qwen3.6 | 21/42 | 22.2 s |
| Qwen3.8 | **29/42** | **62.3 s** |

Earlier models' numbers come from the prior reference/selected-citation review,
not a new blinded claim-by-claim re-adjudication in this pass. Their core-only
scores were not recorded, so do not compare Qwen3.8's 34 to their 24/21.
Qwopus timing excludes its one untimed provider exception; other recorded
incomplete runs are included. Different runtimes/quantizations mean this is a
stack comparison, not an isolated model-intelligence experiment.

Qwen3.8 is the strongest candidate from this run, not a proven production winner.
It correctly handles the exchange-rate comparison MD-003 despite failing the
same fact alone in SP-002, fixes the Qwen3.6 INPC-period error, reads both water
law object chunks, and handles the PND approval question that both older runs
failed. Multi-document discovery and needless extra assertions remain weaknesses.
Keep Qwopus live until the quality/latency trade-off is accepted; no model switch
was made during this review.

## Shared configuration and audit check

All runs used agent revision `03bf11568fa02a4d0751a7abf4bede6991f2f96d`,
identical agent/query hashes, thinking off, 32768 context tokens, 2400 output
tokens per turn, at most eight model turns and eight tool calls, temperature
1.0, top-p .95 and top-k 20. Hybrid retrieval was available; the agent chose
its actual strategy. One generation request ran at a time on a shared DGX Spark.

| Stack | Target | Runtime / drafter | KV cache |
|---|---|---|---|
| Qwopus | sojufx/Qwopus3.8-27B-Flash-NVFP4 | SGLang / DFlash2 K=16 | BF16 |
| Qwen3.6 | unsloth/Qwen3.6-35B-A3B-NVFP4 | vLLM / DSpark K=8 | FP8 |
| Qwen3.8 | RadixArk/Qwen3.8-27B-NVFP4 | SGLang / DFlash2 K=16 | BF16 |

Run from the repository root to check export counts and citation referential
integrity (this checks the artifact, not semantic correctness):

```sh
python3 - <<'PY'
import json
from collections import Counter
from pathlib import Path
p = Path('eval/results/local-models-full-suite-2026-09-07.json')
d = json.loads(p.read_text())
assert len(d['results']) == 126
for model, expected in [('qwopus', 24), ('qwen36', 21), ('qwen38', 29)]:
    rows = [r for r in d['results'] if r['model'] == model]
    assert len(rows) == len({r['id'] for r in rows}) == 42
    counts = Counter(r['verdict'] for r in rows)
    assert counts['supported'] == expected
    assert dict(counts) == d['models'][model]['review_counts']
    for row in rows:
        for citation in (row['answer'] or {}).get('citations', []):
            assert str(citation) in d['cited_passages']
print('126 outcomes; review totals and citation references verified')
PY
```
