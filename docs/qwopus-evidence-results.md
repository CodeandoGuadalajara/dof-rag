# Qwopus evidence discipline: changes and measured effects

## Conclusion

This change hardens reasoning/final-answer separation, preserves the ability to
look for missing evidence, and labels incomplete coverage explicitly. It is **not
a demonstrated overall answer-quality win** and does not justify enabling thinking
by default or opening the PoC to a public audience.

The strongest workload result is MO-001 (INPC + UMA): all six post-change runs
reported both sets of values with both gold chunks cited. The earlier single
thinking-off run falsely denied that UMA was published. Thinking had already fixed
that case in a pre-change diagnostic, so the improvement cannot be attributed to
this patch alone.

## What changed

- Strip closed and unfinished `<think>` blocks before final JSON parsing and
  persisted turn text. Keep extracted reasoning separately in the model's private
  conversation history. This avoids selecting draft JSON inside reasoning and
  prevents those blocks from appearing in the public result/progress payload.
- Propagate provider output truncation (`finish_reason=length`, or Responses
  `max_output_tokens`) as `output_token_limit`. Do not execute truncated tool calls
  or treat partial output as a complete answer. This stops the run rather than
  increasing its output budget silently.
- Keep targeted retrieval, outlines and publication listing available after
  reading evidence. Previously any satisfied set of heuristic coverage checks
  forced finalization and removed all tools, even if another passage was needed.
- Add INPC/UMA coverage anchors checked against **read text**, not just titles.
  These remain lexical heuristics, not a general semantic coverage validator.
- Ask the agent to track each requested item, check applicability/transitory
  clauses, and distinguish a failed search from evidence of nonexistence.
- When coverage remains incomplete at the final turn, mark the answer `unclear`
  and append the missing requirements. This warning does **not** remove or prove
  the model's preceding assertions; partial answers still require review.
- Add `--thinking on|off` to the local Chat Completions evaluation path; no change
  to the deployed model's default thinking setting.
- Correct NE-001 reference prose: full chunk 6721511 includes QUINTO. The decree
  has five substantive clauses and four transitory clauses, not four substantive
  clauses. Gold IDs and the false-premise label are unchanged.

## Experiment

DGX Spark / GB10, Qwopus3.8-27B-Flash NVFP4, SGLang with DFlash2 K=16:

- Target: `sojufx/Qwopus3.8-27B-Flash-NVFP4`, revision
  `892f62b41382aa34bd096b24843ea380beb98839`.
- Drafter: `maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal`, revision
  `bd7a934213c47a9e7ef69eef36bb3325f47fd1f1`.
- Recipe checkout `8b5f4e9`; missing overlay `models/dflash.py` restored from
  upstream SGLang `1cf2b8c54d81802abc15dcf23a29b9cc687bc01e`.
- One running request, 32K context, 0.50 static memory fraction, BF16 KV.
- 2,400 output tokens per turn, eight model turns and eight tool calls.
- Server-default sampling in both arms: temperature 1.0, top-k 20, top-p 0.95.
- Hybrid retrieval available; the agent chooses its actual search strategy.
- Five previously problematic questions, three repeats each per setting: 30
  local-only runs. Order: off/on, on/off, off/on. No paid provider fallback.
- Corpus is the locally updated corpus, not the original frozen snapshot;
  each question retains its eval-v4 date cutoff.

[Compact machine-readable results](../eval/results/qwopus-evidence-2026-09-06.json)
include every answer, stop reason, citation IDs, coverage, latency and token usage.
They omit private reasoning and full tool traces. Provenance records the base Git
revision and hashes of the exact modified agent and evaluation data tested.

## Results: post-change thinking off versus on

All latency and gold-overlap figures below include **all 15 runs in each arm**,
including failures. This differs from the existing CLI's completed-only summary.

| Metric | Thinking off | Thinking on |
|---|---:|---:|
| Completed under agent checks | 9/15 | 9/15 |
| Mean end-to-end latency | 38.3 s | 67.1 s |
| Median end-to-end latency | 42.7 s | 58.2 s |
| Range | 13.3–70.6 s | 23.8–116.5 s |
| Macro exact-gold citation precision | 52.8% | 61.9% |
| Macro exact-gold citation recall | 53.3% | 70.0% |

For each run, precision is gold/cited intersection divided by cited IDs (zero
when none); recall divides by gold IDs. These metrics are averaged over all runs.
Gold overlap does not establish semantic support, and extra non-gold citations
can still be relevant. Completion is **not** a correctness score.

| Case | Off: completed / 3 | On: completed / 3 | Review of observed behavior |
|---|---:|---:|---|
| LI-001 worker ranges | 1 | 0 | Still unreliable. Thinking misses the <=15 applicability evidence every time. The completed off answer also overstates other <=15 duties (all Chapter 8 rather than 8.1/8.2). |
| TE-001 effective dates | 2 | 3 | Completed answers now cite the actual transitory clause and give the dates/deferred list. |
| CR-001 cross-reference | 2 | 1 | Supported completed answers, but thinking also hits one output-token limit and one citation-required failure. |
| MO-001 INPC + UMA | 3 | 3 | Both values and both supporting gold chunks found in all six runs. |
| NE-001 nonexistent article | 1 | 2 | The completed off run merely abstains; the two completed on runs explicitly refute article 99 with the correct gold evidence. |

Thinking costs approximately **75% more mean latency**, with no improvement in
completion rate in this sample. It improves gold-evidence overlap and some cases,
but is not uniformly better. The earlier pre-change thinking diagnostic completed
4/5 once; that apparent reliability did not hold in these repeated trials.

## What can and cannot be claimed

The parser/privacy, truncation, tool-availability and partial-labeling changes
have deterministic regression tests. These are the clearest demonstrated effects.

The workload comparison tests **thinking off versus on after the patch**. It is
not a balanced pre/post experiment: the earlier baseline was one trial per case,
the prompts/tool policy/parsing changed together, and cache state was not reset.
The five cases were selected because they failed; they are a development slice,
not a held-out estimate of general accuracy. There is no statistical significance
claim or independent legal adjudication.

## Validation and follow-up

- Full unit suite: 202 tests exercised, one skipped, no failures.
- Ruff checks and `git diff --check` pass.
- Dataset validator: 15 existing failures, identical with original and corrected
  reference data. These include corpus growth beyond the frozen snapshot and
  quoted spans that do not match regenerated chunks. No new validator failures
  from the NE-001 prose correction; this is not a clean frozen-corpus benchmark.

Do not enable thinking by default from these numbers. Next isolate the <=15
applicability retrieval failure and repeated final-format/citation failures, then
run a balanced pre/post comparison with the same model settings and corpus.

### Reproduce an arm

Stop web admission and the scheduler when sharing their model/embedding server;
hold the scheduler execution lock for the experiment. Restore services afterward.
Use a dedicated test environment without paid-provider credentials.

```bash
.venv/bin/python scripts/eval_v4_agent.py \
  --provider llama-server --model ornith \
  --base-url http://127.0.0.1:8001/v1 --reasoning-effort '' \
  --thinking off --ids CR-001,LI-001,MO-001,NE-001,TE-001 \
  --gguf /path/to/jina-v5-small-retrieval-F16.gguf \
  --output var/off-1.json
```

Repeat three times per setting with distinct output paths; alternate arm order.
Keep failed runs rather than reporting only the successful subset.
