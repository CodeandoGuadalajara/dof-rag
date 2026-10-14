# All-Qwen model selection results — 2026-10-14

**Purpose:** durable record of the completed all-Qwen (plus MiniCPM-V4.7 and LFM2.5) serving-configuration selection on the DGX Spark. When a new candidate model or serving configuration appears, rerun the same frozen DOF 42-question evaluation + profile using the runbook below and compare against this table.

- **Machine-readable full index (33 arms):** `docs/all-qwen-model-selection-results-2026-10-14.json` (copy of `var/model-selection-2026-10-10/reviewed-arms-comparison-interim.json`; the `var/model-selection-2026-10-10/` tree holds per-arm answer reviews, review packs, wire/integrity proofs, profiles and restoration proofs).
- **Plan/runbook:** `docs/qwen-model-selection.md`
- **Final recommendation + rollback:** `var/model-selection-2026-10-10/final-model-selection-recommendation.md`
- **Requirement audit:** `var/model-selection-2026-10-10/goal-completion-audit.md`

**How to add a new candidate:** sequential launch only (never compete with the live model); freeze inputs/helpers; fresh 42-question arm with temp1/topP0.95/topK20/seed1234/2400-token/eight-turn budgets; own-passage correctness review of all 42; matched text profile (21-row benchmark + Pi + images if supported); wire/integrity proof; automatic native-FP8 restore verified BEFORE pilots. Counts below are passage-reviewed supported answers, not completions.

## Ranked results (of 42 questions, reviewed)

| # | Arm | Supported | Completed | Mean s | E2E tok/s* |
|---:|---|---:|---:|---:|---:|
| 1 | Stock Qwen3.8 (27B NVFP4+DFlash2) corrected | 38 | 40 | 63.5 | 12.0 |
| 2 | Stock Qwen3.8 corrected r2 | 37 | 38 | 54.1 | 11.4 |
| 3 | Stock Qwen3.8 baseline | 37 | 40 | 58.3 | 11.4 |
| 4 | GSQ corrected | 36 | 39 | 76.0 | 8.9 |
| 5 | GSQ baseline | 35 | 38 | 75.5 | 9.0 |
| 6 | Myllmbox corrected JSON-final | 35 | 40 | 145.3 | 10.4 |
| 7 | Qwopus baseline | 33 | 36 | 41.8 | 14.3 |
| 8 | TensorFold BF16 corrected r1 | 33 | 39 | 33.4 | 26.4 |
| 9 | TensorFold BF16 corrected r2 | 33 | 39 | 34.2 | 25.8 |
| 10 | Qwen3.6 historical026 | 32 | 36 | 70.4 | 13.9 |
| 11 | Qwen3.6 baseline | 28 | 37 | 20.2 | 47.7 |
| 12 | Qwen3.6 context64 | 29 | 37 | 20.2 | 44.9 |
| 13 | Qwen3.6 production262 | 30 | 36 | 20.4 | 44.8 |
| 14 | Qwen3.6 corrected r1 | 29 | 36 | 22.4 | 44.9 |
| 15 | Qwen3.6 historical0271-r2 | 29 | 35 | 96.5 | 11.1 |
| 16 | Qwen3.6 historical0272-r1 | 29 | 35 | 100.1 | 11.6 |
| 17 | TensorFold FP8 corrected-final r2 (**default**) | 31 | 37 | 31.6 | 28.5 |
| 18 | TensorFold FP8 corrected-final r1 | 31 | 37 | 33.0 | 27.3 |
| 19 | Qwopus JSON-final | 31 | 36 | 41.5 | 13.9 |
| 20 | Stock Qwen3.8 thinking-on | 31 | 35 | 117.7 | 22.9 |
| 21 | Qwopus production80 | 29 | 37 | 41.6 | 14.2 |
| 22 | Qwopus production65 | 29 | 32 | 45.1 | 14.8 |
| 23 | TensorFold FP8 read-only-final | 29 | 36 | 28.9 | 31.2 |
| 24 | Qwen3.6 corrected r0 | 25 | 37 | 21.0 | 44.4 |
| 25 | MiniCPM-V4.7 initial (unmatched sampling†) | 27 | 35 | 25.1 | 30.5 |
| 26 | MiniCPM-V4.7 matched r1 | 26 | 32 | 24.5 | 31.3 |
| 27 | Myllmbox JSON-final | 26 | 36 | 132.6 | 10.7 |
| 28 | Old TensorFold 0.3.6.3 | 25 | 33 | 37.1 | 26.7 |
| 29 | Qwopus corrected | 22 | 26 | 42.8 | 13.6 |
| 30 | TensorFold FP8 baseline | 22 | 25 | 31.3 | 27.0 |
| 31 | LFM2.5 bounded256+JSON-final (tuned‡) | 22 | 33 | 41.8 | 56.4‡ |
| 32 | LFM2.5 baseline r1 | 14 | 19 | 62.8 | 74.5‡ |
| 33 | Myllmbox baseline | 23 | 28 | 126.6 | 10.7 |

\* End-to-end effective output throughput (retrieval/tools/prefill/reasoning included) — not decode TPS. † Implicit min_p=0.05, superseded by matched r1. ‡ Reasoning-heavy; tuned arm's jump is confounded/unrepeated; license limits use >$10M revenue.

## Decision and next step

- **Default (unchanged):** TensorFold 1.0 native FP8 (`azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`, 4×262K seats, MTP, vision). Corrected-final arm: 31/37 supported, 31.6s mean, min ~52GiB headroom.
- **Highest observed potential on this exact stack:** TensorFold arms — baseline 22 → premise-correction helpers alone 31 → +read-only-final 29 @ 28.9s (fastest TF arm). The remaining gap to stock's 38 is harness-side (finalization/serialization/evidence handling), so **harness improvement is the agreed next step**, with a fresh full TensorFold arm rerun before any score claim.
- Correctness-first alternative if ever needed: stock Qwen3.8 corrected (38/37). Excluded by user: Unsloth27B-MTP3, Radix52d1+Z-Lab50307/K8, Radix52d1+Maurienne/K8.
