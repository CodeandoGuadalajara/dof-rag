# TensorFold tool-calling investigation handoff

Prepared 2026-09-30 for a fresh Pi session. Goal: diagnose and fix reliable
function/tool calls when the DOF agent uses Qwen3.8 Flash-Next served by TensorFold.
Do not promote the model or weaken tool validation as a first response.

## Current status

- Endpoint used: `http://192.168.1.117:8888/v1`; model id
  `Qwen3.8-Flash-Next`. It is the existing TensorFold service, not a test service
  this repo owns. Do not restart/stop it without permission.
- Recipe: <https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold>
- Local server command at test time: Vontra Qwen3.8-Flash-Next MLX 4-bit MTP,
  TensorFold, context 262144, four streams, int8 KV, MTP drafts 6, confidence .60,
  vision and thinking enabled by default. Evaluation explicitly disabled thinking.
- Test: `var/flash-next-tensorfold-2026-09-30-compat/flash-next/off-1.json`.
  42 questions completed; 29 stop reason `completed`, 25 provisionally supported
  after semantic review. Mean end-to-end latency 38.5 s.
- Review: `var/flash-next-tensorfold-2026-09-30-compat/answer-review.md` and
  `.tsv`. Strict initial run (first two questions; no bool-string compatibility)
  is preserved under `var/flash-next-tensorfold-2026-09-30/`.
- Exact runner and provenance: `var/run-flash-next-eval.py` and
  `var/flash-next-tensorfold-2026-09-30-compat/provenance.json`.
- Pilot scheduler and web were restored after the run. Flash-Next remained live.
  Verify live status before further tests; do not assume it is still running.
- Working tree has unrelated user files. Do not clean, stage, or include them.

## Observed failures (compatibility run)

Across 42 questions, there are 58 failed tool invocations:

| Tool | Schema/input validation errors | Runtime `ValueError` |
|---|---:|---:|
| `search_documents` | 46 | 1 |
| `search_evidence` | 7 | 0 |
| `list_publications` | 3 | 0 |
| `read_chunks` | 0 | 1 |
| **Total** | **56** | **2** |

Recurring validation patterns, with examples from traces:

1. **Required fields omitted.** E.g. `search_documents` omits `strategy` and is
   rejected with `'strategy' is a required property`; sometimes `section` too.
   A `list_publications` call omits `section`.
2. **Boolean/null encoded as strings.** E.g. `"prefer_recent": "true"` fails
   the declared `boolean|null` schema. The strict run shows this directly.
   In the 42-run compatibility arm, the current existing normalizer converted
   canonical string values (`"true"`/`"false"`, `"null"`/`"none"`) for nullable
   booleans/null. This one compatibility change is recorded in provenance;
   do not mistake that arm for a completely unmodified frozen agent.
3. **Invented/duplicate/renamed properties.** Examples: `limit` and `query_type`
   sent to `search_documents`; `query_2`, `query2`, `query_strategy`, `query_mode`,
   `prefer_recent_2`, `section_2`, `as_of_type`, and similar keys. Extra keys are
   correctly rejected (`additionalProperties: false`).
4. **Wrong value shapes/types.** `document_ids: "165057"` instead of an integer
   array; serialized string values for integer parameters; `chunk_ids: "[]"`.
5. **IDs not obtained from tools.** One `read_chunks` raised
   `chunk ids were not returned by an earlier tool: [631807]`.
6. **Bad filter despite validation.** One `search_documents` used a `date_to`
   after the question's `as_of`, correctly rejected by the run cutoff.

Error detail and the call arguments are preserved in every result's
`run.traces` in `off-1.json`. The `answer-review.json` also summarizes per-question
errors. No credentials are in these artifacts.

## Relevant agent contract

See `agent_tools/agent.py`:

- `_object_schema()` around line 466 marks **every property required** and sets
  `additionalProperties: false`.
- `_normalize_nullable_literals()` around line 475 normalizes only exact
  canonical string encodings for nullable values/booleans. It does not repair
  omitted fields, arbitrary extra fields, wrong array/scalar shapes, invalid
  dates, or unreturned IDs.
- `DofToolbox._build_schemas()` around line 680 declares all five tool schemas;
  all nullable properties must be present, with JSON null when unused because of
  the strict required-fields schema.
- Tool call validation/errors are returned to the model by `DofToolbox.call()`.
- The tool schemas are sent to the OpenAI-compatible backend with strict mode.

TensorFold's recipe says typed tool-call parameters are supported, and its
`tools/toolcheck.py` smoke test checks an array parameter. Our smoke test passed
one simple `search_documents` call with a string query. Neither test establishes
correct adherence to the agent's full strict schemas. The evidence so far shows
schema-invalid arguments at the agent boundary; it does **not** yet isolate whether
the defect is the model's tool-call generation, TensorFold's template/serialization,
or an interaction among model/runtime/schema. Avoid prematurely labelling it a
TensorFold-only bug.

## Next session: minimal diagnostic sequence

1. Preserve current branch/worktree and inspect status before touching files.
   Read this handoff, the evaluation provenance, relevant code above, and
   `run.traces` for representative failures. Never print or save API keys.
2. Reproduce against the running endpoint with a small matrix of schemas:
   - one required string only;
   - nullable boolean plus required string;
   - a strict full `search_documents` schema;
   - array-of-integer `document_ids` / `chunk_ids`.
   Repeat with TensorFold drafting enabled and `"draft": false` (serial reference)
   if the endpoint supports the flag. Use fixed prompt, seed and settings; capture
   raw Chat Completions request/response locally with secrets redacted.
3. Separate layers: inspect whether malformed values are in raw assistant
   `tool_calls[].function.arguments` before the OpenAI SDK, or introduced while
   parsing/normalizing the response. Record exact endpoint version/image digest,
   checkpoint revision and request JSON.
4. Fix the narrowest proven layer. Normalize only lossless canonical types;
   retain strict schema checks and clear errors for missing/extra fields, invalid
   IDs and out-of-cutoff dates. If schema shape is the trigger, test the same
   schema with another local backend before changing the production contract.
5. Add one runnable regression test for each demonstrated parser/adapter defect;
   unit-test accepted and rejected schemas. Run existing unit tests + Ruff.
6. Re-run a small relevant subset first, then all 42 with per-question
   checkpoints, one generation request at a time. Keep generation-model runs
   sequential; don't alter the embedding runtime. Automatically restore pilot
   services and verify Flash-Next is left untouched.
7. Review supported answers and latency after the fix. Completion count is not
   correctness. Record any compatibility shim as a comparison deviation and keep
   the strict-run artifacts.

## Example recorded malformed calls

- `search_documents`: `{..., "prefer_recent":"true", ...}`; `strategy` absent.
- `search_documents`: `{..., "limit":"20", "strategy":"hybrid", ...}`;
  `limit` is unknown (use `top_k`).
- `search_evidence`: `document_ids: "165057"`; schema requires an array of
  integers, and also requires `strategy`/`prefer_recent`.
- `list_publications`: missing required `section`, `date_from`/`date_to` on
  separate calls.

These are concrete symptoms, not an established root cause. The article at
<https://flowtivity.ai/blog/tensorfold-inference-engine-review/> reports an
array-parameter tool smoke test, not broad strict-schema conformance. Treat the
project's exact speculative decoding/output claim separately from tool-call
correctness.

## Root cause identified (2026-09-30 probe)

Probe script `var/flash-next-schema-probe.py` (results in
`var/flash-next-schema-probe-results.json`) sent fixed prompts at temperature 0
with `tool_choice: required` against the live endpoint, one request at a time,
with drafting on and off (`"draft": false`). Findings:

- The defect is **server-side, in raw
  `tool_calls[].function.arguments`** — before the OpenAI SDK or any agent
  parsing. TensorFold stringifies every value whose schema type is a union
  (nullable): `"flag":"true"`, `"note":"null"`, `"n":"42"`.
- Plain (non-union) types are emitted correctly: booleans, integers, strings,
  and arrays of integers (`{"chunk_ids":[101,202,303],"neighbor_window":1}`)
  all validated clean.
- `anyOf: [{type: T}, {type: null}]` instead of `type: [T, "null"]` does **not**
  help — still stringified.
- Drafting on/off makes no difference; speculative decoding is not involved.

Conclusion: TensorFold's tool-call grammar/serialization mishandles union-typed
fields. Fix applied at the narrowest layer available in this repo:
`_normalize_nullable_literals()` in `agent_tools/agent.py` now also coerces
canonical stringified integers/numbers for (nullable) integer/number fields,
alongside the existing boolean/null coercion. Coercion is lossless and exact
(`^-?\d+$` / `^-?\d+(\.\d+)?`); anything else remains a strict schema error.
Regression test: `test_tool_nullable_integer_strings_are_normalized` in
`tests/test_chat_runner.py`. The underlying server defect should be reported
upstream to TensorFold; the agent-side coercion can be removed once the server
emits typed values.
