# Running the Flash-Next (TensorFold) evaluation

How to re-run the 42-question tool-calling evaluation of Qwen3.8-Flash-Next
served by TensorFold. Companion docs:
`docs/tensorfold-tool-calling-investigation.md` (diagnosis and root cause).

## Prerequisites (verify, never restart)

- TensorFold endpoint live: `curl -s http://127.0.0.1:8888/v1/models`
  returns `Qwen3.8-Flash-Next`. Container `qwen38-flash-next-tf`.
  **Do not restart/stop it**; it is shared infrastructure.
- `llama-server` (llama.cpp embedding server binary) on PATH — it lives in
  `~/.local/bin`, which non-interactive shells may not include. Prefix runs
  with `PATH="$HOME/.local/bin:$PATH"` or the eval dies with
  `FileNotFoundError: 'llama-server'` after the smoke tests.
- Embedding model present: `~/dof-gguf/jina-v5-small-retrieval-F16.gguf`.
- Python: use the repo venv `.venv/bin/python` (system python lacks deps).

## Running

```bash
cd ~/Documents/dof/dof-rag
PATH="$HOME/.local/bin:$PATH" nohup .venv/bin/python var/run-flash-next-eval-post-fix.py \
  > var/flash-next-compat2-run.log 2>&1 & disown
tail -f var/flash-next-compat2-run.log
```

The runner (`var/run-flash-next-eval-post-fix.py`, derived from
`var/run-flash-next-eval.py` and `var/run-three-models-grounded.py`):

1. Loads the frozen agent snapshot `var/grounded-finalization-2026-09-14/agent.py.snapshot`
   and overrides only `_normalize_nullable_literals` with the live version —
   so the 42-run is comparable to prior arms except for the normalization shim.
2. Asserts source hashes match the 2026-09-14 comparison provenance.
3. Records full provenance (docker image, checkpoint ref, sampling, hashes)
   in the output dir's `provenance.json`.
4. Stops `dof-human-eval-web` and `dof-human-eval-scheduler` (pilot services)
   and takes the human-evaluation SQLite execution lock.
5. Runs all 42 questions sequentially, thinking off, temperature 1.0 /
   top_p 0.95 / top_k 20, checkpointing per question. ~30–45 min total.

## Output

- `var/flash-next-tensorfold-2026-09-30-compat2/` — provenance.json,
  tracked.patch, agent.py.snapshot, `flash-next/off-1.json` (per-question
  results with full `run.traces`), smoke-*.json.
- Prior arms, do not overwrite: `var/flash-next-tensorfold-2026-09-30`
  (strict, aborted), `...-compat` (first compat arm).
- Schema probes: `var/flash-next-schema-probe.py`, results in
  `var/flash-next-schema-probe-results.json`.

## myllmbox trial arm (2026-10-08)

Runner: `var/run-myllmbox-eval.py`. It imports the compat2 runner's frozen-agent setup and reuses the same 42-question evaluator, hashes, normalization shim, sampling and limits. Generation endpoint is `http://127.0.0.1:8000/v1`, model `Qwen/Qwen3.8-Flash-Next`, container `qwen38-flash-next`. Do not restart generation servers.

Output: `var/flash-next-myllmbox-2026-10-08-compat2/` (must not already exist); log: `var/myllmbox-compat2-run.log`. Provenance records the different checkpoint/runtime, recipe settings, image identity and download metadata. Results are historical comparisons, not identical-checkpoint trials.

The run launched as user systemd unit `dof-myllmbox-eval-20261008`, with `ExecStopPost` restoring both previously active pilot services even on failure or cancellation. The unit owns its embedding subprocess; the app's normal embedder returns when its scheduler is restored.

```bash
systemctl --user status dof-myllmbox-eval-20261008
tail -f var/myllmbox-compat2-run.log
# Cancel safely through systemd, not by killing an individual Python process:
systemctl --user stop dof-myllmbox-eval-20261008
systemctl --user is-active dof-human-eval-web dof-human-eval-scheduler
```

Completion counts require a separate correctness review, just like prior arms. Preserve existing results; use a new output directory for another trial.

## Gotchas

- **The output dir must not exist** (`mkdir(exist_ok=False)` at both the arm
  and OUT level). For a fresh run, `rm -rf` the whole compat2 dir (provenance
  is regenerated identically). A rerun after a crash mid-arm needs at least
  `rm -rf <OUT>/flash-next`.
- **Killing the runner bypasses pilot-service restore.** After a manual kill,
  run `systemctl --user start dof-human-eval-web dof-human-eval-scheduler`
  and kill any stray `llama-server --port 8086` the eval spawned (the app
  starts its own embedder).
- One generation request at a time; do not run other GPU jobs against the
  endpoint concurrently. Don't alter the embedding runtime.
- Correctness review (`answer-review`) is a separate manual step; completion
  count is not correctness. Record any normalization shim as a comparison
  deviation (it is, in provenance `compatibility_override`).
