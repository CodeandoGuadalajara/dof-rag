"""Fresh 42-question arm, reusing the frozen compat2 harness with immutable inputs.

Run under a user systemd unit whose ExecStopPost restores previously active pilot
services. Never starts/stops a generation server. HTTP captures omit auth headers.
"""

import argparse
import hashlib
import inspect
import json
import os
import runpy
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import httpx
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_bundle(bundle, manifest):
    for name, record in manifest["databases"].items():
        assert sha256(bundle / name) == record["sha256"], (
            f"Input database changed: {name}"
        )
    for name, digest in manifest["sources"].items():
        assert sha256(bundle / "sources" / name) == digest, (
            f"Frozen source changed: {name}"
        )


def redact_arguments(arguments):
    result = list(arguments)
    for index, value in enumerate(result):
        if value in ("--api-key", "--api_key") and index + 1 < len(result):
            result[index + 1] = "<redacted>"
        elif value.startswith(("--api-key=", "--api_key=")):
            result[index] = value.split("=", 1)[0] + "=<redacted>"
    return result


def runtime_info(container, binary):
    if container:
        info = json.loads(subprocess.check_output(["docker", "inspect", container]))[0]
        assert info["State"]["Running"], f"{container} is not running"
        return {
            "container": container,
            "image": info["Image"],
            "image_tag": info["Config"]["Image"],
            "command": redact_arguments(info["Config"]["Cmd"]),
            "restart_count": info["RestartCount"],
        }
    if binary:
        return {
            "binary": str(binary.resolve()),
            "sha256": sha256(binary),
            "version": subprocess.check_output(
                [str(binary), "--version"], stderr=subprocess.STDOUT, text=True
            ).strip(),
        }
    raise ValueError("Specify --container or --server-binary for runtime evidence")


def all_run_summary(data):
    runs = [item["run"] for item in data["results"] if "run" in item]
    seconds = sum(run["elapsed_ms"] for run in runs) / 1000
    tokens = sum(run["usage"].get("output_tokens", 0) for run in runs)
    return {
        "outcomes": len(data["results"]),
        "timed_runs": len(runs),
        "untimed_outcomes": len(data["results"]) - len(runs),
        "completed": sum(run["stop_reason"] == "completed" for run in runs),
        "stop_reasons": dict(Counter(run["stop_reason"] for run in runs)),
        "all_trace_tool_errors": sum(
            not trace["output"].get("ok", False)
            for run in runs
            for trace in run["traces"]
        ),
        "total_output_tokens": tokens,
        "total_elapsed_s": seconds,
        "all_run_mean_elapsed_s": seconds / len(runs) if runs else None,
        "effective_output_tokens_per_s": tokens / seconds if seconds else None,
        "correctness": "pending passage-based review",
        "timing_caveat": (
            "Effective output throughput includes retrieval, tools, prefill and returned "
            "failures; not decode TPS. Thrown backend-error outcomes lack frozen "
            "elapsed/usage/traces and are excluded; see untimed_outcomes and raw HTTP logs."
        ),
    }


def capture_chat_request(request, output):
    """Record prepared transport attempts, not proof of server receipt; omit headers."""
    if request.url.path.endswith("/chat/completions"):
        output.write(json.dumps({"time": time.time(), "request": json.loads(request.content)}, ensure_ascii=False) + "\n")
        output.flush()


def sample_memory(output, stopped):
    with output.open("x") as stream:
        while not stopped.is_set():
            sample = {"time": time.time()}
            try:
                sample["meminfo_kib"] = {
                    key.rstrip(":"): int(value.split()[0])
                    for key, value in (
                        line.split(":", 1)
                        for line in Path("/proc/meminfo").read_text().splitlines()
                    )
                    if key
                    in (
                        "MemTotal",
                        "MemFree",
                        "MemAvailable",
                        "SwapTotal",
                        "SwapFree",
                        "Cached",
                    )
                }
                sample["pressure"] = {
                    name: Path(f"/proc/pressure/{name}").read_text().strip()
                    for name in ("cpu", "memory", "io")
                }
                sample["gpu"] = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=temperature.gpu,utilization.gpu,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=5,
                ).strip()
            except Exception as exc:
                sample["error"] = f"{type(exc).__name__}: {exc}"
            stream.write(json.dumps(sample) + "\n")
            stream.flush()
            stopped.wait(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--container")
    parser.add_argument("--server-binary", type=Path)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--startup-max-tokens", type=int, choices=(512, 2400), default=512,
                        help="Startup chat/tool smokes only; frozen evaluation remains 2400")
    parser.add_argument("--thinking", choices=("off", "on"), default="off")
    parser.add_argument(
        "--finalization", choices=("original", "read-only-final", "json-object"), default="original"
    )
    parser.add_argument(
        "--premise-correction",
        action="store_true",
        help="Explicit separately-proven correction-form guard fixes",
    )
    parser.add_argument("--notes", required=True)
    args = parser.parse_args()
    assert os.environ.get("INVOCATION_ID"), (
        "Use systemd with pilot-service ExecStopPost restoration"
    )
    parsed = urlparse(args.base_url)
    assert parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost"), (
        "Local endpoints only"
    )
    assert parsed.port and parsed.path.rstrip("/") == "/v1"
    os.chdir(ROOT)
    bundle = args.inputs.resolve()
    output = args.output.resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    verify_bundle(bundle, manifest)
    sys.path.insert(0, str(ROOT))
    import agent_tools.agent as live_agent

    premise_helpers = {
        name: getattr(live_agent, name)
        for name in (
            "_has_affirmative_premise_correction",
            "_has_explicit_premise_correction",
        )
    }
    frozen = runpy.run_path(str(ROOT / "var/run-flash-next-eval-post-fix.py"))
    h = frozen["h"]
    if args.premise_correction:
        for name, function in premise_helpers.items():
            setattr(frozen["agent"], name, function)
    normalizer_source = inspect.getsource(frozen["normalize"])
    assert (
        hashlib.sha256(normalizer_source.encode()).hexdigest()
        == manifest["normalizer_sha256"]
    )
    live_hashes = h.hashes

    def evaluated_hashes():
        values = live_hashes()
        values["agent_tools/agent.py"] = sha256(frozen["SNAPSHOT"])
        return values

    h.hashes = evaluated_hashes
    h.EXPECTED = h.hashes()
    previous = json.loads(
        (ROOT / "var/grounded-finalization-2026-09-14/provenance.json").read_text()
    )
    assert h.EXPECTED == previous["hashes"], (
        "Evaluated sources differ from comparison provenance"
    )
    assert (
        h.EXPECTED["agent_tools/agent.py"]
        == manifest["sources"]["var/grounded-finalization-2026-09-14/agent.py.snapshot"]
    )
    for name in (
        "agent_tools/retrieval.py",
        "scripts/eval_v4_agent.py",
        "eval/dof_queries_v4.jsonl",
    ):
        assert h.EXPECTED[name] == manifest["sources"][name], (
            f"Source differs from bundle: {name}"
        )
    key = args.key_file.read_text().strip() if args.key_file else "local-test-only"
    with OpenAI(base_url=args.base_url, api_key=key, timeout=5) as client:
        assert args.model in [model.id for model in client.models.list().data]
    runtime = runtime_info(args.container, args.server_binary)
    transform = None
    if args.finalization == "read-only-final":
        from scripts.replay_qwen_finalization import compact_read_final_messages

        transform = compact_read_final_messages
    output.mkdir(parents=True, exist_ok=False)
    provenance = {
        "label": args.label,
        "model": args.model,
        "endpoint": args.base_url,
        "runtime": runtime,
        "notes": args.notes,
        "runner_sha256": sha256(Path(__file__)),
        "finalization": args.finalization,
        "premise_correction_override": {
            name: inspect.getsource(function)
            for name, function in premise_helpers.items()
        }
        if args.premise_correction
        else None,
        "premise_patterns": {
            "numeric_contrast": live_agent.NUMERIC_PREMISE_CONTRAST_RE.pattern,
            "affirmative_assertion": live_agent.CORRECTION_ASSERTION_RE.pattern,
            "failed_search": live_agent.SEARCH_FAILURE_RE.pattern,
        }
        if args.premise_correction
        else None,
        "final_input_transform": {
            "source": inspect.getsource(transform),
            "sha256": hashlib.sha256(inspect.getsource(transform).encode()).hexdigest(),
        }
        if transform
        else None,
        "source_hashes": h.EXPECTED,
        "live_checkout_hashes": live_hashes(),
        "inputs": str(bundle),
        "input_manifest": manifest,
        "compatibility_override": {
            "function": "_normalize_nullable_literals",
            "source": normalizer_source,
            "sha256": manifest["normalizer_sha256"],
        },
        "selection": "All 42 sequential; frozen agent; one trial",
        "sampling": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "seed": args.seed,
            "thinking": args.thinking,
        },
        "startup_max_tokens": args.startup_max_tokens,
        "limits": {"model_turns": 8, "tool_calls": 8, "output_per_turn": 2400},
        "deviations": [
            "Fixed seed per model request; historical arms did not explicitly seed.",
            "Corpus/index and questions pinned to the immutable input bundle.",
            "Different checkpoints/runtime/context/KV precision are explicitly recorded; not isolated intelligence.",
            "External endpoint users cannot be excluded; local pilot is paused.",
            "Prepared chat transport attempts additionally logged before responses, including timeouts; "
            "not proof of server receipt. Response capture and request payloads unchanged; logging adds I/O overhead.",
        ],
    }
    if args.premise_correction:
        provenance["deviations"].append(
            "Proven correction-form false-negative fixes recognize numeric no es/son X sino Y, structured-body/composition alternatives and explicit fija/fijó/expropia/expropió assertions; "
            "citation, type, evidence and unsuccessful-search rejection remain unchanged. Separately labelled arm."
        )
    if transform:
        provenance["deviations"].append(
            "Experimental final input replaces tool history with successful read payloads already sent; "
            "system/question/final constraints, call/turn/output limits and answer validation are unchanged. "
            "This is a separately labelled improved-harness arm, not the original baseline."
        )
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "tracked.patch").write_bytes(
        subprocess.check_output(["git", "diff", "HEAD"])
    )
    (output / "agent.py.snapshot").write_bytes(frozen["SNAPSHOT"].read_bytes())
    (output / "runner.py.snapshot").write_bytes(Path(__file__).read_bytes())
    # Capture raw JSON bodies, never authorization headers or environment secrets.
    http_log = (output / "http.jsonl").open("x")
    request_log = (output / "http-requests.jsonl").open("x")
    transform_log = (
        (output / "final-input-transform.jsonl").open("x") if transform else None
    )
    original_main = h.evaluation.main

    def capture_response(response):
        response.read()
        if response.request.url.path.endswith("/chat/completions"):
            try:
                body = json.loads(response.content)
            except json.JSONDecodeError:
                body = {"non_json_body": response.text}
            record = {
                "time": time.time(),
                "status": response.status_code,
                "request_elapsed_s": response.elapsed.total_seconds(),
                "request": json.loads(response.request.content),
                "response": body,
            }
            http_log.write(json.dumps(record, ensure_ascii=False) + "\n")
            http_log.flush()

    def local_client(**kwargs):
        client = OpenAI(
            **kwargs,
            http_client=httpx.Client(event_hooks={
                "request": [lambda request: capture_chat_request(request, request_log)],
                "response": [capture_response],
            }),
        )
        original_create = client.chat.completions.create

        def seeded_create(**request):
            if request.get("max_tokens") == 512 and args.startup_max_tokens != 512:
                request["max_tokens"] = args.startup_max_tokens
                request.setdefault("extra_body", {}).setdefault("chat_template_kwargs", {})[
                    "enable_thinking"
                ] = args.thinking == "on"
            request.setdefault("seed", args.seed)
            if args.finalization == "json-object" and request.get("tool_choice") == "none":
                request["response_format"] = {"type": "json_object"}
            if transform and request.get("tool_choice") == "none":
                original_messages = request["messages"]
                new_messages = transform(original_messages)
                transform_log.write(
                    json.dumps(
                        {
                            "original_messages": original_messages,
                            "sent_messages": new_messages,
                            "changed": original_messages != new_messages,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                transform_log.flush()
                request["messages"] = new_messages
            return original_create(**request)

        client.chat.completions.create = seeded_create
        return client

    def frozen_input_main():
        sys.argv.extend(
            [
                "--queries",
                str(bundle / "sources/eval/dof_queries_v4.jsonl"),
                "--corpus-db",
                str(bundle / "dof_corpus_l3.sqlite"),
                "--chunks-db",
                str(bundle / "dof_chunks.sqlite"),
                "--vec0-db",
                str(bundle / "dof_vec0_jina_binary.sqlite"),
            ]
        )
        if args.thinking == "on":
            # Last argparse value wins; preserve the existing harness and all other limits.
            sys.argv.extend(["--thinking", "on"])
        return original_main()

    h.OpenAI = local_client
    h.evaluation.main = frozen_input_main
    h.OUT = output
    h.BACKEND = h.evaluation.OpenAIChatCompletionsBackend
    h.command(
        "systemctl", "--user", "stop", "dof-human-eval-web", "dof-human-eval-scheduler"
    )
    lock = h.acquire_execution_lock(ROOT / "var/human_evaluation.sqlite")
    stopped = threading.Event()
    sampler = threading.Thread(
        target=sample_memory, args=(output / "memory.jsonl", stopped), daemon=True
    )
    sampler.start()
    try:
        h.evaluate(args.label, args.model, parsed.port, key)
        data = json.loads((output / args.label / "off-1.json").read_text())
        assert len(data["results"]) == 42
        assert {item["id"] for item in data["results"]} == {
            json.loads(line)["id"]
            for line in (bundle / "sources/eval/dof_queries_v4.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        }
        verify_bundle(bundle, manifest)
        summary = all_run_summary(data)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        stopped.set()
        sampler.join(timeout=10)
        http_log.close()
        request_log.close()
        if transform_log:
            transform_log.close()
        os.close(lock)


if __name__ == "__main__":
    main()
