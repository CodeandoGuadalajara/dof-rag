"""Matched streaming benchmark, with raw chunks, fixed fixtures/seeds and explicit clocks.

First prepare shared fixtures using the pinned dense-27B tokenizer. Then run one
candidate at a time while pilot services are paused; never during a DOF eval.
"""

import argparse
import hashlib
import importlib.util
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = "Write a Python quicksort function with a docstring, edge-case handling and executable assertions."
CHAT = (
    "Explain why the sky is blue and sunsets are red, in three informative paragraphs."
)
TASK = "\n\nWrite a Python summarise_run(rows) function computing median, minimum and maximum decode_tok_s per label. Include a docstring and assertions."


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(output, tokenizer_path):
    from tokenizers import Tokenizer

    spec = importlib.util.spec_from_file_location(
        "ctxsweep", ROOT.parent / "qwen38-spark/bench/ctxsweep.py"
    )
    sweep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokens = tokenizer.encode(sweep.seed_corpus(repeat=30)).ids
    assert len(tokens) > 28672
    corpus = {
        str(length): tokenizer.decode(tokens[:length], skip_special_tokens=False)
        for length in (1024, 8192, 28672)
    }
    data = {
        "tokenizer": str(tokenizer_path.resolve()),
        "tokenizer_sha256": sha256(tokenizer_path),
        "corpus_source_sha256": sha256(ROOT.parent / "qwen38-spark/bench/ctxsweep.py"),
        "corpus": corpus,
        "code": CODE,
        "chat": CHAT,
        "task": TASK,
        "limitations": [
            "Lengths are raw-text counts on the pinned dense-27B tokenizer; "
            "actual rendered counts come from each server's usage.",
            "Identical texts, not forcibly identical token counts across different tokenizers.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(data, stream, indent=2)
    print(f"Shared benchmark fixtures: {output}", flush=True)


def stream_request(base_url, key, body):
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer " + key},
    )
    started_at_unix_s = time.time()
    start, first = time.perf_counter(), None
    chunks, answer, reasoning, usage, stats, done = [], "", "", {}, {}, False
    with urllib.request.urlopen(request, timeout=900) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            elapsed = time.perf_counter() - start
            if payload == "[DONE]":
                done = True
                break
            chunk = json.loads(payload)
            if chunk.get("error") or chunk.get("object") == "error":
                raise RuntimeError(f"Streaming server error: {chunk}")
            chunks.append({"elapsed_s": elapsed, "chunk": chunk})
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                content = delta.get("content") or ""
                thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if first is None and (content or thought):
                    first = elapsed
                answer += content
                reasoning += thought
            usage = chunk.get("usage") or usage
            stats = chunk.get("tensorfold") or stats
    wall = time.perf_counter() - start
    assert done and first is not None, "Stream ended without DONE or output"
    assert usage.get("completion_tokens", 0) >= 8, (
        "Insufficient completion usage for timing"
    )
    decode_interval = wall - first
    assert decode_interval > 0 and len(chunks) >= 3, (
        "Buffered/too-short stream cannot estimate decode"
    )
    return {
        "started_at_unix_s": started_at_unix_s,
        "finished_at_unix_s": time.time(),
        "wall_s": wall,
        "ttft_s": first,
        "usage": usage,
        "server_stats": stats,
        "client_decode_tokens_per_s": usage["completion_tokens"] / decode_interval,
        "client_prefill_proxy_tokens_per_s": usage["prompt_tokens"] / first,
        "answer": answer,
        "reasoning": reasoning,
        "chunks": chunks,
        "output_sha256": hashlib.sha256((reasoning + answer).encode()).hexdigest(),
    }


def summaries(results):
    output = []
    for case in sorted({item["case"] for item in results}):
        rows = [item for item in results if item["case"] == case and "response" in item]
        if not rows:
            output.append({"case": case, "successful_requests": 0})
            continue
        responses = [row["response"] for row in rows]
        output.append(
            {
                "case": case,
                "successful_requests": len(rows),
                "median_ttft_s": statistics.median(r["ttft_s"] for r in responses),
                "median_client_decode_tokens_per_s": statistics.median(
                    r["client_decode_tokens_per_s"] for r in responses
                ),
                "median_rendered_prompt_tokens": statistics.median(
                    r["usage"]["prompt_tokens"] for r in responses
                ),
                "median_output_tokens": statistics.median(
                    r["usage"]["completion_tokens"] for r in responses
                ),
                "median_prefill_proxy_tokens_per_s": statistics.median(
                    r["client_prefill_proxy_tokens_per_s"] for r in responses
                ),
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--thinking", choices=("off", "on"), default="off")
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.prepare:
        if args.tokenizer is None:
            parser.error("--prepare requires --tokenizer")
        prepare(args.fixtures, args.tokenizer)
        return
    if not all((args.base_url, args.model, args.output)):
        parser.error("--base-url, --model and --output required")
    data = json.loads(args.fixtures.read_text())
    artifact = {
        "model": args.model,
        "base_url": args.base_url,
        "fixtures_sha256": sha256(args.fixtures),
        "runner_sha256": sha256(Path(__file__)),
        "conditions": {
            "thinking": args.thinking,
            "concurrency": 1,
            "seeds": [1234, 1235, 1236],
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 512,
        },
        "limitations": [
            "Client decode estimate counts the first streamed chunk's tokens too; "
            "batch/chunk boundaries affect this estimate.",
            "Prefill proxy is rendered prompt tokens / TTFT, including other startup work, not pure kernel throughput.",
            "EOS may end a reply before 512 tokens; actual counts retained.",
            "Deterministic case/seed header distinguishes cold prompts; warm follow-up repeats the exact body.",
            "UTC epoch annotations allow overlap audits; all performance intervals use the monotonic clock, not UTC.",
            "All results, including HTTP errors, are retained.",
        ],
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(artifact, stream, indent=2)
    key = args.key_file.read_text().strip() if args.key_file else "local-test-only"
    cases = [
        (f"{name}-{'greedy' if temp == 0 else 'sampled'}", prompt, temp, False)
        for name, prompt in (("code", data["code"]), ("chat", data["chat"]))
        for temp in (0, 1.0)
    ]
    cases += [
        (f"prefill-{length}", text + data["task"], 0, True)
        for length, text in data["corpus"].items()
    ]
    for case, prompt, temperature, warm in cases:
        for seed in artifact["conditions"]["seeds"]:
            body = {
                "model": args.model,
                "messages": [
                    {
                        "role": "user",
                        "content": f"[fixture {case} seed {seed}]\n" + prompt,
                    }
                ],
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 20,
                "seed": seed,
                "max_tokens": 512,
                "stream": True,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": args.thinking == "on"},
            }
            record = {"case": case, "seed": seed, "request": body}
            try:
                record["response"] = stream_request(args.base_url, key, body)
                if warm:
                    record["warm_response"] = stream_request(args.base_url, key, body)
            except urllib.error.HTTPError as exc:
                record["error"] = {"http_status": exc.code, "body": exc.read().decode()}
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            artifact["results"].append(record)
            artifact["summary"] = summaries(artifact["results"])
            args.output.write_text(
                json.dumps(artifact, ensure_ascii=False, indent=2) + "\n"
            )
            print(
                case,
                seed,
                record.get("error")
                or {
                    key: record["response"][key]
                    for key in ("ttft_s", "client_decode_tokens_per_s")
                },
                flush=True,
            )


if __name__ == "__main__":
    main()
