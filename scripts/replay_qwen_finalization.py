"""Exact captured final-request replay and explicitly labelled prompt ablations.

No tool is executed. Parser success is not correctness or whole-agent completion.
Prepare fixtures after an arm finishes; replay only while no evaluation is running.
"""

import argparse
import copy
import hashlib
import json
import runpy
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def make_fixtures(results, traffic):
    by_id = {
        item["response"].get("id"): item
        for item in traffic
        if isinstance(item.get("response"), dict)
    }
    fixtures = []
    for item in results["results"]:
        run = item.get("run")
        if not run or run["stop_reason"] == "completed":
            continue
        turn = run["turns"][-1]
        captured = by_id.get(turn["response_id"])
        if not captured or captured["request"].get("tool_choice") != "none":
            continue
        evidence = {}
        for trace in run["traces"]:
            if trace["name"] == "read_chunks" and trace["output"].get("ok"):
                for chunk in trace["output"]["data"]["chunks"]:
                    evidence[chunk["chunk_id"]] = chunk
        previous = (
            by_id.get(run["turns"][-2]["response_id"])
            if len(run["turns"]) > 1
            else None
        )
        fixtures.append(
            {
                "id": item["id"],
                "question": item["question"],
                "reference_answer": item["reference_answer"],
                "source_stop_reason": run["stop_reason"],
                "coverage_state": run["coverage"],
                "required_hops": item["required_hops"],
                "request": captured["request"],
                "source_response": captured["response"],
                "prior_tool_definitions": previous["request"].get("tools", [])
                if previous
                else [],
                "read_evidence": list(evidence.values()),
            }
        )
    return fixtures


def compact_read_final_messages(messages):
    """Replace tool history with exactly the successful read payloads already sent.

    Evaluation-only final-input ablation; never creates evidence from search
    snippets, assistant text, references or failed tools. Existing answer/citation
    validation still runs unchanged after generation.
    """
    if (
        len(messages) < 3
        or messages[0].get("role") != "system"
        or messages[1].get("role") != "user"
        or messages[-1].get("role") != "user"
    ):
        raise ValueError("Unexpected frozen finalization message shape")
    read_calls, evidence = set(), {}
    for message in messages:
        if message.get("role") == "assistant":
            read_calls.update(
                call["id"]
                for call in message.get("tool_calls", [])
                if call.get("function", {}).get("name") == "read_chunks"
            )
        elif (
            message.get("role") == "tool" and message.get("tool_call_id") in read_calls
        ):
            payload = json.loads(message["content"])
            if not isinstance(payload, dict):
                raise ValueError("Malformed trusted tool output")
            if payload.get("ok") is not True:
                continue
            chunks = payload.get("chunks")
            if not isinstance(chunks, list):
                raise ValueError("Malformed successful read_chunks output")
            for chunk in chunks:
                if (
                    not isinstance(chunk, dict)
                    or type(chunk.get("chunk_id")) is not int
                    or type(chunk.get("document_id")) is not int
                    or not isinstance(chunk.get("text"), str)
                    or chunk["chunk_id"] <= 0
                    or chunk["document_id"] <= 0
                ):
                    raise ValueError("Malformed read evidence")
                evidence[chunk["chunk_id"]] = chunk
    if not evidence:
        return copy.deepcopy(messages)
    return [
        copy.deepcopy(messages[0]),
        copy.deepcopy(messages[1]),
        {
            "role": "user",
            "content": "Evidencia ya leída en esta ejecución; no hay herramientas disponibles:\n"
            + json.dumps(list(evidence.values()), ensure_ascii=False),
        },
        copy.deepcopy(messages[-1]),
    ]


def variant_request(fixture, variant, model, seed):
    request = copy.deepcopy(fixture["request"])
    request["model"], request["seed"] = model, seed
    assert request.get("tool_choice") == "none"
    if variant == "omitted-tools":
        request.pop("tools", None)
    elif variant == "json-object":
        request["response_format"] = {"type": "json_object"}
    elif variant == "token-ids":
        # vLLM diagnostic extension; no decoding/recovery or tool execution here.
        request["return_token_ids"] = True
    elif variant == "declared-none":
        request["tools"] = copy.deepcopy(fixture["prior_tool_definitions"])
    elif variant == "explicit-json":
        request["messages"].append(
            {
                "role": "user",
                "content": "Cierre obligatorio: devuelve solamente el objeto JSON de respuesta, citas y estado de premisa. "
                "No escribas <tool_call>, <function> ni solicites herramientas, incluso si faltan datos. "
                "No inventes evidencia ni citas. Explica dentro de answer cualquier limitación.",
            }
        )
    elif variant == "read-evidence-only":
        original = request["messages"]
        assert original[0]["role"] == "system" and original[1]["role"] == "user"
        request["messages"] = [
            copy.deepcopy(original[0]),
            copy.deepcopy(original[1]),
            {
                "role": "user",
                "content": "Evidencia que ya fue leída en esta ejecución (no hay herramientas disponibles):\n"
                + json.dumps(fixture["read_evidence"], ensure_ascii=False),
            },
            copy.deepcopy(original[-1]),
        ]
    elif variant == "compact-read-evidence-only":
        request["messages"] = compact_read_final_messages(request["messages"])
    elif variant != "original":
        raise ValueError(f"Unknown variant: {variant}")
    return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--results", type=Path)
    parser.add_argument("--traffic", type=Path)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--ids", help="Comma-separated case IDs; default all eligible failures"
    )
    parser.add_argument(
        "--seeds",
        default="1234",
        help="Original arm seed first; extra seeds are labelled mutations",
    )
    parser.add_argument(
        "--variants", default="original,declared-none,explicit-json,read-evidence-only",
        help="Comma-separated variants; optional omitted-tools/json-object/token-ids are explicit diagnostics, not default fixes. Runtime support must be checked.",
    )
    args = parser.parse_args()
    if args.prepare:
        if not (args.results and args.traffic):
            parser.error("--prepare requires --results and --traffic")
        traffic = [
            json.loads(line)
            for line in args.traffic.read_text().splitlines()
            if line.strip()
        ]
        data = make_fixtures(json.loads(args.results.read_text()), traffic)
        assert data, "No eligible tool-disabled final requests captured"
        args.fixtures.parent.mkdir(parents=True, exist_ok=True)
        with args.fixtures.open("x") as stream:
            json.dump(
                {
                    "results_source": str(args.results),
                    "traffic_source": str(args.traffic),
                    "results_sha256": hashlib.sha256(
                        args.results.read_bytes()
                    ).hexdigest(),
                    "traffic_sha256": hashlib.sha256(
                        args.traffic.read_bytes()
                    ).hexdigest(),
                    "fixtures": data,
                },
                stream,
                ensure_ascii=False,
                indent=2,
            )
        print(
            f"Prepared {len(data)} exact captured final-request fixtures.", flush=True
        )
        return
    if not all((args.base_url, args.model, args.output)):
        parser.error("--base-url, --model, --output required")
    frozen = runpy.run_path(str(ROOT / "var/run-flash-next-eval-post-fix.py"))
    artifact = {
        "model": args.model,
        "base_url": args.base_url,
        "fixtures_sha256": hashlib.sha256(args.fixtures.read_bytes()).hexdigest(),
        "limitations": [
            "Only original variant with unchanged model and seed reproduces the exact captured JSON request.",
            "Runtime/cache state may differ from original; replies need not match.",
            "Prompt ablations change final input, never tool budgets or evidence validation.",
            "Parser success alone is not semantic correctness or coverage completion.",
        ],
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(artifact, stream, indent=2)
    selected = set(args.ids.split(",")) if args.ids else None
    key = args.key_file.read_text().strip() if args.key_file else "local-test-only"
    for fixture in json.loads(args.fixtures.read_text())["fixtures"]:
        if selected and fixture["id"] not in selected:
            continue
        documents = {
            chunk["chunk_id"]: chunk["document_id"]
            for chunk in fixture["read_evidence"]
        }
        for variant in args.variants.split(","):
            for seed in map(int, args.seeds.split(",")):
                body = variant_request(fixture, variant, args.model, seed)
                record = {
                    "id": fixture["id"],
                    "variant": variant,
                    "seed": seed,
                    "request": body,
                    "request_equals_captured": body == fixture["request"],
                }
                start = time.monotonic()
                try:
                    request = urllib.request.Request(
                        args.base_url.rstrip("/") + "/chat/completions",
                        json.dumps(body).encode(),
                        {
                            "Content-Type": "application/json",
                            "Authorization": "Bearer " + key,
                        },
                    )
                    with urllib.request.urlopen(request, timeout=300) as response:
                        record["response"] = json.load(response)
                    message = record["response"]["choices"][0]["message"]
                    record["has_tool_calls"] = bool(message.get("tool_calls"))
                    try:
                        if record["has_tool_calls"]:
                            raise ValueError("Tool call during forced finalization")
                        if (
                            record["response"]["choices"][0].get("finish_reason")
                            == "length"
                        ):
                            raise ValueError("Truncated final answer")
                        answer = frozen["agent"]._parse_final_answer(
                            message.get("content") or "",
                            set(documents),
                            citation_documents=documents,
                            required_hops=fixture["required_hops"],
                        )
                        record["valid_final"] = True
                        record["answer"] = {
                            "answer": answer.answer,
                            "citations": answer.citations,
                            "premise_status": answer.premise_status,
                            "invalid_citations": answer.invalid_citations,
                        }
                    except Exception as exc:
                        record["valid_final"] = False
                        record["parse_error"] = f"{type(exc).__name__}: {exc}"
                except urllib.error.HTTPError as exc:
                    record["error"] = {
                        "http_status": exc.code,
                        "body": exc.read().decode(),
                    }
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                record["elapsed_s"] = time.monotonic() - start
                artifact["results"].append(record)
                args.output.write_text(
                    json.dumps(artifact, ensure_ascii=False, indent=2) + "\n"
                )
                print(
                    fixture["id"],
                    variant,
                    seed,
                    "parser-valid"
                    if record.get("valid_final")
                    else record.get("error", record.get("parse_error")),
                    flush=True,
                )


if __name__ == "__main__":
    main()
