"""Controlled raw-API finalization probes; no retrieval, writes, or executed tool calls.

Run from dof-rag with its venv. Output is a fresh JSON artifact, checkpointed after
all requests. Synthetic history isolates the interface; it is NOT historical replay.
"""

import argparse
import hashlib
import json
import runpy
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = {
    "type": "function",
    "function": {
        "name": "read_chunks",
        "description": "Read evidence returned by search.",
        "parameters": {
            "type": "object",
            "properties": {
                "chunk_ids": {"type": "array", "items": {"type": "integer"}},
                "neighbor_window": {"type": "integer"},
            },
            "required": ["chunk_ids", "neighbor_window"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


def fixture(mode, model, seed, serial):
    system = (
        "Eres un investigador del DOF. Usa solamente la evidencia ya leída. "
        "Entrega JSON con answer (texto), citations (lista de enteros), y "
        'premise_status ("supported", "false", o "unclear"). '
        "No quedan herramientas ni turnos. No solicites más herramientas."
    )
    question = "¿Cuál es el salario mínimo diario de la ZSMG en 2026?"
    evidence = {
        "ok": True,
        "chunks": [
            {
                "chunk_id": 101,
                "document_id": 201,
                "text": "Los salarios mínimos generales vigentes desde el 1 de enero de 2026 "
                "son 315.04 pesos diarios en la ZSMG y 440.87 en la ZLFN.",
            }
        ],
    }
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    if mode == "no_history":
        messages.append(
            {"role": "user", "content": "Evidencia leída: " + json.dumps(evidence)}
        )
    else:
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "fixture-read-1",
                            "type": "function",
                            "function": {
                                "name": "read_chunks",
                                "arguments": '{"chunk_ids":[101],"neighbor_window":0}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "fixture-read-1",
                    "content": json.dumps(evidence),
                },
            ]
        )
    messages.append(
        {
            "role": "user",
            "content": "Turnos restantes: 0. Herramientas restantes: 0. "
            "Chunks leídos: [101]. Entrega ahora el JSON final. No pidas herramientas.",
        }
    )
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "seed": seed,
        "parallel_tool_calls": False,
        "tool_choice": "none",
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if mode in ("declared_none", "declared_auto"):
        body["tools"] = [TOOL]
    if mode == "declared_auto":
        body["tool_choice"] = "auto"
    if serial:
        body["draft"] = False
    return body


def classify(response, parse_final):
    message = response["choices"][0]["message"]
    content = message.get("content") or ""
    result = {
        "finish_reason": response["choices"][0].get("finish_reason"),
        "has_tool_calls": bool(message.get("tool_calls")),
        "xml_tool_content": "<tool_call>" in content,
    }
    try:
        answer = parse_final(content, {101}, citation_documents={101: 201})
        result["valid_final"] = True
        result["answer"] = answer.answer
    except Exception as exc:
        result["valid_final"] = False
        result["parse_error"] = f"{type(exc).__name__}: {exc}"
    return result


def self_check(parse_final):
    absent = fixture("absent_tools", "fixture", 1234, False)
    declared = fixture("declared_none", "fixture", 1234, True)
    assert "tools" not in absent and absent["tool_choice"] == "none"
    assert declared["tools"] == [TOOL] and declared["draft"] is False
    answer = '{"answer":"315.04 pesos diarios","citations":[101],"premise_status":"supported"}'
    good = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": "```json\n" + answer + "\n```"},
            }
        ]
    }
    assert classify(good, parse_final)[
        "valid_final"
    ]  # Existing harness accepts fences.
    bad = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": "<tool_call>read_chunks</tool_call>"},
            }
        ]
    }
    result = classify(bad, parse_final)
    assert result["xml_tool_content"] and not result["valid_final"]
    print("Fixture and actual frozen-parser self-check passed.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    parser.add_argument("--model", default="Qwen3.8-Flash-Next")
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Also send draft:false; only on servers supporting it",
    )
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    frozen = runpy.run_path(str(ROOT / "var/run-flash-next-eval-post-fix.py"))
    parse_final = frozen["agent"]._parse_final_answer
    self_check(parse_final)
    if args.self_check:
        return
    if args.output is None:
        parser.error("--output required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "kind": "synthetic-finalization-probe",
        "base_url": args.base_url,
        "model": args.model,
        "snapshot_sha256": hashlib.sha256(frozen["SNAPSHOT"].read_bytes()).hexdigest(),
        "limitations": [
            "Synthetic history, not saved historical conversation.",
            "Fixed small evidence and seeds; not a quality benchmark.",
            "draft:false is meaningful only if the server implements it.",
        ],
        "results": [],
    }
    # Refuse overwriting prior evidence, including incomplete probe artifacts.
    with args.output.open("x") as stream:
        json.dump(artifact, stream, indent=2)
    key = args.key_file.read_text().strip() if args.key_file else "local-test-only"
    for mode in ("no_history", "absent_tools", "declared_none", "declared_auto"):
        for serial in (False, True) if args.serial else (False,):
            for seed in (1234, 1235, 1236):
                body = fixture(mode, args.model, seed, serial)
                request = urllib.request.Request(
                    args.base_url.rstrip("/") + "/chat/completions",
                    json.dumps(body).encode(),
                    {
                        "Content-Type": "application/json",
                        "Authorization": "Bearer " + key,
                    },
                )
                record = {"mode": mode, "serial": serial, "seed": seed, "request": body}
                start = time.monotonic()
                try:
                    with urllib.request.urlopen(request, timeout=180) as response:
                        record["response"] = json.load(response)
                    record["classification"] = classify(record["response"], parse_final)
                except urllib.error.HTTPError as exc:
                    record["http_error"] = {
                        "status": exc.code,
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
                    mode,
                    "serial" if serial else "default",
                    seed,
                    record.get(
                        "classification", record.get("http_error", record.get("error"))
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
