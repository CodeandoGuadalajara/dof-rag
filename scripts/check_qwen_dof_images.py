"""Read two real DOF page images with a fixed, visually verified extraction rubric.

No tools or OCR text are supplied to the model; only page pixels and the prompt.
Exact field extraction, bare-JSON formatting, and model correctness are distinct.
"""

import argparse
import base64
import hashlib
import json
import sys
import time
import unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_object(text):
    from agent_tools.agent import _without_thinking

    text = _without_thinking(text)
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character == "{":
            try:
                value, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise ValueError("No JSON object in response")


def compare_fields(value, expected, numeric):
    def fold(text):
        return " ".join(
            "".join(
                character
                for character in unicodedata.normalize("NFD", str(text)).casefold()
                if not unicodedata.combining(character)
            ).split()
        )

    matches = {}
    for key, answer in expected.items():
        if key not in value:
            matches[key] = False
        elif key in numeric:
            try:
                matches[key] = Decimal(str(value[key]).rstrip("%")) == Decimal(answer)
            except (InvalidOperation, ValueError):
                matches[key] = False
        else:
            matches[key] = fold(value[key]) == fold(answer)
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--unsupported",
        action="store_true",
        help="Record known text-only deployment; do not send images",
    )
    args = parser.parse_args()
    fixture_data = json.loads(args.fixtures.read_text())
    artifact = {
        "model": args.model,
        "base_url": args.base_url,
        "fixtures_sha256": hashlib.sha256(args.fixtures.read_bytes()).hexdigest(),
        "conditions": {
            "temperature": 0,
            "seeds": [1234, 1235, 1236],
            "thinking": "off",
            "max_tokens": 1200,
        },
        "limitations": [
            "Two real PDF page renders; not scanned-page/OCR coverage of the entire corpus.",
            "Rubric was visually inspected and cross-checked against PDF text; not a legal adjudication.",
            "Field extraction and JSON formatting are reported separately; human semantic review may clarify failures.",
        ],
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(artifact, stream, indent=2)
    if args.unsupported:
        artifact["status"] = "unsupported_in_current_deployment"
        args.output.write_text(json.dumps(artifact, indent=2) + "\n")
        print("Text-only deployment recorded; no image requests sent.", flush=True)
        return
    key = args.key_file.read_text().strip() if args.key_file else "local-test-only"
    with OpenAI(
        base_url=args.base_url, api_key=key, timeout=180, max_retries=0
    ) as client:
        for case in fixture_data["cases"]:
            image = (args.fixtures.parent / case["image"]).read_bytes()
            assert hashlib.sha256(image).hexdigest() == case["image_sha256"], (
                "Image fixture changed"
            )
            content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,"
                        + base64.b64encode(image).decode()
                    },
                },
                {"type": "text", "text": case["prompt"]},
            ]
            for seed in artifact["conditions"]["seeds"]:
                body = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": 1200,
                    "temperature": 0,
                    "top_p": 0.95,
                    "seed": seed,
                    "extra_body": {
                        "top_k": 20,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                }
                record = {
                    "case": case["id"],
                    "seed": seed,
                    "image_sha256": case["image_sha256"],
                    "prompt": case["prompt"],
                    "expected": case["expected"],
                }
                start = time.monotonic()
                try:
                    response = client.chat.completions.create(**body)
                    record["response"] = response.model_dump(mode="json")
                    text = response.choices[0].message.content or ""
                    try:
                        value = parse_object(text)
                        record["field_matches"] = compare_fields(
                            value, case["expected"], case.get("numeric_fields", [])
                        )
                        record["all_fields_match"] = all(
                            record["field_matches"].values()
                        )
                        record["shape_matches"] = set(value) == set(
                            case["expected"]
                        ) and all(isinstance(v, str) for v in value.values())
                        try:
                            record["bare_json"] = json.loads(text.strip()) == value
                        except json.JSONDecodeError:
                            record["bare_json"] = False
                    except ValueError as exc:
                        record["parse_error"] = str(exc)
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                record["elapsed_s"] = time.monotonic() - start
                artifact["results"].append(record)
                args.output.write_text(
                    json.dumps(artifact, ensure_ascii=False, indent=2) + "\n"
                )
                print(
                    case["id"],
                    seed,
                    record.get("error")
                    or record.get("parse_error")
                    or {
                        "all_fields_match": record["all_fields_match"],
                        "bare_json": record["bare_json"],
                        "elapsed_s": round(record["elapsed_s"], 2),
                    },
                    flush=True,
                )


if __name__ == "__main__":
    main()
