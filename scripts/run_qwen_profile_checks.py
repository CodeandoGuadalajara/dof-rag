"""Run benchmark, real DOF image checks and isolated Pi tasks sequentially.

Requires paused pilot services and their execution lock held by the parent unit.
Does not switch generation models. Existing helpers own their detailed artifacts.
"""

import argparse
import json
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from scripts.evaluate_qwen_candidate import runtime_info, sample_memory

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--inputs-root", required=True, type=Path)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--container")
    parser.add_argument("--server-binary", type=Path)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument(
        "--text-thinking", choices=("off", "low"), default="off",
        help="Text benchmark/Pi reasoning; image checks retain their thinking-off control",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    connection = ["--base-url", args.base_url, "--model", args.model]
    if args.key_file:
        connection += ["--key-file", str(args.key_file.resolve())]
    python = str(ROOT / ".venv/bin/python")
    root = args.inputs_root.resolve()
    stages = [
        (
            "benchmark",
            [
                python,
                str(ROOT / "scripts/benchmark_qwen_candidate.py"),
                *connection,
                "--thinking",
                "on" if args.text_thinking != "off" else "off",
                "--fixtures",
                str(root / "benchmark-fixtures.json"),
                "--output",
                str(output / "benchmark.json"),
            ],
        ),
        (
            "images",
            [
                python,
                str(ROOT / "scripts/check_qwen_dof_images.py"),
                *connection,
                "--fixtures",
                str(root / "images/fixtures.json"),
                "--output",
                str(output / "images.json"),
            ]
            + (["--unsupported"] if args.text_only else []),
        ),
        (
            "pi",
            [
                python,
                str(ROOT / "scripts/check_qwen_pi_tasks.py"),
                *connection,
                "--context",
                str(args.context),
                "--output",
                str(output / "pi"),
                "--thinking",
                args.text_thinking,
            ],
        ),
    ]
    summary = {
        "model": args.model,
        "endpoint": args.base_url,
        "text_thinking": args.text_thinking,
        "runtime_before": runtime_info(args.container, args.server_binary),
        "stages": [],
    }
    stopped = threading.Event()
    sampler = threading.Thread(
        target=sample_memory, args=(output / "memory.jsonl", stopped), daemon=True
    )
    sampler.start()
    try:
        for name, command in stages:
            with (output / (name + ".log")).open("x") as log:
                result = subprocess.run(
                    command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
                )
            summary["stages"].append({"stage": name, "exit": result.returncode})
            (output / "stages.json").write_text(json.dumps(summary, indent=2) + "\n")
            assert result.returncode == 0, (
                f"{name} script failed; inspect {output / (name + '.log')}"
            )
        summary["runtime_after"] = runtime_info(args.container, args.server_binary)
        (output / "stages.json").write_text(json.dumps(summary, indent=2) + "\n")
    finally:
        stopped.set()
        sampler.join(timeout=10)


if __name__ == "__main__":
    main()
