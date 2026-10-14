"""Run the installed Pi-web CLI runtime on isolated coding tasks in CPU-only Docker.

No cloud/user credentials, Docker socket, user home or production checkout are mounted.
Only the explicitly supplied local model test key enters the isolated config.
Only task files are writable; verifiers run separately with network disabled.
Host networking reaches loopback model APIs, so this is filesystem isolation, not
network isolation. Run sequentially outside any DOF eval and pause pilot services.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NODE = Path.home() / ".local/share/mise/installs/node/24.20.0"
PACKAGE = "lib/node_modules/@agegr/pi-web/node_modules/@earendil-works/pi-coding-agent"
CLI = "/opt/node/" + PACKAGE + "/dist/cli.js"
IMAGE = "tensorfold-qwen38:zig-db28187"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summarize_events(events):
    messages = [
        event["message"]
        for event in events
        if event.get("type") == "message_end"
        and event.get("message", {}).get("role") == "assistant"
    ]
    return {
        "agent_end": any(event.get("type") == "agent_end" for event in events),
        "assistant_turns": len(messages),
        "tool_calls": sum(
            event.get("type") == "tool_execution_start" for event in events
        ),
        "tool_errors": sum(
            event.get("type") == "tool_execution_end" and event.get("isError", False)
            for event in events
        ),
        "output_tokens": sum(
            message.get("usage", {}).get("output", 0) for message in messages
        ),
        "final_stop_reason": messages[-1].get("stopReason") if messages else None,
    }


def docker_base(name, network, memory="1g"):
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--network",
        network,
        "--runtime",
        "runc",
        "-e",
        "NVIDIA_VISIBLE_DEVICES=void",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "128",
        "--memory",
        memory,
        "--cpus",
        "2",
        "--user",
        "1000:1000",
        "--tmpfs",
        "/tmp:rw,size=256m",
        "-e",
        "HOME=/tmp/pi-home",
    ]


def verify_task(work, verifier, name):
    command = docker_base(name, "none", "256m")
    command += [
        "-v",
        f"{work}:/work:ro",
        "-v",
        f"{verifier}:/verify/verify.py:ro",
        "-e",
        "PYTHONPATH=/work",
        "-w",
        "/work",
        "--entrypoint",
        "python3",
        IMAGE,
        "/verify/verify.py",
    ]
    try:
        return subprocess.run(command, text=True, capture_output=True, timeout=30)
    finally:
        subprocess.run(
            ["docker", "rm", "-f", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def self_check():
    from tempfile import TemporaryDirectory

    references = {
        "ceil_div": ("calc.py", "def ceil_div(a, b):\n    return -(-a // b)\n"),
        "publication_date": (
            "dates.py",
            "import re\nfrom datetime import date\n"
            "def publication_date(text):\n"
            '    months = "enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre".split()\n'
            '    match = re.search(r"(\\d+)\\s+de\\s+([a-záéíóúñ]+)\\s+de\\s+(\\d{4})", text, re.I)\n'
            '    if not match: raise ValueError("missing date")\n'
            "    day, month, year = match.groups(); month = month.lower()\n"
            '    if month == "setiembre": month = "septiembre"\n'
            '    if month not in months: raise ValueError("unknown month")\n'
            "    return date(int(year), months.index(month) + 1, int(day)).isoformat()\n",
        ),
        "citation_guard": (
            "guards.py",
            "def validate_answer(payload, read_ids):\n"
            '    if not isinstance(payload, dict) or set(payload) != {"answer", "citations", "premise_status"}: raise ValueError("shape")\n'
            '    text, ids, status = payload["answer"], payload["citations"], payload["premise_status"]\n'
            '    if not isinstance(text, str) or not text.strip(): raise ValueError("text")\n'
            '    if not isinstance(ids, list) or not ids or any(type(i) is not int or i <= 0 for i in ids): raise ValueError("ids")\n'
            '    if len(set(ids)) != len(ids) or not set(ids) <= set(read_ids): raise ValueError("unread/duplicate")\n'
            '    if not isinstance(status, str) or status not in {"supported", "false", "unclear"}: raise ValueError("status")\n'
            '    return {"answer": text.strip(), "citations": list(ids), "premise_status": status}\n',
        ),
    }
    with TemporaryDirectory() as directory:
        for case, (filename, implementation) in references.items():
            work = Path(directory) / case
            work.mkdir()
            source = ROOT / "eval/pi_tasks" / case
            shutil.copyfile(source / filename, work / filename)
            baseline = verify_task(
                work, source / "verify.py", "dof-pi-verify-selfcheck"
            )
            assert baseline.returncode != 0, f"Broken input unexpectedly passes: {case}"
            (work / filename).write_text(implementation)
            fixed = verify_task(work, source / "verify.py", "dof-pi-verify-selfcheck")
            assert fixed.returncode == 0, (
                f"Verifier/reference failed: {case}\n{fixed.stderr}"
            )
            print(
                f"Pi fixture {case}: broken input fails, reference passes in read-only/no-network container.",
                flush=True,
            )
    print("No model inference used for self-check.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--thinking", choices=["off", "low"], default="off")
    parser.add_argument("--timeout", type=int, default=480)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not all((args.base_url, args.model, args.output)):
        parser.error("--base-url, --model, --output required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    key = args.key_file.read_text().strip() if args.key_file else "local-test-only"
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE], text=True
    ).strip()
    version = json.loads((NODE / PACKAGE / "package.json").read_text())["version"]
    artifact = {
        "model": args.model,
        "base_url": args.base_url,
        "pi_web_runtime_version": version,
        "cli_sha256": sha256(NODE / PACKAGE / "dist/cli.js"),
        "cpu_container_image": image_id,
        "thinking": args.thinking,
        "timeout_per_task_s": args.timeout,
        "conditions": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "seed": args.seed,
            "max_output_per_turn": 4096,
        },
        "isolation": "No GPU, user/cloud credentials, home, Docker socket or production mounts; local endpoint test key only; host networking.",
        "limitations": [
            "Three small stdlib coding tasks, not a broad software benchmark.",
            "Pi-web dependency CLI 0.84.3, not the separately installed global CLI.",
            "Host networking is not network confinement; prompts and source fixtures are trusted.",
            "Verifiers run read-only in a separate no-network container; inspect final source for test circumvention.",
        ],
        "results": [],
    }
    for case in ["ceil_div", "publication_date", "citation_guard"]:
        source = ROOT / "eval/pi_tasks" / case
        case_out = output / case
        work, config = case_out / "work", case_out / "config"
        work.mkdir(parents=True)
        config.mkdir()
        for path in source.iterdir():
            if path.is_file() and path.name != "verify.py":
                shutil.copyfile(path, work / path.name)
        kwargs = {"enable_thinking": args.thinking != "off", "preserve_thinking": True}
        if args.thinking != "off":
            kwargs["reasoning_effort"] = "low"
        models = {
            "providers": {
                "eval-local": {
                    "baseUrl": args.base_url,
                    "api": "openai-completions",
                    "apiKey": key,
                    "compat": {
                        "supportsDeveloperRole": False,
                        "supportsReasoningEffort": False,
                        "supportsStore": False,
                        "maxTokensField": "max_tokens",
                        "thinkingFormat": "qwen-chat-template",
                    },
                    "models": [
                        {
                            "id": args.model,
                            "reasoning": True,
                            "contextWindow": args.context,
                            "maxTokens": 4096,
                            "samplingParams": {
                                "temperature": 0.6,
                                "top_p": 0.95,
                                "top_k": 20,
                                "seed": args.seed,
                                "chat_template_kwargs": kwargs,
                            },
                        }
                    ],
                }
            }
        }
        (config / "models.json").write_text(json.dumps(models, indent=2))
        (config / "models.json").chmod(0o600)
        settings = {
            "compaction": {"enabled": False},
            "retry": {"enabled": False, "provider": {"maxRetries": 0}},
            "enableInstallTelemetry": False,
            "defaultProjectTrust": "never",
        }
        (config / "settings.json").write_text(json.dumps(settings))
        name = "dof-qwen-pi-case"
        command = docker_base(name, "host")
        # ponytail: host network reaches loopback APIs; use an allowlisted proxy before adversarial prompts.
        command += [
            "-v",
            f"{NODE / 'bin/node'}:/opt/node/bin/node:ro",
            "-v",
            f"{NODE / 'lib/node_modules/@agegr/pi-web/node_modules'}:/opt/node/lib/node_modules/@agegr/pi-web/node_modules:ro",
            "-v",
            f"{work}:/work:rw",
            "-v",
            f"{config}:/config:rw",
            "-e",
            "PATH=/opt/node/bin:/usr/local/bin:/usr/bin:/bin",
            "-e",
            "PI_CODING_AGENT_DIR=/config",
            "-e",
            "PI_OFFLINE=1",
            "-e",
            "PI_TELEMETRY=0",
            "-w",
            "/work",
            "--entrypoint",
            "/opt/node/bin/node",
            IMAGE,
            CLI,
            "--mode",
            "json",
            "--print",
            "--no-session",
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--no-themes",
            "--no-context-files",
            "--no-approve",
            "--provider",
            "eval-local",
            "--model",
            args.model,
            "--thinking",
            args.thinking,
            "--tools",
            "read,bash,edit,write",
            "Read README.md and fix the implementation. Use only standard-library tools. "
            "No network access or dependency installation. Do not change the task requirements. "
            "Check your work with runnable tests, then explain the change briefly.",
        ]
        before = {path.name: sha256(path) for path in work.iterdir() if path.is_file()}
        record = {
            "case": case,
            "fixture_hashes": before,
            "verifier_sha256": sha256(source / "verify.py"),
        }
        start = time.monotonic()
        try:
            with (
                (case_out / "events.jsonl").open("x") as stdout,
                (case_out / "stderr.log").open("x") as stderr,
            ):
                process = subprocess.run(
                    command, stdout=stdout, stderr=stderr, timeout=args.timeout
                )
                record["pi_exit"] = process.returncode
        except subprocess.TimeoutExpired:
            record["timeout"] = True
        finally:
            subprocess.run(
                ["docker", "rm", "-f", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        record["pi_elapsed_s"] = time.monotonic() - start
        events, unparsed = [], []
        for line in (case_out / "events.jsonl").read_text().splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                unparsed.append(line)
        record["events_summary"] = summarize_events(events)
        record["non_json_lines"] = unparsed
        verification = verify_task(work, source / "verify.py", "dof-qwen-pi-verify")
        (case_out / "verification.log").write_text(
            verification.stdout + verification.stderr
        )
        record["verifier_exit"] = verification.returncode
        if case == "publication_date":
            diagnostic = ROOT / "scripts/probe_pi_date_contract.py"
            checked = verify_task(work, diagnostic, "dof-qwen-pi-verify")
            (case_out / "contract-diagnostic.log").write_text(
                checked.stdout + checked.stderr
            )
            try:
                diagnostic_cases = (
                    json.loads(checked.stdout) if checked.returncode == 0 else None
                )
            except json.JSONDecodeError:
                diagnostic_cases = None
            record["date_contract_diagnostic"] = {
                "source_sha256": sha256(diagnostic),
                "exit": checked.returncode,
                "cases": diagnostic_cases,
                "scope": "Post-submission first-invalid-date precedence; original verifier/score unchanged.",
            }
        record["final_hashes"] = {
            path.name: sha256(path) for path in work.iterdir() if path.is_file()
        }
        record["passed"] = (
            record.get("pi_exit") == 0
            and verification.returncode == 0
            and record["events_summary"]["agent_end"]
            and record["events_summary"]["tool_calls"] > 0
            and record["events_summary"]["final_stop_reason"] == "stop"
        )
        artifact["results"].append(record)
        (output / "results.json").write_text(json.dumps(artifact, indent=2) + "\n")
        print(
            case,
            "PASS" if record["passed"] else "FAIL",
            round(record["pi_elapsed_s"], 1),
            record["events_summary"],
            flush=True,
        )


if __name__ == "__main__":
    main()
