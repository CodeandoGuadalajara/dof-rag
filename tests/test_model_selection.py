"""Regression checks for comparison accounting and secret redaction."""

import io
import json
import runpy
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx

from agent_tools.agent import (
    PremiseCorrectionRequiredError,
    _has_affirmative_premise_correction,
    _has_explicit_premise_correction,
    _parse_final_answer,
)
from scripts.benchmark_qwen_candidate import stream_request, summaries
from scripts.check_qwen_dof_images import compare_fields, parse_object
from scripts.check_qwen_pi_tasks import docker_base, summarize_events
from scripts.evaluate_qwen_candidate import (
    all_run_summary,
    capture_chat_request,
    redact_arguments,
)
from scripts.prepare_qwen_answer_review import review_rows
from scripts.replay_qwen_finalization import (
    compact_read_final_messages,
    make_fixtures,
    variant_request,
)


class ModelSelectionTests(unittest.TestCase):
    def test_prepared_chat_capture_survives_timeout_without_headers_or_body_changes(self):
        output = io.StringIO()
        capture_chat_request(httpx.Request("GET", "http://127.0.0.1:1/v1/models"), output)
        self.assertEqual(output.getvalue(), "")
        body = {"model": "test", "messages": [{"role": "user", "content": "DOF"}], "max_tokens": 2400}

        def timeout(request):
            self.assertEqual(json.loads(request.content), body)
            self.assertEqual(json.loads(output.getvalue())["request"], body)
            raise httpx.ReadTimeout("simulated", request=request)

        with (
            httpx.Client(transport=httpx.MockTransport(timeout), event_hooks={"request": [lambda request: capture_chat_request(request, output)]}) as client,
            self.assertRaises(httpx.ReadTimeout),
        ):
            client.post("http://127.0.0.1:1/v1/chat/completions", json=body, headers={"Authorization": "Bearer secret-test-key"})
        self.assertNotIn("secret-test-key", output.getvalue())
        self.assertEqual(len(output.getvalue().splitlines()), 1)

    def test_benchmark_thinking_default_and_opt_in_match_payload(self):
        from scripts import benchmark_qwen_candidate as benchmark

        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture.json"
            fixture.write_text(json.dumps({"code": "code", "chat": "chat", "corpus": {}, "task": "task"}))
            for mode in ("off", "on"):
                output = root / f"{mode}.json"
                argv = ["benchmark", "--fixtures", str(fixture), "--base-url", "http://127.0.0.1:1/v1", "--model", "test", "--output", str(output)]
                if mode == "on":
                    argv += ["--thinking", mode]
                with patch("sys.argv", argv), patch.object(benchmark, "stream_request", return_value={"ttft_s": 1, "client_decode_tokens_per_s": 2}) as request, patch.object(benchmark, "summaries", return_value=[]), redirect_stdout(io.StringIO()):
                    benchmark.main()
                self.assertEqual(json.loads(output.read_text())["conditions"]["thinking"], mode)
                self.assertEqual(request.call_count, 12)
                for call in request.call_args_list:
                    self.assertEqual(call.args[2]["chat_template_kwargs"], {"enable_thinking": mode == "on"})
                    self.assertEqual(call.args[2]["max_tokens"], 512)

    def test_profile_text_thinking_propagates_without_changing_image_control(self):
        from scripts import run_qwen_profile_checks as profile

        with TemporaryDirectory() as directory:
            root = Path(directory)
            for mode in ("off", "low"):
                output = root / mode
                argv = ["profile", "--base-url", "http://127.0.0.1:1/v1", "--model", "test", "--output", str(output), "--inputs-root", str(root), "--text-only"]
                if mode == "low":
                    argv += ["--text-thinking", mode]
                with patch("sys.argv", argv), patch("scripts.evaluate_qwen_candidate.runtime_info", return_value={"test": True}), patch("scripts.evaluate_qwen_candidate.sample_memory"), patch.object(profile.subprocess, "run", return_value=types.SimpleNamespace(returncode=0)) as run:
                    profile.main()
                commands = [call.args[0] for call in run.call_args_list]
                self.assertEqual(commands[0][commands[0].index("--thinking") + 1], "on" if mode == "low" else "off")
                self.assertEqual(commands[2][commands[2].index("--thinking") + 1], mode)
                self.assertNotIn("--thinking", commands[1])
                self.assertIn("--unsupported", commands[1])
                self.assertEqual(json.loads((output / "stages.json").read_text())["text_thinking"], mode)

    def test_failed_runs_count_in_latency_tokens_and_tool_errors(self):
        data = {
            "results": [
                {
                    "run": {
                        "stop_reason": "completed",
                        "elapsed_ms": 1000,
                        "usage": {"output_tokens": 20},
                        "traces": [{"output": {"ok": True}}],
                    }
                },
                {
                    "run": {
                        "stop_reason": "model_turn_limit",
                        "elapsed_ms": 3000,
                        "usage": {"output_tokens": 12},
                        "traces": [{"output": {"ok": False}}],
                    }
                },
                {"error": {"type": "ProviderError"}},
            ]
        }
        summary = all_run_summary(data)
        self.assertEqual(summary["outcomes"], 3)
        self.assertEqual(summary["timed_runs"], 2)
        self.assertEqual(summary["completed"], 1)
        self.assertEqual(summary["all_trace_tool_errors"], 1)
        self.assertEqual(summary["all_run_mean_elapsed_s"], 2)
        self.assertEqual(summary["effective_output_tokens_per_s"], 8)
        self.assertEqual(summary["correctness"], "pending passage-based review")

    def test_stream_clocks_and_usage_are_distinct(self):
        chunks = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "hello"}}]},
            {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 10}},
        ]
        wire = b"".join(
            ("data: " + json.dumps(chunk) + "\n\n").encode() for chunk in chunks
        )
        wire += b"data: [DONE]\n\n"
        with (
            patch("urllib.request.urlopen", return_value=io.BytesIO(wire)),
            patch(
                "scripts.benchmark_qwen_candidate.time.time",
                side_effect=[1600000000, 1600000001],
            ),
            patch(
                "scripts.benchmark_qwen_candidate.time.perf_counter",
                side_effect=[100, 100.1, 100.3, 100.5, 100.6, 100.8],
            ),
        ):
            result = stream_request("http://127.0.0.1:8000/v1", "dummy", {})
        self.assertEqual(result["started_at_unix_s"], 1600000000)
        self.assertEqual(result["finished_at_unix_s"], 1600000001)
        self.assertAlmostEqual(result["ttft_s"], 0.3)
        self.assertAlmostEqual(result["wall_s"], 0.8)
        self.assertAlmostEqual(result["client_decode_tokens_per_s"], 20)
        self.assertEqual(result["usage"]["completion_tokens"], 10)
        self.assertEqual(result["answer"], "hello")

    def test_stream_errors_are_not_successful_tokens(self):
        wire = b'data: {"error":"invalid request"}\n\n'
        with (patch("urllib.request.urlopen", return_value=io.BytesIO(wire)),
              self.assertRaisesRegex(RuntimeError, "Streaming server error")):
            stream_request("http://127.0.0.1:8000/v1", "dummy", {})
        self.assertEqual(
            summaries([{"case": "bad", "error": "failed"}]),
            [{"case": "bad", "successful_requests": 0}],
        )

    def test_final_replay_preserves_original_and_only_uses_read_evidence(self):
        original = {
            "model": "fixture",
            "seed": 1234,
            "tool_choice": "none",
            "messages": [
                {"role": "system", "content": "rules"},
                {"role": "user", "content": "question"},
                {"role": "user", "content": "close now"},
            ],
        }
        results = {
            "results": [
                {
                    "id": "case",
                    "question": "question",
                    "reference_answer": "reference",
                    "required_hops": 1,
                    "run": {
                        "stop_reason": "bad final",
                        "coverage": {},
                        "turns": [{"response_id": "final"}],
                        "traces": [
                            {
                                "name": "read_chunks",
                                "output": {
                                    "ok": True,
                                    "data": {
                                        "chunks": [
                                            {
                                                "chunk_id": 101,
                                                "document_id": 201,
                                                "text": "read evidence",
                                            }
                                        ]
                                    },
                                },
                            },
                            {
                                "name": "search_evidence",
                                "output": {
                                    "ok": True,
                                    "data": {
                                        "evidence": [
                                            {
                                                "chunk_id": 999,
                                                "text": "unread candidate",
                                            }
                                        ]
                                    },
                                },
                            },
                        ],
                    },
                }
            ]
        }
        traffic = [{"request": original, "response": {"id": "final"}}]
        fixture = make_fixtures(results, traffic)[0]
        self.assertEqual(
            variant_request(fixture, "original", "fixture", 1234), original
        )
        transformed = variant_request(fixture, "read-evidence-only", "fixture", 1234)
        self.assertIn("read evidence", json.dumps(transformed))
        self.assertNotIn("unread candidate", json.dumps(transformed))
        self.assertNotIn("reference", json.dumps(transformed))
        self.assertEqual(fixture["request"], original)
        self.assertEqual(transformed["tool_choice"], "none")

    def test_final_protocol_diagnostics_preserve_evidence_and_budgets(self):
        original = {
            "model": "fixture", "seed": 1234, "tool_choice": "none",
            "max_tokens": 2400,
            "tools": [{"type": "function", "function": {"name": "read_chunks"}}],
            "messages": [{"role": "user", "content": "already observed evidence"}],
        }
        fixture = {"request": original}
        for variant in ("omitted-tools", "json-object", "token-ids"):
            with self.subTest(variant=variant):
                request = variant_request(fixture, variant, "fixture", 1234)
                self.assertEqual(request["messages"], original["messages"])
                self.assertEqual(request["max_tokens"], 2400)
                self.assertEqual(request["tool_choice"], "none")
                if variant == "omitted-tools":
                    self.assertNotIn("tools", request)
                elif variant == "json-object":
                    self.assertEqual(request["response_format"], {"type": "json_object"})
                else:
                    self.assertIs(request["return_token_ids"], True)
                request["messages"][0]["content"] = "mutated"
                self.assertEqual(original["messages"][0]["content"], "already observed evidence")
        self.assertIn("tools", original)
        self.assertNotIn("response_format", original)
        self.assertNotIn("return_token_ids", original)
        self.assertEqual(variant_request(fixture, "original", "fixture", 1234), original)
        with self.assertRaises(ValueError):
            variant_request(fixture, "unsupported", "fixture", 1234)

    def test_image_grading_never_uses_reasoning_as_answer(self):
        text = '<think>{"rate":"999"}</think>```json\n{"rate":"20.5397"}\n```'
        value = parse_object(text)
        self.assertEqual(value, {"rate": "20.5397"})
        self.assertTrue(compare_fields(value, {"rate": "20.5397"}, ["rate"])["rate"])
        self.assertFalse(
            compare_fields({"rate": True}, {"rate": "20.5397"}, ["rate"])["rate"]
        )
        self.assertTrue(
            compare_fields(
                {"city": "CIUDAD DE MEXICO"}, {"city": "Ciudad de México"}, []
            )["city"]
        )
        with self.assertRaises(ValueError):
            parse_object('<think>{"rate":"999"}')

    def test_pi_container_has_no_gpu_or_privileged_runtime(self):
        command = docker_base("test", "none")
        self.assertEqual(command[command.index("--runtime") + 1], "runc")
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertIn("NVIDIA_VISIBLE_DEVICES=void", command)
        self.assertIn("--read-only", command)
        self.assertIn("no-new-privileges", command)
        self.assertNotIn("--gpus", command)
        self.assertNotIn("--privileged", command)

    def test_pi_usage_counts_final_messages_not_stream_deltas(self):
        events = [
            {"type": "message_update", "usage": {"output": 100}},
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "usage": {"output": 12},
                    "stopReason": "stop",
                },
            },
            {"type": "agent_end", "messages": [{"usage": {"output": 12}}]},
            {"type": "tool_execution_start"},
            {"type": "tool_execution_end", "isError": True},
        ]
        summary = summarize_events(events)
        self.assertEqual(summary["output_tokens"], 12)
        self.assertEqual(summary["tool_calls"], 1)
        self.assertEqual(summary["tool_errors"], 1)
        self.assertTrue(summary["agent_end"])
        self.assertEqual(summary["final_stop_reason"], "stop")

    def test_compact_final_input_only_contains_successful_read_payloads(self):
        messages = [
            {"role": "system", "content": "guardrails"},
            {"role": "user", "content": "question and cutoff"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "read", "function": {"name": "read_chunks"}}],
            },
            {
                "role": "tool",
                "tool_call_id": "read",
                "content": json.dumps(
                    {
                        "ok": True,
                        "chunks": [
                            {
                                "chunk_id": 101,
                                "document_id": 201,
                                "text": "trusted read text",
                            }
                        ],
                    }
                ),
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "search", "function": {"name": "search_evidence"}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "search",
                "content": json.dumps(
                    {
                        "ok": True,
                        "chunks": [
                            {
                                "chunk_id": 999,
                                "document_id": 201,
                                "text": "unread snippet",
                            }
                        ],
                    }
                ),
            },
            {"role": "assistant", "content": "hallucinated answer"},
            {"role": "user", "content": "final constraints"},
        ]
        before = json.dumps(messages)
        result = compact_read_final_messages(messages)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], messages[0])
        self.assertEqual(result[1], messages[1])
        self.assertEqual(result[-1], messages[-1])
        text = json.dumps(result)
        self.assertIn("trusted read text", text)
        self.assertNotIn("unread snippet", text)
        self.assertNotIn("hallucinated answer", text)
        self.assertEqual(json.dumps(messages), before)
        messages[3]["content"] = '{"ok":false,"error":"failed"}'
        self.assertEqual(compact_read_final_messages(messages), messages)
        messages[3]["content"] = (
            '{"ok":true,"chunks":[{"chunk_id":true,"document_id":201,"text":"bad"}]}'
        )
        with self.assertRaisesRegex(ValueError, "Malformed read evidence"):
            compact_read_final_messages(messages)

    def test_review_pack_checks_reads_per_run_not_globally(self):
        def item(case, cited, read):
            return {
                "id": case,
                "category": "test",
                "question": "q",
                "reference_answer": "ref",
                "gold_documents": [{"evidence": [{"chunk_id": 101}]}],
                "run": {
                    "stop_reason": "completed",
                    "answer": {"citations": [cited]},
                    "traces": [
                        {
                            "name": "read_chunks",
                            "output": {
                                "ok": True,
                                "data": {
                                    "chunks": [{"chunk_id": read, "text": "read text"}]
                                },
                            },
                        }
                    ],
                },
            }

        rows, passages = review_rows(
            {"results": [item("first", 999, 101), item("second", 999, 999)]}
        )
        self.assertEqual(rows[0]["cited_ids_not_read_in_this_run"], [999])
        self.assertEqual(rows[1]["cited_ids_not_read_in_this_run"], [])
        self.assertEqual(rows[0]["review_verdict"], "pending")
        self.assertEqual(set(passages), {"999"})

    def test_numeric_premise_correction_not_rejected_as_failed_search(self):
        text = (
            "La superficie total no es de 50,000 m², sino de 48,661.703 m². "
            "El número de inmuebles no es 15, sino 14."
        )
        self.assertTrue(_has_affirmative_premise_correction(text))
        self.assertTrue(_has_explicit_premise_correction(text))
        payload = {"answer": text, "citations": [101], "premise_status": "false"}
        self.assertEqual(
            _parse_final_answer(json.dumps(payload), {101}).premise_status, "false"
        )
        for bad in [
            "No se encontró evidencia de 15 inmuebles.",
            "La evidencia es insuficiente para confirmar 15 inmuebles.",
            "No es 15, sino una cifra desconocida.",
            "No es 15. Se buscó otra cosa sino 14.",
        ]:
            self.assertFalse(_has_affirmative_premise_correction(bad), bad)
            self.assertFalse(_has_explicit_premise_correction(bad), bad)
            with self.assertRaises(PremiseCorrectionRequiredError):
                _parse_final_answer(json.dumps(dict(payload, answer=bad)), {101})
        with self.assertRaises(ValueError):
            _parse_final_answer(json.dumps(dict(payload, citations=["101"])), {101})
        with self.assertRaises(ValueError):
            _parse_final_answer(json.dumps(dict(payload, citations=[999])), {101})

    def test_structural_alternative_is_an_affirmative_correction(self):
        text = (
            "El decreto no contiene un artículo 99. Su texto está estructurado "
            "con artículos de PRIMERO a QUINTO más transitorios."
        )
        self.assertTrue(_has_affirmative_premise_correction(text))
        payload = {"answer": text, "citations": [101], "premise_status": "false"}
        self.assertEqual(
            _parse_final_answer(json.dumps(payload), {101}).premise_status, "false"
        )
        for bad in [
            "No se encontró un texto estructurado que confirme el artículo 99.",
            "No está estructurado en los fragmentos leídos.",
            "El texto no  está estructurado en cinco artículos.",
        ]:
            self.assertFalse(_has_affirmative_premise_correction(bad))

    def test_expropriation_is_an_affirmative_premise_correction(self):
        text = "El decreto expropia 14 inmuebles y 48,661.703 m², no 15 ni 50,000."
        self.assertTrue(_has_affirmative_premise_correction(text))
        payload = {"answer": text, "citations": [101], "premise_status": "false"}
        self.assertEqual(
            _parse_final_answer(json.dumps(payload), {101}).premise_status, "false"
        )
        for bad in [
            "No se encontró un decreto que expropia esos inmuebles.",
            "El decreto no expropia esos inmuebles.",
            "El inmueble no  se expropia en los fragmentos leídos.",
            "No se verificó una expropiación.",
            "No fue posible expropiar los inmuebles.",
        ]:
            self.assertFalse(_has_affirmative_premise_correction(bad), bad)

    def test_composition_is_an_affirmative_structural_correction(self):
        text = (
            "El decreto no contiene un artículo 99. Se compone únicamente de cinco "
            "artículos y cuatro transitorios."
        )
        self.assertTrue(_has_affirmative_premise_correction(text))
        payload = {"answer": text, "citations": [101], "premise_status": "false"}
        self.assertEqual(
            _parse_final_answer(json.dumps(payload), {101}).premise_status, "false"
        )
        for bad in [
            "No se encontró un decreto que se compone de cinco artículos.",
            "No se compone de cinco artículos.",
            "El decreto no  se compone de cinco artículos.",
            "No se pudo verificar cómo se compone de cinco artículos.",
            "Se desconoce cómo componerse de cinco artículos.",
        ]:
            self.assertFalse(_has_affirmative_premise_correction(bad), bad)

    def test_explicit_fixation_is_an_affirmative_premise_correction(self):
        text = (
            "La resolución fija los salarios para regir a partir del 1 de enero; "
            "9 de diciembre es publicación, no inicio de vigencia."
        )
        self.assertTrue(_has_affirmative_premise_correction(text))
        payload = {"answer": text, "citations": [101], "premise_status": "false"}
        self.assertEqual(
            _parse_final_answer(json.dumps(payload), {101}).premise_status, "false"
        )
        for bad in [
            "No se encontró una resolución que fija esos salarios.",
            "No fija esa fecha de inicio.",
            "La resolución no  fija esa fecha de inicio.",
            "La fecha no se fijó en el texto.",
            "La evidencia es insuficiente para fijar la fecha.",
            "La fijación de la fecha no fue posible.",
        ]:
            self.assertFalse(_has_affirmative_premise_correction(bad), bad)

    def test_comenzaron_a_regir_is_an_affirmative_effective_date(self):
        text = (
            "Los salarios mínimos de 2026 comenzaron a regir el 1º de enero de "
            "2026, y no desde su publicación del 9 de diciembre de 2025."
        )
        self.assertTrue(_has_affirmative_premise_correction(text))
        payload = {"answer": text, "citations": [101], "premise_status": "false"}
        self.assertEqual(
            _parse_final_answer(json.dumps(payload), {101}).premise_status, "false"
        )
        for bad in [
            "No se encontró evidencia de cuándo comenzaron a regir.",
            "Los salarios no  comenzaron a regir el 1 de enero.",
            "Los salarios no se comenzaron a regir en esa fecha.",
            "Los salarios nunca comenzaron a regir en esa fecha.",
            "La evidencia es insuficiente para saber cuándo comenzaron a regir.",
        ]:
            self.assertFalse(_has_affirmative_premise_correction(bad), bad)
            payload["answer"] = bad
            with self.assertRaises(PremiseCorrectionRequiredError):
                _parse_final_answer(json.dumps(payload), {101})

    def test_date_diagnostic_detects_skipping_an_unknown_first_month(self):
        for skips_unknown in (False, True):
            module = types.ModuleType("dates")

            def date(text, skips_unknown=skips_unknown):
                if "inventado" in text and skips_unknown:
                    return "2016-02-05"
                if "inventado" in text or "30 de febrero" in text:
                    raise ValueError("invalid first date")
                return "2016-09-04"

            module.publication_date = date
            output = io.StringIO()
            with patch.dict("sys.modules", {"dates": module}), redirect_stdout(output):
                runpy.run_path(
                    str(
                        Path(__file__).resolve().parents[1]
                        / "scripts/probe_pi_date_contract.py"
                    )
                )
            results = json.loads(output.getvalue())
            self.assertTrue(results[0]["passed"])
            self.assertTrue(results[1]["passed"])
            self.assertEqual(results[2]["passed"], not skips_unknown)

    def test_wire_capture_records_success_and_backend_error_latency(self):
        import ast
        import time

        import httpx

        source = Path(__file__).resolve().parents[1] / "scripts/evaluate_qwen_candidate.py"
        callback = next(
            node
            for node in ast.walk(ast.parse(source.read_text()))
            if isinstance(node, ast.FunctionDef) and node.name == "capture_response"
        )
        log = io.StringIO()
        env = {"json": json, "time": time, "http_log": log}
        exec(compile(ast.Module(body=[callback], type_ignores=[]), str(source), "exec"), env)  # noqa: S102
        payload = {"messages": [], "seed": 1234, "max_tokens": 2400}
        for status in (200, 400):
            with httpx.Client(
                transport=httpx.MockTransport(
                    lambda request, status=status: httpx.Response(
                        status,
                        stream=httpx.ByteStream(json.dumps({"status": status}).encode()),
                    )
                ),
                event_hooks={"response": [env["capture_response"]]},
            ) as client:
                client.post(
                    "http://localhost/v1/chat/completions",
                    json=payload,
                    headers={"Authorization": "Bearer secret"},
                )
        records = [json.loads(line) for line in log.getvalue().splitlines()]
        self.assertEqual([row["status"] for row in records], [200, 400])
        for row in records:
            self.assertGreaterEqual(row["request_elapsed_s"], 0)
            self.assertEqual(row["request"], payload)
        self.assertNotIn("secret", log.getvalue())

    def test_startup_budget_override_does_not_change_evaluation_budget(self):
        import ast
        import types

        source = Path(__file__).resolve().parents[1] / "scripts/evaluate_qwen_candidate.py"
        callback = next(node for node in ast.walk(ast.parse(source.read_text()))
                        if isinstance(node, ast.FunctionDef) and node.name == "seeded_create")
        for budget in (512, 2400):
            env = {"args": types.SimpleNamespace(seed=1234, startup_max_tokens=budget,
                                                thinking="on", finalization="original"),
                   "transform": None, "original_create": lambda **request: request}
            exec(compile(ast.Module(body=[callback], type_ignores=[]), str(source), "exec"), env)  # noqa: S102
            smoke = env["seeded_create"](max_tokens=512, extra_body={
                "chat_template_kwargs": {"enable_thinking": False}})
            self.assertEqual(smoke["max_tokens"], budget)
            self.assertEqual(smoke["extra_body"]["chat_template_kwargs"]["enable_thinking"],
                             budget == 2400)
            evaluation = env["seeded_create"](max_tokens=2400, tool_choice="auto")
            self.assertEqual(evaluation, {"max_tokens": 2400, "tool_choice": "auto", "seed": 1234})

    def test_no_timed_runs(self):
        summary = all_run_summary({"results": [{"error": {"type": "ProviderError"}}]})
        self.assertIsNone(summary["all_run_mean_elapsed_s"])
        self.assertIsNone(summary["effective_output_tokens_per_s"])
        self.assertEqual(summary["untimed_outcomes"], 1)

    def test_credentials_redacted_without_mutating_commands(self):
        args = [
            "serve",
            "--api-key",
            "secret",
            "--api_key=other",
            "--api-key-file",
            "/private/key",
        ]
        redacted = redact_arguments(args)
        self.assertNotIn("secret", redacted)
        self.assertIn("--api_key=<redacted>", redacted)
        self.assertEqual(args[2], "secret")
        self.assertEqual(redacted[-1], "/private/key")


if __name__ == "__main__":
    unittest.main()
