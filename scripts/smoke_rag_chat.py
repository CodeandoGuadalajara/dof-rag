"""Development-only real-server smoke test; uses a temporary evaluation DB.

Run: python -m scripts.smoke_rag_chat
"""
import tempfile
from pathlib import Path

from human_eval.agent_executor import AgentExecutorConfig, AgentRunExecutor
from human_eval.contracts import RunRequest
from human_eval.scheduler import execute_claimed_run
from human_eval.store import EvaluationStore


def main():
    with tempfile.TemporaryDirectory() as directory:
        store = EvaluationStore(Path(directory) / "smoke.sqlite")
        store.initialize()
        executor = AgentRunExecutor(AgentExecutorConfig.from_env(Path.cwd()))
        try:
            questions = [
                "¿Cuál fue el tipo de cambio obtenido el 9 de agosto de 2006 según el DOF?",
                "¿En qué fecha se publicó ese aviso?",
            ]
            charged = 0
            for i, question in enumerate(questions):
                remaining = store.token_balance("smoke")
                run, _ = store.create_run(RunRequest(question, client_request_id=f"smoke-{i}"),
                                          user_id="smoke", reserved_tokens=remaining)
                run_id = run["run_id"]
                request = store.get_request(run_id)
                print("History messages:", len(request.history), "Reserved:", remaining, flush=True)
                store.start_run(run_id, provenance=executor.provenance())
                execute_claimed_run(store, executor, run_id)
                run = store.get_run(run_id)
                print("Status:", run["status"], "Balance:", store.token_balance("smoke"), flush=True)
                assert run["status"] == "succeeded", run.get("error")
                result = run["result"]
                print("Answer:", result["answer"], "Usage:", result["usage"], flush=True)
                for trace in result.get("trace", []):
                    print("Tool:", trace["name"], "Arguments:", trace.get("arguments"),
                          "OK:", trace["output"].get("ok"), "Error:", trace["output"].get("error"), flush=True)
                print("Warnings:", result.get("warnings"), flush=True)
                assert result["answer"]["citation_ids"]
                used = result["usage"]["total_tokens"]
                assert 0 < used <= remaining
                charged += used
                assert store.token_balance("smoke") == 50_000 - charged
                if i:
                    assert request.history
        finally:
            executor.close()


if __name__ == "__main__":
    main()
