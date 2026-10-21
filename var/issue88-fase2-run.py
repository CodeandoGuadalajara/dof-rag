"""Issue #88 Fase 2: preguntas fijas x3, baseline vs fase 1, contra el servidor real.

Usage (from a repo checkout; imports resolve against cwd/PYTHONPATH):
  PYTHONPATH=. .venv/bin/python var/issue88-fase2-run.py --variant baseline --out var/issue88-fase2
  PYTHONPATH=. .venv/bin/python var/issue88-fase2-run.py --variant fase1-hybrid --out var/issue88-fase2
  PYTHONPATH=. .venv/bin/python var/issue88-fase2-run.py --variant fase1-lexical --out var/issue88-fase2

Una petición de generación a la vez (secuencial); reutiliza el embedder vivo
del scheduler en :8086 sin lanzar procesos nuevos. Seed no forzado (igual que
el despliegue); la variabilidad entre repeticiones se reporta.
"""

import argparse
import json
import time
from pathlib import Path

TODAY = "2026-10-15"
MODEL = "Qwen3.8-Flash-Next"
BASE_URL = "http://127.0.0.1:8888/v1"

QUESTIONS = [
    ("Q1-apoyos-desempleo",
     "¿Qué apoyos del gobierno federal existen para una persona que perdió su empleo?"),
    ("Q2-programa-especifico",
     "¿Cuáles son las reglas de operación vigentes del Programa Jóvenes Construyendo "
     "el Futuro y cuánto es el apoyo mensual que otorga?"),
    ("Q3-consulta-historica",
     "¿Qué reforma laboral se publicó en el DOF en 2021 sobre la subcontratación "
     "y qué establecía respecto al reparto de utilidades?"),
    ("Q4-modificaciones",
     "¿Qué modificaciones han tenido las reglas de operación del Programa Jóvenes "
     "Construyendo el Futuro y cuál es la versión más reciente publicada en el DOF?"),
    ("Q5-amplia-diferente",
     "¿Qué apoyos económicos directos a personas publicó el gobierno federal en 2025?"),
]
REPS = 3


class ExistingEmbedder:
    def embed_query(self, query: str) -> bytes:
        from corpus_store.embed import PREFIX_QUERY, embed_batch, pack_binary

        return pack_binary(embed_batch([PREFIX_QUERY + query], 8086)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True,
                        choices=["baseline", "fase1-hybrid", "fase1-lexical",
                                 "fase3-hybrid"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out) / args.variant
    out.mkdir(parents=True, exist_ok=True)

    from agent_tools.agent import DofToolbox, OpenAIChatCompletionsBackend
    from agent_tools.retrieval import DofRetriever
    from human_eval import chat_runner
    from human_eval.contracts import RunRequest

    # Bases siempre del repo principal (el worktree de línea base no las tiene).
    root = Path("/home/jackbravo/Documents/dof/dof-rag")
    config = {
        "variant": args.variant,
        "today": TODAY,
        "model": MODEL,
        "base_url": BASE_URL,
        "reasoning_effort": "low",
        "enable_thinking": True,
        "seed": None,
        "code_revision": __import__("subprocess").run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        ).stdout.strip(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))

    for qid, question in QUESTIONS:
        for rep in range(1, REPS + 1):
            backend = OpenAIChatCompletionsBackend(
                model=MODEL, api_key="llama-server", base_url=BASE_URL,
                enable_thinking=True, reasoning_effort="low",
            )
            backend.max_output_tokens = 2400
            started = time.perf_counter()
            with DofRetriever(
                corpus_db=root / "dof_db/dof_corpus_l3.sqlite",
                chunks_db=root / "dof_db/dof_chunks.sqlite",
                vec0_db=root / "dof_db/dof_vec0_jina_binary.sqlite",
            ) as retriever:
                toolbox = DofToolbox(retriever, embedder=ExistingEmbedder())
                kwargs = {}
                if args.variant != "baseline":
                    kwargs["today"] = TODAY
                    kwargs["retrieval_strategy"] = (
                        "hybrid" if args.variant.endswith("hybrid") else "lexical")
                result = chat_runner.run_chat(backend, toolbox, RunRequest(question), **kwargs)
            record = {
                "qid": qid, "rep": rep, "question": question,
                "elapsed_s": round(time.perf_counter() - started, 3),
                "stop_reason": result["stop_reason"],
                "model_turns": result["model_turns"],
                "tool_calls": result["tool_calls"],
                "usage": result["usage"],
                "citations": result["answer"]["citations"],
                "reasoning_tokens": [item["tokens"] for item in result["reasoning"]],
                "searches": [
                    {"arguments": t["arguments"],
                     "strategy": (t["output"].get("data") or {}).get("strategy"),
                     "vector_candidates": (t["output"].get("data") or {}).get(
                         "vector_candidates_scanned"),
                     "documents": len((t["output"].get("data") or {}).get("documents") or []),
                     "ok": t["output"].get("ok")}
                    for t in result["traces"] if t["name"] == "search_documents"],
                "answer": result["answer"]["answer"],
            }
            path = out / f"{qid}-r{rep}.json"
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
            print(f"[{args.variant}] {qid} rep{rep}: "
                  f"{record['stop_reason']} turns={record['model_turns']} "
                  f"{record['elapsed_s']}s out={record['usage']['output_tokens']}tok",
                  flush=True)


if __name__ == "__main__":
    main()
