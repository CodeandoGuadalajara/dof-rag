"""Freeze the three DOF evaluation databases and question/agent provenance.

SQLite backup handles journal/WAL content. Reject a bundle if any database changes
during the multi-file snapshot; never stop the unrelated daily-update service.
"""

import argparse
import hashlib
import inspect
import json
import shutil
import sqlite3
from contextlib import ExitStack, closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATABASES = ("dof_corpus_l3.sqlite", "dof_chunks.sqlite", "dof_vec0_jina_binary.sqlite")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def freeze(source, output):
    output.mkdir(exist_ok=False, parents=True)
    records = {}
    with ExitStack() as stack:
        readers = {}
        versions = {}
        for name in DATABASES:
            connection = sqlite3.connect(
                (source / name).resolve().as_uri() + "?mode=ro", uri=True
            )
            stack.callback(connection.close)
            readers[name] = connection
            versions[name] = connection.execute("PRAGMA data_version").fetchone()[0]
        for name, connection in readers.items():
            target = output / name
            with closing(sqlite3.connect(target)) as destination:
                connection.backup(destination)
            records[name] = {
                "source": str((source / name).resolve()),
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
                "data_version": versions[name],
            }
            print(f"Frozen {name}: {target.stat().st_size / 2**30:.2f} GiB", flush=True)
        for name, connection in readers.items():
            assert (
                connection.execute("PRAGMA data_version").fetchone()[0]
                == versions[name]
            ), f"{name} changed during snapshot; bundle is NOT valid"
    return records


def self_check():
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        source = Path(directory) / "source"
        source.mkdir()
        for name in DATABASES:
            with sqlite3.connect(source / name) as connection:
                connection.execute("CREATE TABLE facts(value TEXT)")
                connection.execute("INSERT INTO facts VALUES ('preserved')")
        output = Path(directory) / "snapshot"
        records = freeze(source, output)
        assert set(records) == set(DATABASES)
        for name in DATABASES:
            with sqlite3.connect(output / name) as connection:
                assert (
                    connection.execute("SELECT value FROM facts").fetchone()[0]
                    == "preserved"
                )
        try:
            freeze(source, output)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Existing bundle was overwritten")
    print(
        "SQLite backup integrity and overwrite-refusal self-check passed.", flush=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.output is None:
        parser.error("--output required")
    records = freeze(ROOT / "dof_db", args.output)
    source_paths = [
        ROOT / "var/grounded-finalization-2026-09-14/agent.py.snapshot",
        ROOT / "agent_tools/agent.py",
        ROOT / "agent_tools/retrieval.py",
        ROOT / "scripts/eval_v4_agent.py",
        ROOT / "eval/dof_queries_v4.jsonl",
    ]
    sources = {}
    for source in source_paths:
        relative = source.relative_to(ROOT)
        target = args.output / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        assert sha256(source) == sha256(target), (
            f"Source changed while copied: {source}"
        )
        sources[str(relative)] = sha256(target)
    # Import only after copying to ensure the normalizer source is recorded verbatim.
    import sys

    sys.path.insert(0, str(ROOT))
    from agent_tools.agent import _normalize_nullable_literals

    normalizer = inspect.getsource(_normalize_nullable_literals)
    manifest = {
        "databases": records,
        "sources": sources,
        "normalizer_source": normalizer,
        "normalizer_sha256": hashlib.sha256(normalizer.encode()).hexdigest(),
        "limitations": [
            "SQLite data_version verifies no commits during collection; "
            "this is not a production database backup policy.",
            "Evaluators must explicitly select these paths, not live dof_db defaults.",
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for path in args.output.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    print(f"Verified input bundle: {args.output}", flush=True)


if __name__ == "__main__":
    main()
