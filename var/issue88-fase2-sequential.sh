#!/bin/bash
# Issue #88 Fase 2: baseline -> fase1-hybrid -> fase1-lexical, secuencial.
set -u
VENV=/home/jackbravo/Documents/dof/dof-rag/.venv/bin/python
MAIN=/home/jackbravo/Documents/dof/dof-rag
BASE=/home/jackbravo/Documents/dof/dof-rag-baseline-720d1be
OUT=$MAIN/var/issue88-fase2
cd "$BASE" && PYTHONPATH="$BASE" PATH="$HOME/.local/bin:$PATH" "$VENV" "$BASE/issue88-fase2-run.py" --variant baseline --out "$OUT"
BASE_RC=$?
echo "baseline rc=$BASE_RC"
cd "$MAIN" && PYTHONPATH="$MAIN" PATH="$HOME/.local/bin:$PATH" "$VENV" "$MAIN/var/issue88-fase2-run.py" --variant fase1-hybrid --out "$OUT"
HYB_RC=$?
echo "fase1-hybrid rc=$HYB_RC"
cd "$MAIN" && PYTHONPATH="$MAIN" PATH="$HOME/.local/bin:$PATH" "$VENV" "$MAIN/var/issue88-fase2-run.py" --variant fase1-lexical --out "$OUT"
LEX_RC=$?
echo "fase1-lexical rc=$LEX_RC"
echo "DONE base=$BASE_RC hybrid=$HYB_RC lexical=$LEX_RC"
