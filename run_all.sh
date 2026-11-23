#!/usr/bin/env bash
# One command for the whole experiment:
#   generate.py (dataset, already committed) -> verify.py -> access.py run (qwen3-32b, then qwen3-30b-a3b)
#   -> conclude.py per model -> compare.py (difference between the two models) -> failures.py (cases to read by hand)
#
#   ./run_all.sh
#   DEADLINE_32B="2026-11-24 10:30" DEADLINE_30B="2026-11-24 16:00" ./run_all.sh
#
# Runs are resumable: re-running skips every request already answered. Leave a DEADLINE empty to run to completion.
# The server allows 2 parallel requests per key, so the two models run one after the other, never at the same time.
set -u
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
DEADLINE_32B=${DEADLINE_32B-2026-11-24 10:30}
DEADLINE_30B=${DEADLINE_30B-2026-11-24 16:00}

echo "== verify ground truth  [$(date +%H:%M)]"
$PY verify.py | tail -3

echo "== qwen3-32b (dense)  [$(date +%H:%M)]"
$PY access.py run --model qwen3-32b --out results_qwen3-32b.jsonl ${DEADLINE_32B:+--deadline "$DEADLINE_32B"}

echo "== qwen3-30b-a3b (MoE)  [$(date +%H:%M)]"
$PY access.py run --model qwen3-30b-a3b --out results_qwen3-30b-a3b.jsonl ${DEADLINE_30B:+--deadline "$DEADLINE_30B"}

echo "== conclusions  [$(date +%H:%M)]"
for m in qwen3-32b qwen3-30b-a3b; do
  $PY conclude.py --results results_$m.jsonl --model $m --out conclusions_$m.txt
done
$PY compare.py
$PY failures.py --n 30
echo "== done  [$(date +%H:%M)]"
