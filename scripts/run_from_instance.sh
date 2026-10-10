#!/usr/bin/env bash
# Re-run every step downstream of the success model (steps 4-12).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python}
[ -x "$PY" ] || PY=.venv/bin/python
run() { echo "=== $1"; "$PY" -u "scripts/$1" "${@:2}"; }
run 04_build_instance.py
run 05_schedule_study.py
run 05b_paired_differences.py
run 06_qaoa_study.py
run 07_hybrid_lns.py
run 09_next_measurement.py
run 08_figures.py
run 10_build_demo.py
run 11_manifest.py
run 12_forward_plan.py
echo "pipeline complete"
