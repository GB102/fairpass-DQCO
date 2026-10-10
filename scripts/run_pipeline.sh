#!/usr/bin/env bash
# Run every analysis step after the data fetch (step 1) has completed. Usage: bash scripts/run_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python}
[ -x "$PY" ] || PY=.venv/bin/python
run() { echo "=== $1"; "$PY" "scripts/$1" "${@:2}"; }
"$PY" -m tests.test_core
"$PY" -m tests.test_qaoa
run 02_validate_passes.py "*.json"
run 03_fit_link_model.py
run 04_build_instance.py
run 05_schedule_study.py
run 06_qaoa_study.py
run 07_hybrid_lns.py
run 09_next_measurement.py
run 08_figures.py
run 10_build_demo.py
run 11_manifest.py
run 12_forward_plan.py
echo "pipeline complete"
