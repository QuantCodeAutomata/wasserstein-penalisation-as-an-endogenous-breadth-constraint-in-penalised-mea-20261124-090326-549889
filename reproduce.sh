#!/usr/bin/env bash
# Runs the full test suite and every experiment of this replication, in the
# same commands as README.md. A-share-panel experiments (exp2, exp4-exp8)
# download data from the Hugging Face Hub the first time they run.
set -uo pipefail

python -m pip install -r requirements.txt

python -m pytest -m "not market_data"

python -m exp.exp1
python -m exp.exp2
python -m exp.exp3
python -m exp.exp4
python -m exp.exp5
python -m exp.exp6
python -m exp.exp7
python -m exp.exp8
python -m exp.exp9
