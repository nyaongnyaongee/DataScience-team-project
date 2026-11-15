#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/src"
P=${PYTHON:-python}
# D046 iALS d1024, reg grid (val)
for r in 50 100 130 200; do $P cf.py ials --d 1024 --alpha 10 --reg $r --iters 15 --seed 0; done
