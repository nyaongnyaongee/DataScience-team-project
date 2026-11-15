#!/bin/bash
# D046 iALS d1024 (val), D043 끝난 뒤 GPU 사용
cd "$(dirname "$0")/../../src"
P=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=0
until grep -q '^DONE' ../reports/20261115-extra/D043.log; do sleep 30; done
for r in 100 50; do /usr/bin/time -v $P cf.py ials --d 1024 --alpha 10 --reg $r --iters 15 --seed 0 2>&1 | grep -E "recall|Elapsed|Error|error" || echo "ISSUE D046 r$r"; done
echo DONE
