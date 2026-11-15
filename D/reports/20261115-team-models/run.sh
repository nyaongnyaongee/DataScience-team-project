#!/bin/bash
# D036-D038 팀원 모델 이식: val/final/inner top-100
cd "$(dirname "$0")/../../src"
P=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=0
set -x
for m in val final inner; do $P cf.py bm25 --k 2000 --pop 0.4 --mode $m || echo "ISSUE bm25 $m"; done
for m in val final inner; do $P team_models.py lightgcn --mode $m || echo "ISSUE lightgcn $m"; done
for m in val final inner; do $P team_models.py simplex --mode $m || echo "ISSUE simplex $m"; done
echo DONE
