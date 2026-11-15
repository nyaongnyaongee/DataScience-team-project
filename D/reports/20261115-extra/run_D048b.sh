#!/bin/bash
# D048 두 번째 변형 재실행 (OOM 후). D047 CatBoost가 끝난 뒤.
cd "$(dirname "$0")/../../src"
P=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=0
until grep -qE "D047_cat|ISSUE D047" ../reports/20261115-extra/rerank_extra.log; do sleep 60; done
BASE="content_fuse_1_0.5_1_k10_pop7 vbpr3seed_e200 freedom3seed_e100 ease_l500 rp3b_a0.4_b0_k200 ials3seed blend6_e1_r0_i0"
OPT="--lr 0.02 --leaves 31 --min-leaf 500 --objective binary"
$P rerank.py --models $BASE $OPT --train-modes inner inner2 --apply val --name D048_rerank_v4models_2split || echo "ISSUE D048b"
echo DONE
