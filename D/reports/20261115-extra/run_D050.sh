#!/bin/bash
cd "$(dirname "$0")/../../src"
P=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=0
until grep -q '^DONE' ../reports/20261115-extra/D043.log && grep -q '^DONE' ../reports/20261115-extra/rerank_extra.log; do sleep 60; done
BASE="content_fuse_1_0.5_1_k10_pop7 vbpr3seed_e200 freedom3seed_e100 ease_l500 rp3b_a0.4_b0_k200 ials3seed blend6_e1_r0_i0"
OPT="--lr 0.02 --leaves 31 --min-leaf 500 --objective binary"
$P rerank.py --models $BASE bm25_k4000_sh50_pw0.5_p0.4_b0.25 lgcnC3seed simplexC3seed $OPT --apply val final --name D050_rerank_team3v2 || echo "ISSUE D050"
echo DONE
