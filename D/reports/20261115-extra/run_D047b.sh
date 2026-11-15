#!/bin/bash
cd "$(dirname "$0")/../../src"
P=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=0
until grep -q '^DONE' ../reports/20261115-extra/rerank_extra.log; do sleep 60; done
BASE="content_fuse_1_0.5_1_k10_pop7 vbpr3seed_e200 freedom3seed_e100 ease_l500 rp3b_a0.4_b0_k200 ials3seed blend6_e1_r0_i0"
$P rerank.py --models $BASE bm25_k2000_sh50_pw0.5_p0.4 lgcnC_e575_s42 simplexC_e650_s42 --algo catboost --depth 6 --lr 0.05 --rounds 2000 --apply val --name D047_cat_team3_d6 || echo "ISSUE D047b"
echo DONE
