#!/bin/bash
# D040 reranker + 팀원 모델. v4 설정 그대로, 모델 목록만 다름.
cd "$(dirname "$0")/../../src"
P=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=0
until grep -q '^DONE' ../reports/20261115-team-models/run.log; do sleep 30; done
BASE="content_fuse_1_0.5_1_k10_pop7 vbpr3seed_e200 freedom3seed_e100 ease_l500 rp3b_a0.4_b0_k200 ials3seed blend6_e1_r0_i0"
BM=bm25_k2000_sh50_pw0.5_p0.4; LG=lgcnC_e575_s42; SX=simplexC_e650_s42
OPT="--lr 0.02 --leaves 31 --min-leaf 500 --objective binary --apply val final"
$P rerank.py --models $BASE $BM $LG $SX $OPT --name D040_rerank_team3 || echo "ISSUE D040 team3"
$P rerank.py --models $BASE $BM $OPT --name D040_rerank_bm25 || echo "ISSUE D040 bm25"
echo DONE
