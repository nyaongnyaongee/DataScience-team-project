#!/bin/bash
# D049 C SGL·NCL 이식 (val/final/inner/inner2) → 블렌드·reranker 확인
cd "$(dirname "$0")/../../src"
P=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=0
until grep -q '^DONE' ../reports/20261115-extra/D043.log; do sleep 30; done
for m in val final inner inner2; do
  $P team_models.py sgl --mode $m || echo "ISSUE D049 sgl $m"
  $P team_models.py ncl --mode $m || echo "ISSUE D049 ncl $m"
done
echo MODELS_DONE
until grep -q '^DONE' ../reports/20261115-extra/D048.log; do sleep 60; done
BASE="content_fuse_1_0.5_1_k10_pop7 vbpr3seed_e200 freedom3seed_e100 ease_l500 rp3b_a0.4_b0_k200 ials3seed blend6_e1_r0_i0"
BM=bm25_k2000_sh50_pw0.5_p0.4; LG=lgcnC_e575_s42; SX=simplexC_e650_s42
OPT="--lr 0.02 --leaves 31 --min-leaf 500 --objective binary"
$P rerank.py --models $BASE $BM $LG $SX sglC_e525_s42 nclC_e425_s42 $OPT --apply val --name D049_rerank_team5 || echo "ISSUE D049 rerank"
echo DONE
