#!/bin/bash
# D044 (topn 100), D047 (CatBoost), D045 (5-seed): D040 끝난 뒤, D040 두 변형 중 val 높은 모델 목록 사용
cd "$(dirname "$0")/../../src"
P=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=0
until grep -q '^DONE' ../reports/20261115-team-rerank/run.log; do sleep 60; done
BASE="content_fuse_1_0.5_1_k10_pop7 vbpr3seed_e200 freedom3seed_e100 ease_l500 rp3b_a0.4_b0_k200 ials3seed blend6_e1_r0_i0"
BM=bm25_k2000_sh50_pw0.5_p0.4; LG=lgcnC_e575_s42; SX=simplexC_e650_s42
T3=$(grep ',D040_rerank_team3,' ../experiments.csv | tail -1 | cut -d, -f6)
B1=$(grep ',D040_rerank_bm25,' ../experiments.csv | tail -1 | cut -d, -f6)
if $P -c "import sys; sys.exit(0 if float('$T3') >= float('$B1') else 1)"; then M="$BASE $BM $LG $SX"; TAG=team3; else M="$BASE $BM"; TAG=bm25; fi
echo "chosen $TAG (team3 $T3, bm25 $B1)"
OPT="--lr 0.02 --leaves 31 --min-leaf 500 --objective binary"
$P rerank.py --models $M $OPT --topn 100 --apply val --name D044_rerank_${TAG}_top100 || echo "ISSUE D044"
$P rerank.py --models $M --algo catboost --depth 6 --lr 0.05 --rounds 2000 --apply val --name D047_cat_${TAG}_d6 || echo "ISSUE D047"
$P rerank.py --models $M $OPT --seeds 0 1 2 3 4 --apply val final --name D045_rerank_${TAG}_5seed || echo "ISSUE D045"
echo DONE
