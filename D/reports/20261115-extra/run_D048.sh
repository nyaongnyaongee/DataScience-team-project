#!/bin/bash
# D048 두 번째 내부 split(seed 8) 입력 모델 생성 → reranker를 inner+inner2로 학습
cd "$(dirname "$0")/../../src"
P=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=0
M=inner2
$P run_content.py --features fuse_1_0.5_1 --k 10 --pop-beta 7 --mode $M || echo "ISSUE D048 content"
for s in 0 1 2; do $P vbpr.py --features fuse --reg 5e-3 --epochs 200 --eval-every 200 --seed $s --mode $M || echo "ISSUE D048 vbpr $s"; done
for s in 0 1 2; do $P freedom.py --epochs 100 --eval-every 100 --seed $s --mode $M || echo "ISSUE D048 freedom $s"; done
V=vbpr_fuse_d64v64_lr0.001_reg0.005_e200; FR=freedom_d64_L2_k10_wi0.1_do0.8_rw0.0001_wd0_lr0.001_e100
$P blend.py --mode $M --inputs ${V}_s0 ${V}_s1 ${V}_s2 --name vbpr3seed_e200 || echo "ISSUE D048 vbpr3"
$P blend.py --mode $M --inputs ${FR}_s0 ${FR}_s1 ${FR}_s2 --name freedom3seed_e100 || echo "ISSUE D048 fr3"
$P cf.py ease --lam 500 --mode $M || echo "ISSUE D048 ease"
$P cf.py rp3b --alpha 0.4 --beta 0 --k 200 --mode $M || echo "ISSUE D048 rp3b"
for s in 0 1 2; do $P cf.py ials --d 128 --alpha 10 --reg 100 --iters 15 --seed $s --mode $M || echo "ISSUE D048 ials $s"; done
I=ials_d128_a10_r100_it15
$P blend.py --mode $M --inputs ${I}_s0 ${I}_s1 ${I}_s2 --name ials3seed || echo "ISSUE D048 ials3"
$P blend.py --mode $M --inputs content_fuse_1_0.5_1_k10_pop7 vbpr3seed_e200 freedom3seed_e100 ease_l500 rp3b_a0.4_b0_k200 ials3seed --weights 1 0.7 1.3 1 0 0 --name blend6_e1_r0_i0 || echo "ISSUE D048 blend6"
$P cf.py bm25 --k 2000 --pop 0.4 --mode $M || echo "ISSUE D048 bm25"
$P team_models.py lightgcn --mode $M || echo "ISSUE D048 lgcn"
$P team_models.py simplex --mode $M || echo "ISSUE D048 simplex"
echo BASE_DONE
until grep -q '^DONE' ../reports/20261115-team-rerank/run.log; do sleep 60; done
BASE="content_fuse_1_0.5_1_k10_pop7 vbpr3seed_e200 freedom3seed_e100 ease_l500 rp3b_a0.4_b0_k200 ials3seed blend6_e1_r0_i0"
BM=bm25_k2000_sh50_pw0.5_p0.4; LG=lgcnC_e575_s42; SX=simplexC_e650_s42
T3=$(grep ',D040_rerank_team3,' ../experiments.csv | tail -1 | cut -d, -f6)
B1=$(grep ',D040_rerank_bm25,' ../experiments.csv | tail -1 | cut -d, -f6)
if $P -c "import sys; sys.exit(0 if float('$T3') >= float('$B1') else 1)"; then MS="$BASE $BM $LG $SX"; TAG=team3; else MS="$BASE $BM"; TAG=bm25; fi
echo "chosen $TAG"
OPT="--lr 0.02 --leaves 31 --min-leaf 500 --objective binary"
$P rerank.py --models $MS $OPT --train-modes inner inner2 --apply val final --name D048_rerank_${TAG}_2split || echo "ISSUE D048 rerank"
$P rerank.py --models $BASE $OPT --train-modes inner inner2 --apply val --name D048_rerank_v4models_2split || echo "ISSUE D048 rerank v4models"
echo DONE
