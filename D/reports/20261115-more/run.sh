#!/bin/bash
cd "$(dirname "$0")/../../src"
P=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=0
BASE="content_fuse_1_0.5_1_k10_pop7 vbpr3seed_e200 freedom3seed_e100 ease_l500 rp3b_a0.4_b0_k200 ials3seed blend6_e1_r0_i0"
T3="bm25_k2000_sh50_pw0.5_p0.4 lgcnC_e575_s42 simplexC_e650_s42"
OPT="--lr 0.02 --leaves 31 --min-leaf 500 --objective binary"
OPT2="--lr 0.02 --leaves 63 --min-leaf 1000 --objective binary"
S3="inner inner2 inner3"; S5="inner inner2 inner3 inner4 inner5"; S8="$S5 inner6 inner7 inner8"
$P rerank.py --models $BASE $T3 $OPT2 --train-modes $S3 --apply val --name D057_rerank_3split_l63m1000 || echo "ISSUE D057"
$P rerank.py --models $BASE $T3 $OPT --train-modes $S5 --seeds 0 1 2 3 4 --apply val final --name D055_rerank_5split_5seed || echo "ISSUE D055"
V=vbpr_fuse_d64v64_lr0.001_reg0.005_e200; FR=freedom_d64_L2_k10_wi0.1_do0.8_rw0.0001_wd0_lr0.001_e100; I=ials_d128_a10_r100_it15
for M in inner6 inner7 inner8; do
  $P run_content.py --features fuse_1_0.5_1 --k 10 --pop-beta 7 --mode $M || echo "ISSUE D056 content $M"
  for s in 0 1 2; do $P vbpr.py --features fuse --reg 5e-3 --epochs 200 --eval-every 200 --seed $s --mode $M || echo "ISSUE D056 vbpr $s $M"; done
  for s in 0 1 2; do $P freedom.py --epochs 100 --eval-every 100 --seed $s --mode $M || echo "ISSUE D056 freedom $s $M"; done
  $P blend.py --mode $M --inputs ${V}_s0 ${V}_s1 ${V}_s2 --name vbpr3seed_e200 || echo "ISSUE D056 vbpr3 $M"
  $P blend.py --mode $M --inputs ${FR}_s0 ${FR}_s1 ${FR}_s2 --name freedom3seed_e100 || echo "ISSUE D056 fr3 $M"
  $P cf.py ease --lam 500 --mode $M || echo "ISSUE D056 ease $M"
  $P cf.py rp3b --alpha 0.4 --beta 0 --k 200 --mode $M || echo "ISSUE D056 rp3b $M"
  for s in 0 1 2; do $P cf.py ials --d 128 --alpha 10 --reg 100 --iters 15 --seed $s --mode $M || echo "ISSUE D056 ials $s $M"; done
  $P blend.py --mode $M --inputs ${I}_s0 ${I}_s1 ${I}_s2 --name ials3seed || echo "ISSUE D056 ials3 $M"
  $P blend.py --mode $M --inputs content_fuse_1_0.5_1_k10_pop7 vbpr3seed_e200 freedom3seed_e100 ease_l500 rp3b_a0.4_b0_k200 ials3seed --weights 1 0.7 1.3 1 0 0 --name blend6_e1_r0_i0 || echo "ISSUE D056 blend6 $M"
  $P cf.py bm25 --k 2000 --pop 0.4 --mode $M || echo "ISSUE D056 bm25 $M"
  $P team_models.py lightgcn --mode $M || echo "ISSUE D056 lgcn $M"
  $P team_models.py simplex --mode $M || echo "ISSUE D056 simplex $M"
done
echo BASE_DONE
$P rerank.py --models $BASE $T3 $OPT --train-modes $S8 --apply val final --name D056_rerank_8split || echo "ISSUE D056 rerank"
echo DONE
