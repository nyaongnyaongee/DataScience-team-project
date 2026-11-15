#!/bin/bash
cd "$(dirname "$0")/../../src"
P=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=0
# GPU가 비기를 기다림: CatBoost(D047) 끝난 뒤
until grep -qE "D047_cat|ISSUE D047" ../reports/20261115-extra/rerank_extra.log; do sleep 60; done
M=inner3
$P run_content.py --features fuse_1_0.5_1 --k 10 --pop-beta 7 --mode $M || echo "ISSUE D051 content"
for s in 0 1 2; do $P vbpr.py --features fuse --reg 5e-3 --epochs 200 --eval-every 200 --seed $s --mode $M || echo "ISSUE D051 vbpr $s"; done
for s in 0 1 2; do $P freedom.py --epochs 100 --eval-every 100 --seed $s --mode $M || echo "ISSUE D051 freedom $s"; done
V=vbpr_fuse_d64v64_lr0.001_reg0.005_e200; FR=freedom_d64_L2_k10_wi0.1_do0.8_rw0.0001_wd0_lr0.001_e100
$P blend.py --mode $M --inputs ${V}_s0 ${V}_s1 ${V}_s2 --name vbpr3seed_e200 || echo "ISSUE D051 vbpr3"
$P blend.py --mode $M --inputs ${FR}_s0 ${FR}_s1 ${FR}_s2 --name freedom3seed_e100 || echo "ISSUE D051 fr3"
$P cf.py ease --lam 500 --mode $M || echo "ISSUE D051 ease"
$P cf.py rp3b --alpha 0.4 --beta 0 --k 200 --mode $M || echo "ISSUE D051 rp3b"
for s in 0 1 2; do $P cf.py ials --d 128 --alpha 10 --reg 100 --iters 15 --seed $s --mode $M || echo "ISSUE D051 ials $s"; done
I=ials_d128_a10_r100_it15
$P blend.py --mode $M --inputs ${I}_s0 ${I}_s1 ${I}_s2 --name ials3seed || echo "ISSUE D051 ials3"
$P blend.py --mode $M --inputs content_fuse_1_0.5_1_k10_pop7 vbpr3seed_e200 freedom3seed_e100 ease_l500 rp3b_a0.4_b0_k200 ials3seed --weights 1 0.7 1.3 1 0 0 --name blend6_e1_r0_i0 || echo "ISSUE D051 blend6"
$P cf.py bm25 --k 2000 --pop 0.4 --mode $M || echo "ISSUE D051 bm25"
$P team_models.py lightgcn --mode $M || echo "ISSUE D051 lgcn"
$P team_models.py simplex --mode $M || echo "ISSUE D051 simplex"
# D052용: inner2에 LightGCN·SimpleX seed 0·1 → 3-seed
for s in 0 1; do $P team_models.py lightgcn --mode inner2 --seed $s || echo "ISSUE D052 lgcn $s"; $P team_models.py simplex --mode inner2 --seed $s || echo "ISSUE D052 sx $s"; done
$P blend.py --mode inner2 --inputs lgcnC_e575_s42 lgcnC_e575_s0 lgcnC_e575_s1 --name lgcnC3seed || echo "ISSUE D052 lgcn3"
$P blend.py --mode inner2 --inputs simplexC_e650_s42 simplexC_e650_s0 simplexC_e650_s1 --name simplexC3seed || echo "ISSUE D052 sx3"
echo BASE_DONE
until grep -q '^DONE' ../reports/20261115-extra/D048b.log; do sleep 60; done
BASE="content_fuse_1_0.5_1_k10_pop7 vbpr3seed_e200 freedom3seed_e100 ease_l500 rp3b_a0.4_b0_k200 ials3seed blend6_e1_r0_i0"
OPT="--lr 0.02 --leaves 31 --min-leaf 500 --objective binary"
$P rerank.py --models $BASE bm25_k2000_sh50_pw0.5_p0.4 lgcnC_e575_s42 simplexC_e650_s42 $OPT --train-modes inner inner2 inner3 --apply val final --name D051_rerank_team3_3split || echo "ISSUE D051 rerank"
$P rerank.py --models $BASE bm25_k4000_sh50_pw0.5_p0.4_b0.25 lgcnC3seed simplexC3seed $OPT --train-modes inner inner2 --apply val final --name D052_rerank_team3v2_2split || echo "ISSUE D052 rerank"
echo DONE
