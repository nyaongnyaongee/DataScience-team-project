#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/src"
P=${PYTHON:-python}
K=content_fuse_1_0.5_1_k10_pop7
V=vbpr_fuse_d64v64_lr0.001_reg0.005
FR=freedom_d64_L2_k10_wi0.1_do0.8_rw0.0001_wd0_lr0.001
# FREEDOM (LightGCN 2층 + 고정된 멀티모달 item-item kNN 그래프, 텍스트 0.9 / 이미지 0.1)
# D016 edge dropout {0.5,0.8,0.9} × epoch (seed 0)
for d in 0.5 0.8 0.9; do $P freedom.py --dropout $d --epochs 200 --eval-every 25 --seed 0; done
# D017 seed 1,2 / reg_weight 1e-3 / 이미지 가중치 0.3
for s in 1 2; do $P freedom.py --epochs 125 --eval-every 25 --seed $s; done
$P freedom.py --reg-weight 1e-3 --epochs 125 --eval-every 25
$P freedom.py --w-img 0.3 --epochs 125 --eval-every 25
# D018 3-seed RRF (e100)  → 채택
$P blend.py --inputs ${FR}_e100_s0 ${FR}_e100_s1 ${FR}_e100_s2 --name freedom3seed_e100
for m in final inner; do
  for s in 0 1 2; do $P freedom.py --epochs 100 --eval-every 100 --seed $s --mode $m; done
  $P blend.py --mode $m --inputs ${FR}_e100_s0 ${FR}_e100_s1 ${FR}_e100_s2 --name freedom3seed_e100
done
