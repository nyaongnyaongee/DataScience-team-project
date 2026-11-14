#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/src"
P=${PYTHON:-python}
K=content_fuse_1_0.5_1_k10_pop7
V=vbpr_fuse_d64v64_lr0.001_reg0.005
FR=freedom_d64_L2_k10_wi0.1_do0.8_rw0.0001_wd0_lr0.001
# D027 seed 3 → 5개
for s in 3 4; do $P vbpr.py --features fuse --reg 5e-3 --epochs 200 --eval-every 200 --seed $s; done
for s in 3 4; do $P freedom.py --epochs 100 --eval-every 100 --seed $s; done
$P blend.py --inputs ${V}_e200_s0 ${V}_e200_s1 ${V}_e200_s2 ${V}_e200_s3 ${V}_e200_s4 --name vbpr5seed_e200
$P blend.py --inputs ${FR}_e100_s0 ${FR}_e100_s1 ${FR}_e100_s2 ${FR}_e100_s3 ${FR}_e100_s4 --name freedom5seed_e100
$P blend.py --inputs $K vbpr5seed_e200 freedom5seed_e100 ease_l500 --weights 1 0.7 1.3 1 --name blend4_5seed
# D028 활동량 그룹별 가중치 (kNN × FREEDOM grid, VBPR 0.7 / EASE 1 고정)
for k in 0.5 1 1.5; do for f in 0.8 1.3 2; do
  $P blend.py --inputs $K vbpr3seed_e200 freedom3seed_e100 ease_l500 --weights $k 0.7 $f 1 --name gw_k${k}_f$f; done; done
$P ../D028_group_combo.py
