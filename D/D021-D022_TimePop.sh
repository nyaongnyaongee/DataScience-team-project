#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/src"
P=${PYTHON:-python}
K=content_fuse_1_0.5_1_k10_pop7
V=vbpr_fuse_d64v64_lr0.001_reg0.005
FR=freedom_d64_L2_k10_wi0.1_do0.8_rw0.0001_wd0_lr0.001
# D021 유저 활동 기간 인기도 (bin 30일, 가우시안 σ / window pad / γ / bin 7일)
for s in 0.5 1 2 4 8 16; do $P timepop.py --sigma $s; done
for p in 0 3 12; do $P timepop.py --kernel window --pad $p; done
for g in 0.3 0.6; do $P timepop.py --sigma 4 --gamma $g; done
$P timepop.py --sigma 4 --bin-days 7
# D022 v3 + timepop (실패)
for w in 0.2 0.5 1; do $P blend.py --inputs $K vbpr3seed_e200 freedom3seed_e100 timepop_b30_g8_gm0_pb0 --weights 1 0.7 1.3 $w --name blend3s_tp$w; done
