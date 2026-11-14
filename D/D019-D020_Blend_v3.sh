#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/src"
P=${PYTHON:-python}
K=content_fuse_1_0.5_1_k10_pop7
V=vbpr_fuse_d64v64_lr0.001_reg0.005
FR=freedom_d64_L2_k10_wi0.1_do0.8_rw0.0001_wd0_lr0.001
# D019 kNN + VBPR3 + FREEDOM 단일 seed(s0) 가중치
for vf in "1 0.5" "1 1" "1 1.5" "1 2" "0.5 1" "0 1"; do set -- $vf
  $P blend.py --inputs $K vbpr3seed_e200 ${FR}_e100_s0 --weights 1 $1 $2 --name blend3_k1_v$1_f$2; done
# D020 kNN + VBPR3 + FREEDOM3  → 채택 v3: 1 : 0.7 : 1.3 (LB 0.06068)
for vf in "1 1" "1 1.5" "0.7 1.3" "0.7 1" "0.5 1.3" "0.7 1.6"; do set -- $vf
  $P blend.py --inputs $K vbpr3seed_e200 freedom3seed_e100 --weights 1 $1 $2 --name blend3s_k1_v$1_f$2; done
$P blend.py --mode final --inputs $K vbpr3seed_e200 freedom3seed_e100 --weights 1 0.7 1.3 --name blend3s_k1_v0.7_f1.3
