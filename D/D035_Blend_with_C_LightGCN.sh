#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/src"
P=${PYTHON:-python}
K=content_fuse_1_0.5_1_k10_pop7
V=vbpr_fuse_d64v64_lr0.001_reg0.005
FR=freedom_d64_L2_k10_wi0.1_do0.8_rw0.0001_wd0_lr0.001
# D035 팀원 C LightGCN top-100 (results/C_lightgcn_val_top100.csv.gz) 을 RRF로 추가 (실패)
for w in 0.3 0.5 1.0; do
  $P blend.py --inputs $K vbpr3seed_e200 freedom3seed_e100 C_lightgcn --weights 1 0.7 1.3 $w --name v3_lgcnC_w$w
  $P blend.py --inputs $K vbpr3seed_e200 freedom3seed_e100 ease_l500 C_lightgcn --weights 1 0.7 1.3 1 $w --name blend6_lgcnC_w$w
done
