#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/src"
P=${PYTHON:-python}
K=content_fuse_1_0.5_1_k10_pop7
V=vbpr_fuse_d64v64_lr0.001_reg0.005
FR=freedom_d64_L2_k10_wi0.1_do0.8_rw0.0001_wd0_lr0.001
# RRF 블렌드: Σ w / (60 + rank)
# D014 콘텐츠 kNN + VBPR 단일 seed  → 채택 v1: VBPR e200 s0, 1:1 (LB 0.05081)
for w in 0.3 0.5 1; do $P blend.py --inputs $K ${V}_e100_s0 --weights 1 $w --name blend_rrf_knn_vbpr_w$w; done
for w in 0.3 0.5 1; do $P blend.py --method minmax --inputs $K ${V}_e100_s0 --weights 1 $w --name blend_mm_knn_vbpr_w$w; done
for s in 0 1 2; do $P blend.py --inputs $K ${V}_e200_s$s --weights 1 0.7 --name blend_rrf_knn_vbpr200_w0.7_s$s; done
for w in 0.5 1 1.5 2; do $P blend.py --inputs $K ${V}_e200_s0 --weights 1 $w --name blend_rrf_knn_vbpr200_w${w}_s0; done
$P blend.py --mode final --inputs $K ${V}_e200_s0 --weights 1 1 --name blend_rrf_knn_vbpr200_w1_s0
# D015 콘텐츠 kNN + VBPR 3-seed  → 채택 v2: 1:1 (LB 0.05268)
for w in 0.7 0.9 0.95 1 1.05 1.1 1.3; do $P blend.py --inputs $K vbpr3seed_e200 --weights 1 $w --name blend_knn_vbpr3seed_e200_w$w; done
for w in 0.7 1 1.3; do $P blend.py --inputs $K vbpr3seed_e400 --weights 1 $w --name blend_knn_vbpr3seed_e400_w$w; done
$P blend.py --mode final --inputs $K vbpr3seed_e200 --weights 1 1 --name blend_knn_vbpr3seed_e200_w1
