#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/src"
P=${PYTHON:-python}
K=content_fuse_1_0.5_1_k10_pop7
V=vbpr_fuse_d64v64_lr0.001_reg0.005
FR=freedom_d64_L2_k10_wi0.1_do0.8_rw0.0001_wd0_lr0.001
# VBPR (BPR-MF + 아이템 시각·텍스트 feature 투영), feature = MiniLM+CLIP 융합(fuse) 또는 없음(none, = BPR-MF)
# D010 첫 grid: reg 1e-4, lr {1e-3,5e-3}
for f in fuse none; do for lr in 1e-3 5e-3; do $P vbpr.py --features $f --lr $lr --reg 1e-4 --epochs 100 --eval-every 20; done; done
# D011 reg / lr / d grid
for r in 1e-3 1e-2 3e-2 0.1; do $P vbpr.py --features fuse --reg $r --epochs 80 --eval-every 10; done
for r in 5e-3 2e-2; do $P vbpr.py --features fuse --reg $r --epochs 100 --eval-every 20; done
$P vbpr.py --features fuse --lr 3e-3 --reg 1e-2 --epochs 100 --eval-every 20
$P vbpr.py --features fuse --d 128 --d-vis 128 --reg 1e-2 --epochs 100 --eval-every 20
$P vbpr.py --features none --reg 1e-2 --epochs 100 --eval-every 20
# D012 reg 5e-3 긴 학습 × seed 0,1,2 → 채택: e200
for s in 0 1 2; do $P vbpr.py --features fuse --reg 5e-3 --epochs 400 --eval-every 50 --seed $s; done
# D013 3-seed RRF
for e in 200 300 400; do $P blend.py --inputs ${V}_e${e}_s0 ${V}_e${e}_s1 ${V}_e${e}_s2 --name vbpr3seed_e$e; done
# 채택 설정 final / inner
for m in final inner; do
  for s in 0 1 2; do $P vbpr.py --features fuse --reg 5e-3 --epochs 200 --eval-every 200 --seed $s --mode $m; done
  $P blend.py --mode $m --inputs ${V}_e200_s0 ${V}_e200_s1 ${V}_e200_s2 --name vbpr3seed_e200
done
