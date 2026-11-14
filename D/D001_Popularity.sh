#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/src"
P=${PYTHON:-python}
K=content_fuse_1_0.5_1_k10_pop7
V=vbpr_fuse_d64v64_lr0.001_reg0.005
FR=freedom_d64_L2_k10_wi0.1_do0.8_rw0.0001_wd0_lr0.001
# D001 Popularity (키트 예제, 기준선 0.02284)
cd ..
$P teamkit/example_popularity.py --train teamkit/validation/val_train.csv --users teamkit/validation/val_users.csv --out results/popularity_val.csv
$P teamkit/teamkit.py score --pred results/popularity_val.csv --fit teamkit/validation/val_train.csv --truth teamkit/validation/val_truth.csv
