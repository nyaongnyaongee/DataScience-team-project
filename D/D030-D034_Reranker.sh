#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/src"
P=${PYTHON:-python}
K=content_fuse_1_0.5_1_k10_pop7
V=vbpr_fuse_d64v64_lr0.001_reg0.005
FR=freedom_d64_L2_k10_wi0.1_do0.8_rw0.0001_wd0_lr0.001
# 2단계 reranker: 모델 7개 top-50 합집합 후보 → 피처 → LightGBM
# 내부 split (reranker 학습용, val_truth 미사용): split_id b45fa59b9ee68d8a 이어야 함
(cd ../teamkit && $P teamkit.py split --train validation/val_train.csv --out ../data/inner --seed 7)
# 선행: D002-D009, D010-D013, D016-D018, D023-D026 의 "final / inner" 부분 실행
M7="$K vbpr3seed_e200 freedom3seed_e100 ease_l500 rp3b_a0.4_b0_k200 ials3seed blend6_e1_r0_i0"
# D030 LambdaRank 설정 비교
$P rerank.py --models $M7 --topn 50 --apply val --name rerank_v1
$P rerank.py --models $M7 --lr 0.01 --leaves 31 --min-leaf 500 --apply val --name rerank_lr0.01_l31_m500_lambdarank
$P rerank.py --models $M7 --lr 0.02 --leaves 15 --min-leaf 2000 --apply val --name rerank_lr0.02_l15_m2000_lambdarank
# D031 binary  → 채택 v4 (LB 0.06705)
$P rerank.py --models $M7 --lr 0.02 --leaves 31 --min-leaf 500 --objective binary --apply val final --name rerank_v4
# D032 LightGBM seed 5개 평균 (실패, +0.00011)
$P rerank.py --models $M7 --lr 0.02 --leaves 31 --min-leaf 500 --objective binary --seeds 0 1 2 3 4 --apply val --name rerank_seeds5
# D033 개수형 유저 피처 → 백분위 (v5 파일, 미제출)
$P rerank.py --models $M7 --lr 0.02 --leaves 31 --min-leaf 500 --objective binary --robust --apply val final --name rerank_robust2
# D034 입력 보강: 5-seed 입력, iALS d512, 이력 최대·최근 cosine 피처 (실패, +0.00032)
#   선행: D027 + inner/final 모드의 seed 3·4, iALS d512, blend4_5seed (reports/20261114-rerank-inputs/run.md)
$P rerank.py --models $K vbpr5seed_e200 freedom5seed_e100 ease_l500 rp3b_a0.4_b0_k200 ials_d512_a10_r100_it15_s0 blend4_5seed \
  --lr 0.02 --leaves 31 --min-leaf 500 --objective binary --extra --apply val --name rerank_inputs
