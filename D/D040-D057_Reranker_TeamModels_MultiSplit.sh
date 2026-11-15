#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/src"
P=${PYTHON:-python}
BASE="content_fuse_1_0.5_1_k10_pop7 vbpr3seed_e200 freedom3seed_e100 ease_l500 rp3b_a0.4_b0_k200 ials3seed blend6_e1_r0_i0"
T3="bm25_k2000_sh50_pw0.5_p0.4 lgcnC_e575_s42 simplexC_e650_s42"
T3v2="bm25_k4000_sh50_pw0.5_p0.4_b0.25 lgcnC3seed simplexC3seed"
OPT="--lr 0.02 --leaves 31 --min-leaf 500 --objective binary"   # v4(D031)와 같은 설정

# ---- 추가 내부 split: val_train.csv를 키트 split으로 다시 나눔 (seed만 다름). val_truth는 쓰지 않음.
# inner = seed 7 (v4). 아래는 teamkit/ 에서 실행.
#   python teamkit.py split --train validation/val_train.csv --out ../data/inner2 --seed 8    # 3072f6272361b6b9
#   ... inner3 seed 9 (37bca72ea30afb64), inner4 seed 10 (ff709aa94dc99ea7), inner5 seed 11 (2d1200ceb4f41eee)
#   ... inner6 seed 12 (2a741bca91e25c77), inner7 seed 13 (942b9551f6402225), inner8 seed 14 (94020376f1d805f5)
base_models() {   # 내부 split M에 reranker 입력 모델 10개 생성 (v4 7개 + 팀원 3개)
  M=$1; V=vbpr_fuse_d64v64_lr0.001_reg0.005_e200; FR=freedom_d64_L2_k10_wi0.1_do0.8_rw0.0001_wd0_lr0.001_e100; I=ials_d128_a10_r100_it15
  $P run_content.py --features fuse_1_0.5_1 --k 10 --pop-beta 7 --mode $M
  for s in 0 1 2; do $P vbpr.py --features fuse --reg 5e-3 --epochs 200 --eval-every 200 --seed $s --mode $M; done
  for s in 0 1 2; do $P freedom.py --epochs 100 --eval-every 100 --seed $s --mode $M; done
  $P blend.py --mode $M --inputs ${V}_s0 ${V}_s1 ${V}_s2 --name vbpr3seed_e200
  $P blend.py --mode $M --inputs ${FR}_s0 ${FR}_s1 ${FR}_s2 --name freedom3seed_e100
  $P cf.py ease --lam 500 --mode $M
  $P cf.py rp3b --alpha 0.4 --beta 0 --k 200 --mode $M
  for s in 0 1 2; do $P cf.py ials --d 128 --alpha 10 --reg 100 --iters 15 --seed $s --mode $M; done
  $P blend.py --mode $M --inputs ${I}_s0 ${I}_s1 ${I}_s2 --name ials3seed
  $P blend.py --mode $M --inputs content_fuse_1_0.5_1_k10_pop7 vbpr3seed_e200 freedom3seed_e100 ease_l500 rp3b_a0.4_b0_k200 ials3seed --weights 1 0.7 1.3 1 0 0 --name blend6_e1_r0_i0
  $P cf.py bm25 --k 2000 --pop 0.4 --mode $M
  $P team_models.py lightgcn --mode $M
  $P team_models.py simplex --mode $M
}
for M in inner2 inner3 inner4 inner5 inner6 inner7 inner8; do base_models $M; done

# D040 v4 reranker + 팀원 모델 (inner 1개로 학습)
$P rerank.py --models $BASE $T3 $OPT --apply val final --name D040_rerank_team3
$P rerank.py --models $BASE bm25_k2000_sh50_pw0.5_p0.4 $OPT --apply val final --name D040_rerank_bm25
# D044 후보 모델당 100개
$P rerank.py --models $BASE $T3 $OPT --topn 100 --apply val --name D044_rerank_team3_top100
# D045 D040 5-seed
$P rerank.py --models $BASE $T3 $OPT --seeds 0 1 2 3 4 --apply val final --name D045_rerank_team3_5seed
# D047 CatBoost YetiRank (GPU)
$P rerank.py --models $BASE $T3 --algo catboost --depth 6 --lr 0.05 --rounds 2000 --apply val --name D047_cat_team3_d6
# D049 + SGL·NCL
$P rerank.py --models $BASE $T3 sglC_e525_s42 nclC_e425_s42 $OPT --apply val --name D049_rerank_team5
# D050 튜닝 BM25 + 3-seed LightGCN·SimpleX
$P rerank.py --models $BASE $T3v2 $OPT --apply val final --name D050_rerank_team3v2
# D048 내부 split 2개로 학습 (D048c = early stopping 유저 분리 수정 후 재실행)
$P rerank.py --models $BASE $T3 $OPT --train-modes inner inner2 --apply val --name D048c_rerank_team3_2split_fix
$P rerank.py --models $BASE $OPT --train-modes inner inner2 --apply val --name D048_rerank_v4models_2split
# D051 split 3개, D052 T3v2 + split 2개
$P rerank.py --models $BASE $T3 $OPT --train-modes inner inner2 inner3 --apply val final --name D051_rerank_team3_3split
$P rerank.py --models $BASE $T3v2 $OPT --train-modes inner inner2 --apply val final --name D052_rerank_team3v2_2split
# D053 D051 설정 5-seed  ← 현재 제출 후보
$P rerank.py --models $BASE $T3 $OPT --train-modes inner inner2 inner3 --seeds 0 1 2 3 4 --apply val final --name D053_rerank_D051_5seed
# D054 split 5개, D055 split 5개 5-seed
$P rerank.py --models $BASE $T3 $OPT --train-modes inner inner2 inner3 inner4 inner5 --apply val final --name D054_rerank_team3_5split
$P rerank.py --models $BASE $T3 $OPT --train-modes inner inner2 inner3 inner4 inner5 --seeds 0 1 2 3 4 --apply val final --name D055_rerank_5split_5seed
# D056 split 8개
$P rerank.py --models $BASE $T3 $OPT --train-modes inner inner2 inner3 inner4 inner5 inner6 inner7 inner8 --apply val final --name D056_rerank_8split
# D057 split 3개, leaves 63, min_leaf 1000
$P rerank.py --models $BASE $T3 --lr 0.02 --leaves 63 --min-leaf 1000 --objective binary --train-modes inner inner2 inner3 --apply val --name D057_rerank_3split_l63m1000
