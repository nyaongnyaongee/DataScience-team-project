#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/src"
P=${PYTHON:-python}
K=content_fuse_1_0.5_1_k10_pop7
V=vbpr_fuse_d64v64_lr0.001_reg0.005
FR=freedom_d64_L2_k10_wi0.1_do0.8_rw0.0001_wd0_lr0.001
# D023 EASE λ
for l in 50 100 200 300 500 700 1000 2000 5000; do $P cf.py ease --lam $l; done
# D024 RP3β
for a in 0.5 0.75 1.0; do for b in 0 0.2 0.4 0.6; do $P cf.py rp3b --alpha $a --beta $b --k 200; done; done
for a in 0.25 0.4; do $P cf.py rp3b --alpha $a --beta 0 --k 200; done
for k in 100 500; do $P cf.py rp3b --alpha 0.5 --beta 0 --k $k; done
$P cf.py rp3b --alpha 0.5 --beta 0.1 --k 200
# D025 iALS
for d in 64 128; do for al in 1 10 40; do for r in 1 10 100; do $P cf.py ials --d $d --alpha $al --reg $r --iters 15 --seed 0; done; done; done
for r in 150 200 300 1000 3000; do $P cf.py ials --d 128 --alpha 10 --reg $r --iters 15; done
$P cf.py ials --d 128 --alpha 20 --reg 300 --iters 15
for s in 1 2; do $P cf.py ials --d 128 --alpha 10 --reg 100 --iters 15 --seed $s; done
I=ials_d128_a10_r100_it15
$P blend.py --inputs ${I}_s0 ${I}_s1 ${I}_s2 --name ials3seed
for r in 50 100 150; do $P cf.py ials --d 256 --alpha 10 --reg $r --iters 15; done
for r in 100 150; do $P cf.py ials --d 512 --alpha 10 --reg $r --iters 15; done
# D026 v3 + EASE / RP3β / iALS3  → blend6_e1_r0_i0 (reranker 입력)
for eri in "0.5 0 0" "0.5 0 0.5" "0.5 0.5 0" "0.5 0.5 0.5" "1 0 0" "1 0 0.5" "1 0.5 0" "1 0.5 0.5" "0.7 0 0" "1.5 0 0" "2 0 0" "1 0 0.25" "1 0.25 0"; do set -- $eri
  $P blend.py --inputs $K vbpr3seed_e200 freedom3seed_e100 ease_l500 rp3b_a0.4_b0_k200 ials3seed --weights 1 0.7 1.3 $1 $2 $3 --name blend6_e$1_r$2_i$3; done
# D029 v3 + EASE + iALS d512
for w in 0.25 0.5; do $P blend.py --inputs $K vbpr3seed_e200 freedom3seed_e100 ease_l500 ials_d512_a10_r100_it15_s0 --weights 1 0.7 1.3 1 $w --name blend5_ials512_$w; done
# 채택 설정 final / inner (reranker용)
for m in final inner; do
  $P cf.py ease --lam 500 --mode $m
  $P cf.py rp3b --alpha 0.4 --beta 0 --k 200 --mode $m
  for s in 0 1 2; do $P cf.py ials --d 128 --alpha 10 --reg 100 --iters 15 --seed $s --mode $m; done
  $P blend.py --mode $m --inputs ${I}_s0 ${I}_s1 ${I}_s2 --name ials3seed
  $P blend.py --mode $m --inputs $K vbpr3seed_e200 freedom3seed_e100 ease_l500 rp3b_a0.4_b0_k200 ials3seed --weights 1 0.7 1.3 1 0 0 --name blend6_e1_r0_i0
done
