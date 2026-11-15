#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/src"
P=${PYTHON:-python}
IN="content_fuse_1_0.5_1_k10_pop7 vbpr3seed_e200 freedom3seed_e100 ease_l500"; W="1 0.7 1.3 1"
BM=bm25_k2000_sh50_pw0.5_p0.4; LG=lgcnC_e575_s42; SX=simplexC_e650_s42
# D039 blend6 + 팀원 모델 RRF → 모두 하락
for m in BM LG SX; do for w in 0.3 0.5 1; do $P blend.py --inputs $IN ${!m} --weights $W $w --name D039_b6_${m}_w$w; done; done
$P blend.py --inputs $IN $BM $LG $SX --weights $W 0.5 0.3 0.3 --name D039_b6_all_0.5_0.3_0.3
# D041 결합 방식: RRF c, minmax, mix (모델별 top-100 minmax와 RRF 혼합; 팀원 B 방식과 다름)
for c in 10 20 30; do $P blend.py --inputs $IN --weights $W --method rrf --rrf-c $c --name D041_rrf_c$c; done
$P blend.py --inputs $IN --weights $W --method minmax --name D041_minmax
for c in 20 60; do for r in 0.25 0.5 0.75 0.9; do $P blend.py --inputs $IN --weights $W --method mix --mix $r --rrf-c $c --name D041_mix_r${r}_c$c; done; done
