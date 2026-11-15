#!/bin/bash
# D041 블렌드 결합 방식 grid (blend6 입력 고정)
cd "$(dirname "$0")/../../src"
P=${PYTHON:-python}
IN="content_fuse_1_0.5_1_k10_pop7 vbpr3seed_e200 freedom3seed_e100 ease_l500"
W="1 0.7 1.3 1"
for c in 10 20 30; do $P blend.py --inputs $IN --weights $W --method rrf --rrf-c $c --name D041_rrf_c$c || echo "ISSUE rrf c$c"; done
$P blend.py --inputs $IN --weights $W --method minmax --name D041_minmax || echo "ISSUE minmax"
for c in 20 60; do for r in 0.25 0.5 0.75 0.9; do
  $P blend.py --inputs $IN --weights $W --method mix --mix $r --rrf-c $c --name D041_mix_r${r}_c$c || echo "ISSUE mix r$r c$c"
done; done
echo DONE
