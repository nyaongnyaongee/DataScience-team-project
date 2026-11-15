#!/bin/bash
cd "$(dirname "$0")/../../src"
P=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=0
until grep -q '^DONE' ../reports/20261115-extra/queue.log; do sleep 60; done
BASE="content_fuse_1_0.5_1_k10_pop7 vbpr3seed_e200 freedom3seed_e100 ease_l500 rp3b_a0.4_b0_k200 ials3seed blend6_e1_r0_i0"
T3="bm25_k2000_sh50_pw0.5_p0.4 lgcnC_e575_s42 simplexC_e650_s42"
T3v2="bm25_k4000_sh50_pw0.5_p0.4_b0.25 lgcnC3seed simplexC3seed"
OPT="--lr 0.02 --leaves 31 --min-leaf 500 --objective binary"
echo "== D048c"
$P rerank.py --models $BASE $T3 $OPT --train-modes inner inner2 --apply val --name D048c_rerank_team3_2split_fix || echo "ISSUE D048c"
g() { grep ",$1," ../experiments.csv | tail -1 | cut -d, -f6; }
best=$($P - <<PY
c={"D048c":("$T3","inner inner2"),"D051":("$T3","inner inner2 inner3"),"D052":("$T3v2","inner inner2")}
v={"D048c":"$(g D048c_rerank_team3_2split_fix)","D051":"$(g D051_rerank_team3_3split)","D052":"$(g D052_rerank_team3v2_2split)"}
v={k:float(x) for k,x in v.items() if x}
print(max(v,key=v.get))
PY
)
echo "== D053 chosen $best"
case $best in
  D048c) M="$T3"; TM="inner inner2";;
  D051) M="$T3"; TM="inner inner2 inner3";;
  D052) M="$T3v2"; TM="inner inner2";;
esac
$P rerank.py --models $BASE $M $OPT --train-modes $TM --seeds 0 1 2 3 4 --apply val final --name D053_rerank_${best}_5seed || echo "ISSUE D053"
echo DONE
