#!/bin/bash
# D043 LightGCN·SimpleX 3-seed (42 + 0 + 1) RRF 1:1:1
cd "$(dirname "$0")/../../src"
P=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=0
until grep -q '^DONE' ../reports/20261115-team-models/run.log; do sleep 30; done
for s in 0 1; do for m in val final inner; do
  $P team_models.py lightgcn --mode $m --seed $s || echo "ISSUE D043 lgcn s$s $m"
  $P team_models.py simplex --mode $m --seed $s || echo "ISSUE D043 sx s$s $m"
done; done
for m in val final inner; do
  $P blend.py --mode $m --inputs lgcnC_e575_s42 lgcnC_e575_s0 lgcnC_e575_s1 --name lgcnC3seed || echo "ISSUE D043 lgcn3 $m"
  $P blend.py --mode $m --inputs simplexC_e650_s42 simplexC_e650_s0 simplexC_e650_s1 --name simplexC3seed || echo "ISSUE D043 sx3 $m"
done
echo DONE
