#!/bin/bash
# D039 blend6 + 팀원 모델 RRF (c=60)
cd "$(dirname "$0")/../../src"
P=${PYTHON:-python}
until grep -q '^DONE' ../reports/20261115-team-models/run.log; do sleep 30; done
IN="content_fuse_1_0.5_1_k10_pop7 vbpr3seed_e200 freedom3seed_e100 ease_l500"; W="1 0.7 1.3 1"
BM=bm25_k2000_sh50_pw0.5_p0.4; LG=lgcnC_e575_s42; SX=simplexC_e650_s42
for m in BM LG SX; do for w in 0.3 0.5 1; do
  $P blend.py --inputs $IN ${!m} --weights $W $w --name D039_b6_${m}_w$w || echo "ISSUE D039 $m $w"
done; done
$P blend.py --inputs $IN $BM $LG $SX --weights $W 0.5 0.3 0.3 --name D039_b6_all_0.5_0.3_0.3 || echo "ISSUE D039 all"
echo DONE
