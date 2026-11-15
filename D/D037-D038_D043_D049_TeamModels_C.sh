#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/src"
P=${PYTHON:-python}
# 팀원 C 코드 원문(src/ext, C 폴더에서 복사)을 고정 epoch으로 실행해 top-100 생성. val_truth 미사용.
# D037 LightGCN (575 epoch), D038 SimpleX (C011 X01, 650 epoch)
for m in val final inner; do $P team_models.py lightgcn --mode $m; $P team_models.py simplex --mode $m; done
# D043 seed 0·1 추가 → seed 42·0·1 RRF 3-seed
for s in 0 1; do for m in val final inner inner2; do
  $P team_models.py lightgcn --mode $m --seed $s; $P team_models.py simplex --mode $m --seed $s
done; done
for m in val final inner inner2; do
  $P blend.py --mode $m --inputs lgcnC_e575_s42 lgcnC_e575_s0 lgcnC_e575_s1 --name lgcnC3seed
  $P blend.py --mode $m --inputs simplexC_e650_s42 simplexC_e650_s0 simplexC_e650_s1 --name simplexC3seed
done
# D049 SGL (525 epoch), NCL (425 epoch)
for m in val final inner inner2; do $P team_models.py sgl --mode $m; $P team_models.py ncl --mode $m; done
