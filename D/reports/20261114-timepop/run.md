# 실행 기록: timepop

- 코드: src/timepop.py (결정적, seed 없음). CPU/GPU 상관없이 같은 결과.
- 데이터: val 모드 = teamkit/validation/val_train.csv. 채점: src/evaluate.py → teamkit.py score.
- 명령:
  - `python src/timepop.py --sigma {0.5,1,2,4,8,16}` (bin 30일, 가우시안)
  - `python src/timepop.py --kernel window --pad {0,3,12}`
  - `python src/timepop.py --sigma 4 --gamma {0.3,0.6}`, `--sigma 4 --bin-days 7`
  - 블렌드: `python src/blend.py --inputs content_fuse_1_0.5_1_k10_pop7 vbpr3seed_e200 freedom3seed_e100 timepop_b30_g8_gm0_pb0 --weights 1 0.7 1.3 {0.2,0.5,1} --name blend3s_tp{w}`
