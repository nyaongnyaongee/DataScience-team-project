# 실행 기록: 소규모 개선

- seed 추가: seeds.sh → `vbpr.py --features fuse --reg 5e-3 --epochs 200 --eval-every 200 --seed {3,4}`, `freedom.py --epochs 100 --eval-every 100 --seed {3,4}` (GPU 0)
- 5-seed 블렌드: blend5.log → vbpr5seed_e200, freedom5seed_e100, blend4_5seed (= kNN 1 : VBPR5 0.7 : FREEDOM5 1.3 : EASE 1)
- 그룹별 가중치 (조합 코드: group_combo.py): group.log (kNN ∈ {0.5,1,1.5} × FREEDOM ∈ {0.8,1.3,2}, VBPR 0.7, EASE 1 고정), group_combo.log
  - 그룹 경계: val_train 건수 ≤10 / 11–14 / ≥15 (원본 train 건수에는 정답 개수가 섞여 있어 쓰지 않음)
  - 그룹별 가중치는 evaluate.py가 출력하는 원본 기준 그룹 recall로 고름
