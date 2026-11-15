# 야간 실험 묶음 계획 (2026-11-15 03:00, 사용자 취침 중 자동 실행)

사용자 지시: "돌리고, 생각보다 빨리 끝나면 나머지도 다 실행. 이슈는 중단하지 말고 따로 기록하고 계속."
→ 승인은 이 지시로 갈음. 이슈는 `issues.md`에 기록.

공통: split f0dfad2f4431a876, 지표 micro Recall@20, 키트 `teamkit.py score`(src/evaluate.py)로만 채점.
val_truth는 채점에만 사용. 학습·inner 생성·reranker 학습에 쓰지 않음.
GPU: RTX 4090 1장 (CUDA_VISIBLE_DEVICES=0). tmux 없음 → nohup 백그라운드 + run.log.

| ID | 실험 | 폴더 | 베이스라인 | 성공 기준 (결과 보기 전 고정) | 예상 |
|---|---|---|---|---|---|
| D036 | B BM25-Shrink 이식 (val/final/inner) | team-models | B061 0.042914 | val hits가 B061(5881)과 같음 | 10분 |
| D037 | C LightGCN 이식, 고정 575 epoch (val/final/inner) | team-models | C001 0.036055 (val_truth로 best epoch 고른 값) | val ≥ 0.0350 (C001 −0.001 이내) | 30분 |
| D038 | C SimpleX(C011) 이식, 고정 650 epoch (val/final/inner) | team-models | C011 0.037113 | val ≥ 0.0361 | 30분 |
| D039 | blend6 + BM25/LightGCN/SimpleX RRF 가중치 grid | team-blend | blend6 0.05744 | 최고 조합 > 0.05744 + 0.0005 | 30분 |
| D040 | reranker v4 설정 + 후보 모델 3개 추가 | team-rerank | v4 0.06082 | val ≥ 0.06082 + 0.00052 (seed sd 0.00026 × 2) | 1.5시간 |
| D041 | 블렌드 방식: minmax×r + RRF×(1−r), RRF c | blend-mix | blend6 0.05744 | > 0.05744 + 0.0005 | 1시간 |
| D042~ | 남은 것 (extra): BM25 grid, LightGCN·SimpleX 3-seed, reranker top-100 후보, reranker 5-seed, iALS d1024, CatBoost, inner split 2개 | extra | 각 항목 plan.md | 각 항목 plan.md | 남는 시간 |
