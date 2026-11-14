# 계획: 유저 활동 기간 기반 시간 인기도 (timepop)

- 질문: split이 랜덤이라도 정답 아이템은 유저가 활동하던 기간에 산 것이다. "유저 활동 기간에 인기 있던 아이템" 점수가 전체 인기도보다 정답을 잘 맞히고, 기존 블렌드에 더하면 점수가 오르는가?
- 방법: 유저 u의 학습 timestamp 범위 [t_min - pad, t_max + pad]에서 아이템별 상호작용 수 (가우시안 커널 가중 버전도 비교). pad, 커널 폭은 소규모 grid.
- 비교 대상 / 베이스라인:
  - 단독: Popularity 0.02284
  - 블렌드: v3 blend3s_k1_v0.7_f1.3 = 0.05603 (hits 7678)
- 지표: Recall@20, 키트 `teamkit.py score` (src/evaluate.py 경유). 활동량 그룹별 recall 함께 기록.
- 성공 기준 (결과 보기 전 고정):
  1. timepop 단독 ≥ 0.0275 (Popularity 대비 +20%)
  2. v3 블렌드에 추가했을 때 ≥ 0.05703 (+0.001, 약 137 hits)
- 데이터 규칙: val 모드는 val_train.csv만 사용. val_truth는 채점에만.
- 예상 시간: 1~2시간
- GPU: 불필요 (CPU)
