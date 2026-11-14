# 계획: CF 모델(EASE, RP3β, iALS) 직접 구현 + 블렌드 → v4

- 질문: 순수 CF 모델을 콘텐츠 블렌드에 더하면 점수가 얼마나 오르는가?
- 방법:
  - EASE: λ ∈ {50,100,200,500,1000,2000,5000}, coarse→fine. GPU float32 역행렬.
  - RP3β: α ∈ {0.5,0.75,1.0} × β ∈ {0,0.2,0.4,0.6}, top-k 이웃 ∈ {100,200,500}.
  - iALS: torch로 직접 구현 (implicit 패키지 없음). factors ∈ {64,128}, reg, α grid 소규모, seed 3회로 노이즈 측정.
  - 튜닝 완료 기준: 최고점이 grid 가장자리면 grid 확장.
  - 블렌드: RRF로 기존 3모델 + CF 모델, 가중치 소규모 grid.
- 비교 대상 / 베이스라인: v3 블렌드 0.05603. 단독은 Popularity 0.02284, FREEDOM 3seed 0.05169.
- 지표: Recall@20, 키트 score. 그룹별 recall 기록.
- 성공 기준 (결과 보기 전 고정):
  1. 새 블렌드 ≥ 0.05803 (v3 대비 +0.002)
  2. 충족 시 원본 train.csv 전체로 재학습 → teamkit check 통과 → submission_v4.csv 생성 (제출은 사용자가)
- 예상 시간: 3~5시간 (final 재학습 포함)
- GPU: 사용 (EASE 역행렬, iALS). GPU 0 (24GB, 비어 있음)
