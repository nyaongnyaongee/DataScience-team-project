# 계획: reranker 입력 보강

- 질문: 더 강한 기본 모델과 피처를 넣으면 reranker가 오르는가?
- 방법:
  - 입력 교체: VBPR 3→5-seed, FREEDOM 3→5-seed, iALS d128 3-seed → d512 (inner·val·final 모두 학습 필요)
  - 피처 추가: 유저 이력과의 최대 cosine (MiniLM, CLIP 이미지), 유저 최근 5개 구매와의 평균 cosine
  - 그 외 rerank_v4와 동일 (seed 0)
- 비교 대상 / 베이스라인: rerank_v4 0.06082
- 지표: Recall@20, 키트 score.
- 성공 기준 (결과 보기 전 고정): ≥ 0.06182 (+0.001)
- 예상 시간: 3~4시간 (inner seed 3·4, final seed 3·4, iALS d512 inner/final 학습)
- GPU: 사용
