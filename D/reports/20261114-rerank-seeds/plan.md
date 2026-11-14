# 계획: reranker seed 앙상블

- 질문: LightGBM reranker를 seed 5개(0~4)로 학습해 예측 점수를 평균하면 단일 seed보다 오르는가? seed 간 흔들림은 얼마인가?
- 방법: rerank_v4 설정(binary, lr 0.02, leaves 31, min_leaf 500, 같은 7개 입력) 그대로 seed만 바꿈. seed는 bagging과 early stopping용 유저 분리에 쓰인다. 예측 점수(확률)를 평균.
- 비교 대상 / 베이스라인: rerank_v4 (seed 0) 0.06082 (hits 8335)
- 지표: Recall@20, 키트 score.
- 성공 기준 (결과 보기 전 고정): 5-seed 평균 ≥ 0.06132 (+0.0005). 단일 seed 5개의 평균·표준편차도 기록(성공 여부와 무관).
- 예상 시간: 1시간 (CPU 위주)
- GPU: 피처 계산에만 잠깐
