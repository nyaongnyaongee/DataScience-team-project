# 계획: reranker에 팀원 모델 추가 (D040)
- 질문: v4 reranker 후보·피처에 BM25 / LightGCN / SimpleX를 넣으면 오르나?
- 방법: `rerank.py` v4 설정 그대로(binary, lr .02, leaves 31, min_leaf 500, seed 0, inner 학습, top-50). 모델 목록에 3개 추가(전부 / BM25만).
- 베이스라인: rerank_v4 0.06082, seed 표준편차 0.00026
- 성공 기준: val ≥ 0.06134 (v4 + 2 sd). 넘으면 final 파일 생성(제출은 하지 않음, 사용자 결정).
- 지표: 키트 score, GPU/CPU, 1.5시간.
