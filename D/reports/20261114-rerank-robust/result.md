# 결과: 분포 차이에 강한 피처 — 채택 (비열등 기준 충족)

| | Recall@20 | hits | best iter |
|---|---|---|---|
| rerank_v4 (seed 0) | 0.06082 | 8335 | 274 |
| rerank_robust (1차: user_n_pct + user_span_frac) | 0.06062 | 8307 | 208 |
| **rerank_robust2 (2차: user_n_pct + user_span_pct)** | **0.06064** | 8310 | 208 |

- 성공 기준 (≥ 0.06032, 비열등): 두 버전 모두 충족. 최종 채택은 2차.
- v4 대비 −0.0002는 reranker seed 간 표준편차(0.00026)보다 작다. 또 v4의 seed 0은 5개 seed 중 최고값이었으므로 비교 기준이 v4 쪽에 유리하게 잡혀 있다 (seed 평균 0.06059 기준이면 차이 없음).
- 1차의 user_span_frac은 세 모드의 분모가 같아 순서만 보존하는 변환이었고, 트리 모델에는 효과가 없었다 (유저 기간 비율 중앙값 inner 0.2335 → val 0.2545 → final 0.2757로 여전히 이동). 1차에서 실제로 바뀐 피처는 user_n_pct 하나였다. 2차에서 백분위로 고쳤다.
- val 평가는 inner→val(유저당 약 10→13건) 이동만 포함한다. final(약 17건)에서의 이득은 LB로만 확인할 수 있다.

## v5 제출 파일
- submission/submission_v5.csv = results/rerank_robust2_final.csv, teamkit check valid true.
- v4와 유저별 top-20 겹침 평균 92.8%.
- seed 0 단일 모델이다. seed 평균은 점수 기준을 넘지 못해(rerank-seeds) 쓰지 않았다.

## 검토
- reviewer 1회: 누출 없음, 숫자 재채점 일치, v5 파일 일치. 지적: user_span_frac 무효 → 2차로 수정, v4 기준의 유리함 명시, 기록 보완.
