# 결과: reranker 입력 보강 — 실패

| | Recall@20 | hits |
|---|---|---|
| rerank_v4 (seed 0) | 0.06082 | 8335 |
| **rerank_inputs (seed 0)** | **0.06114** | 8379 |

- 성공 기준 (≥ 0.06182): 실패. +0.00032.
- +0.00032는 reranker seed 간 표준편차(0.00026)와 비슷한 크기라 실제 개선인지 구분하기 어렵다.
- 새 입력은 쓰이긴 한다: 중요도 상위에 blend4_5seed, freedom5seed, maxcos 피처가 올라왔다. 입력 교체와 --extra 피처를 한 번에 바꿔 ablation이 없으므로, 어느 쪽이 +0.00032를 냈는지는 알 수 없다.
- plan과 다른 점: blend6_e1_r0_i0도 5-seed 버전 blend4_5seed로 교체했다. "최대 cosine"은 전체 이력이 아니라 최근 64개 이력 기준이다.
- 후보 recall (inner): 0.1632 (v4 0.1635와 거의 같음). 후보를 바꿔도 상한이 늘지 않았다.

## 검토
- reviewer 1회: 누출 없음 (hist_sims 정렬·패딩·마스킹 정확, recent-5는 timestamp 내림차순), 숫자 일치, 실패 판정 맞음. 지적 반영: 근거 없는 해석 삭제, plan과 다른 점 기록.
