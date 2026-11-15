# 결과: LightGCN·SimpleX 3-seed (D043) — 성공

기준: 3-seed RRF > 단일(seed 42) + 0.001.

| 모델 | seed 42 | seed 0 | seed 1 | 3-seed RRF 1:1:1 | 단일 대비 |
|---|---|---|---|---|---|
| LightGCN (575 epoch) | 0.03605 | 0.03560 | 0.03596 | **0.03830** | +0.00225 |
| SimpleX (650 epoch) | 0.03711 | 0.03740 | 0.03639 | **0.03971** | +0.00260 |

- 둘 다 기준 충족. seed 간 차이는 단일 모델 기준 최대 0.001 (SimpleX).
- val/final/inner 3-seed 파일 생성 (`lgcnC3seed`, `simplexC3seed`). inner2용은 D052에서 생성.
- 쓰임: D050·D052 reranker 입력.
