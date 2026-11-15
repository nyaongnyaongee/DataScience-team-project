# 결과: C SGL·NCL 이식 (D049) — 실패 (모델 이식은 성공, reranker 이득 없음)

plan 기준은 "val ≥ 0.0348 그리고 reranker 이득". reranker 결과(D049_rerank_team5 0.06110, team3 대비 −0.00008)는 result_D047_D049_D050.md.

C 원문 sgl_ncl_runner.py를 그대로 실행(고정 epoch, val_truth 없음), 저장된 가중치로 top-100 생성.

| 모델 | 설정 | val R@20 (hits) | C 보고 | 기준 |
|---|---|---|---|---|
| SGL | C005: 3층 d128 lr .003 SSL .0005 drop .1, 525 epoch | 0.03583 (4910) | 0.035821 | ≥ 0.0348 성공 |
| NCL | C007: 구조 SSL .001, prototype 0, 425 epoch | 0.03551 (4867) | 0.035515 | ≥ 0.0348 성공 |

- val/final/inner/inner2 파일 생성. reranker 추가 효과는 run_queue.sh의 D049 rerank에서 확인.
- 선택 편향: epoch은 C가 val로 고른 값.
