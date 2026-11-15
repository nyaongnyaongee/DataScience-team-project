# 결과: split 5개 (D054) — 실패 (추가 이득 확인 안 됨)

검토: reviewer 1회.

inner4 = teamkit split seed 10 (ff709aa94dc99ea7), inner5 = seed 11 (2d1200ceb4f41eee). team3 모델, v4 설정, seed 0, 수정된 early stopping 분리.

| split 수 | 1 (D040) | 2 (D048c) | 3 (D051) | 5 (D054) |
|---|---|---|---|---|
| val R@20 | 0.06118 | 0.06161 | 0.06218 | 0.06216 (8518) |
| best iter | 322 | 279 | 358 | 458 |

- 기준(> D051 + 0.00052) 실패. 3개 → 5개에서 변화 없음 (−3 hits).
- seed 0 한 번 비교이고, 비교 대상 D051 seed 0은 D053 5개 seed 중 최고값이다 (단일 sd 약 38 hits). D054 0.06216은 D053 단일 평균 0.06183보다는 높다. "포화"라고 말할 근거는 없고, split 5개의 추가 이득이 확인되지 않았을 뿐이다. 확인하려면 5-seed가 필요하다.
- final 파일 생성·check valid. 제출 후보로는 D053(3 split 5-seed)을 우선한다.
