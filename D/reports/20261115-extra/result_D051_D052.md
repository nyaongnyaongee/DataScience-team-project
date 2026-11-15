# 결과: split 3개 (D051), 개선 입력 + split 2개 (D052), 수정 코드 재실행 (D048c)

모두 v4 설정(binary, lr .02, leaves 31, min_leaf 500, seed 0). D051·D052·D048c는 early stopping 분리를 고친 코드로 실행.
inner3 = teamkit split seed 9 (split_id 37bca72ea30afb64).

| ID | 모델 | 학습 split | val R@20 (hits) |
|---|---|---|---|
| D040 | team3 | inner | 0.06118 (8384) |
| D048 (수정 전 코드) | team3 | inner+inner2 | 0.06171 (8457) |
| D048c (수정 코드) | team3 | inner+inner2 | 0.06161 (8443) |
| **D051** | team3 | inner+inner2+inner3 | **0.06218** (8521) |
| D052 | team3v2 (BM25 b.25, LightGCN·SimpleX 3-seed) | inner+inner2 | 0.06157 (8438) |

- D051: 사전 기준(D048 0.06171 + 0.00052 = 0.06223) 대비 −0.00005로 **경계에서 미달**. D048c(같은 수정 코드, split 2개)와 비교하면 +0.00057.
- split 수에 따른 경향: 1개 0.06118 → 2개 0.06161 → 3개 0.06218. seed 0 한 번씩이지만 단조 증가.
- D052: 기준(> 0.06223) 실패. D048c와 같은 수준(−5 hits). split 1개에서 본 입력 개선 효과(D050 +0.00035)가 split 2개에서는 나타나지 않았다.
- D048c 0.06161 vs D048 0.06171: 수정 전 조건의 early stopping 낙관 편향이 컸다고 보기는 어렵다 (10 hits 차이, seed 노이즈 이내).
- 다음: D051 설정 5-seed (D053)로 seed 노이즈 대비 확인, split 5개 (D054).
- final: D051·D052 `results/*_final.csv` 생성.
