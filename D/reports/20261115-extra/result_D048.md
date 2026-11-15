# 결과: reranker를 내부 split 2개로 학습 (D048) — 기준 경계, 확인 안 됨

검토: reviewer 1회 반영. inner2 = `teamkit.py split --train validation/val_train.csv --out ../data/inner2 --seed 8` (split_id 3072f6272361b6b9). 입력 모델 전부 inner2에서 다시 생성(run_D048.sh, 약 15분).

| 변형 | 비교 대상 (같은 모델, split 1개) | split 2개 | 차이 | 기준 (+0.00052 = 71.3 hits) |
|---|---|---|---|---|
| team3 | D040 0.06118 (8384) | 0.06171 (8457) | +73 hits (+0.00053) | 형식상 통과, 여유 1.7 hits |
| v4 모델 7개 | rerank_v4 0.06082 (8335) | 0.06123 (8391) | +56 hits (+0.00041) | 미달 |

- 두 변형 모두 같은 방향(+0.0004~0.0005)이지만 하나만 기준을 넘었고 seed 0 한 번이다.
- **조건 문제 (reviewer 지적)**: 이 실행에서는 early stopping 검증 유저를 split별로 옮긴 id에서 골라, 같은 실제 유저가 inner에선 학습, inner2에선 검증에 들어갈 수 있었다 (inner·inner2 정답 쌍 23% 겹침). best iter가 늘어난 것(322→377, 274→399)과 맞는다. val 누출은 아니지만 split 1개 실험과 조건이 다르다. 05:00에 rerank.py를 고쳤고, 같은 설정 재실행 D048c와 5-seed 확인 D053을 대기열에 넣었다.
- 라벨은 각 split 자기 정답(data/innerN/val_truth.csv, val_train의 부분집합)에서만 온다. 공통 val_truth 누출 없음 (reviewer 확인).
- final: `results/D048_rerank_team3_2split_final.csv` (check valid).
