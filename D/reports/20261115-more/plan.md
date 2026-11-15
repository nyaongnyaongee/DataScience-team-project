# 추가 실험 D055–D057 (2026-11-15 12:50 작성, 결과 보기 전)

질문: D053(team3, 내부 split 3개, 5-seed, val 0.06206 / 단일 평균 0.06183 ± 0.00028)을 더 올릴 수 있는가?
지표: val Recall@20 (evaluate.py → teamkit score). GPU 0번 1장, 순서대로 실행.
공통: BASE 7 + T3 3 모델, OPT = --lr 0.02 --leaves 31 --min-leaf 500 --objective binary (D053과 같음, 바꾸는 것만 적음).

성공 기준(공통): 단일 seed 점수 또는 단일 seed 평균 > D053 단일 평균 + 2×sd = 0.06183 + 0.00056 = **0.06239**.
제출 규칙: 기준을 넘고 5-seed 결과가 있는 설정만 제출 후보로 바꾼다. 넘는 것이 없으면 D053을 후보로 둔다.

| ID | 바꾸는 것 | 예상 시간 |
|---|---|---|
| D057 | split 3개, reranker 용량 키움: --leaves 63 --min-leaf 1000 (seed 0) | ~8분 |
| D055 | split 5개(D054) 5-seed (seed 0–4) | ~30분 |
| D056 | split 8개: inner6/7/8 추가 (teamkit split --seed 12/13/14; split_id 2a741bca91e25c77, 942b9551f6402225, 94020376f1d805f5), seed 0. 기준 넘으면 5-seed | ~50분 (+25분) |

inner6–8 입력 모델은 D054와 같은 명령으로 만든다. val_truth는 채점에만 쓴다(내부 split 정답은 inner*/val_truth.csv).
