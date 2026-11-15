# 결과: 팀원 B·C 모델 이식 (D036–D038) — 성공

검토: reviewer 1회. 누출 없음, BM25 수식 B와 같음, 숫자 재채점 일치. 지적(선택 편향 표기, epoch 설명 오류) 반영.

| ID | 모델 | val R@20 (hits) | 팀원 보고 | 기준 | 판정 |
|---|---|---|---|---|---|
| D036 | BM25-Shrink (B061 설정: item IDF, k1 1.2, b .75, shrink 50, power .5, 이웃 2000, pop .4) | 0.04291 (5881) | B061 0.042914 (5881) | hits 같음 | 성공 |
| D037 | C LightGCN 고정 575 epoch, seed 42 | 0.03605 (4941) | C001 0.036055 | ≥ 0.0350 | 성공 |
| D038 | C SimpleX(C011) 고정 650 epoch, seed 42 | 0.03711 (5086) | C011 0.037113 | ≥ 0.0361 | 성공 |

- 세 모델 모두 팀원 val 점수와 같다 (BM25는 B 노트북 출력 hits 5881과 직접 일치. C는 보고된 recall만 있어 hits는 역산 비교).
- C는 SimpleX를 700 epoch까지 돌리며 val로 650을 골랐다. 같은 seed면 epoch E까지 학습 궤적이 같고 평가 루프가 난수를 쓰지 않으므로, E에서 멈춘 결과가 best 시점 가중치와 같다.
- **선택 편향**: 학습에는 val_truth를 쓰지 않았지만 epoch 575·650과 BM25 pop 0.4는 팀원이 val_truth로 고른 값이다. 그래서 val 점수는 약간 낙관적이다. experiments.csv notes의 "val_truth 미사용"은 "학습에 미사용, epoch는 val에서 선택됨"이라는 뜻이다.
- 출력: `results/{bm25_k2000_sh50_pw0.5_p0.4, lgcnC_e575_s42, simplexC_e650_s42}_{val,final,inner}_top100.csv.gz`
- final top-20 파일 3개 `teamkit.py check --pred results/X_final.csv --train data/train.csv --test data/test.csv` valid (03:17 수동 실행, reviewer 재확인).
- 시간: BM25 모드당 약 20초, LightGCN 약 60초, SimpleX 약 70초 (RTX 4090). 예상(1시간)보다 훨씬 빠름.
- 코드: BM25는 `src/cf.py bm25` (B 노트북 셀 3·4 로직을 GPU 행렬 연산으로 옮김. 동점 처리 순서만 다를 수 있음). C 모델은 `src/ext/` 원문 + `src/team_models.py`.
- C의 val 파일(C_lightgcn_val)과 내 lgcnC_e575_s42_val은 같은 점수. 이제 inner도 있어 reranker에 넣을 수 있다.
