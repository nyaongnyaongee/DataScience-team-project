# 결과: reranker — 성공 (그 시점 최고 블렌드 + 0.00303)

비교 기준: plan의 "그 시점 최고 RRF 블렌드"는 v4 실행 당시 gw_combo = 0.05779 (hits 7919, 그룹별 가중치 실험)였다. 성공 기준 ≥ 0.05979.
(처음에는 reranker 입력으로 쓴 blend6_e1_r0_i0 = 0.05744를 기준으로 적었으나 검토에서 정정.)

| 설정 | 목적함수 | lr / leaves / min_leaf | best iter | val Recall@20 | hits |
|---|---|---|---|---|---|
| rerank_v1 | lambdarank | 0.05 / 63 / 100 | 15 | 0.05804 | 7954 |
| | lambdarank | 0.01 / 31 / 500 | 31 | 0.05993 | 8213 |
| | lambdarank | 0.02 / 15 / 2000 | 169 | 0.06011 | 8237 |
| **rerank_v4** | **binary** | **0.02 / 31 / 500** | 274 | **0.06082** | **8335** |

- 성공 기준: 충족. 0.06082 ≥ 0.05979 (+0.00303). blend6 대비 +0.00338, v3(0.05603) 대비 +0.00479 / +657 hits.
- 그룹별 (원본 train 건수 기준): ≤13 0.06212, 14–18 0.06127, ≥19 0.05931. 세 그룹 모두 블렌드보다 높음.
- 주의 (낙관 편향): reranker 설정 4개를 같은 val로 비교해 골랐고, "규제를 강하게" 하는 방향도 1차 결과(0.05804)를 본 뒤 정했다. LightGBM seed는 0 하나뿐이라 seed 흔들림은 재지 않았다. 설정 간 차이(최대 0.0028)가 기준 대비 여유(+0.0010)보다 크므로, 보수적 추정치로는 강한 규제 3개 중 최저인 **0.05993**(기준 0.05979를 겨우 넘음)을 함께 본다. LB 결과로 확인 필요.
- 재현: rerank_v4 로 다시 학습했을 때 val 0.06082 / 8335 hits 로 튜닝 실행과 완전히 같음.
- 내부 split 후보 recall (정답 중 후보에 들어간 비율): 0.1635. 이것은 inner에서 후보 집합이 맞힐 수 있는 상한이며, 실제 recall@20(약 0.06)과는 차이가 크다. val 후보 recall은 재지 않았다.
- 중요 피처 (gain): blend6 score·rank, timepop, FREEDOM score, store 일치율, 유저 활동량, rrf, 인기도, 카테고리 leaf 일치율, 시간 차이.
  timepop은 블렌드 재료로는 손해였지만 reranker 피처로는 3번째로 중요했다.
- 한계: reranker는 inner(학습 369k건)에서 학습해 final(616k건)에 적용한다. val 평가도 inner→val(479k건)이라 같은 방향의 분포 차이를 이미 포함하지만, final은 차이가 더 크다. LB로 확인 필요.
  특히 user_n, user_span 같은 정규화하지 않은 개수형 피처는 inner(유저당 평균 약 10건) → val(약 13건) → final(약 17건)로 커져, final에서는 학습 때 보지 못한 범위에 적용된다.
- 계획과 다른 점: plan의 timepop 후보, CLIP 최대 유사도는 구현하지 않았다. timepop은 피처로만, 콘텐츠 유사도는 평균 프로필 cosine만 쓴다.
- experiments.csv의 rerank 행 training_interactions 열은 validation/val_train.csv로 찍히지만, reranker 자체는 inner split으로 학습했다 (extra_inputs 열에 표시).
- 산출물: submission/submission_v4.csv (teamkit check valid true).

## 검토
- reviewer 1회: 정답 누출 없음 (inner 데이터 sha256이 val_train과 일치, inner top100은 별도 생성 파일, early stopping 유저도 inner), 숫자 모두 재채점으로 일치, submission_v4.csv = rerank_v4_final.csv, check valid. 지적 반영: 기준선을 그 시점 최고 0.05779로 정정, 낙관 편향과 보수적 추정치 명시, 후보 recall 표현, 분포 차이 구체화, 계획과 다른 점, 재현성 기록(run.md).
