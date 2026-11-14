# 계획: LightGBM LambdaRank reranker

- 질문: 여러 모델의 후보를 피처 기반으로 다시 순위 매기면 RRF 블렌드보다 오르는가?
- 데이터 규칙 (중요):
  - val_truth는 학습에 절대 쓰지 않는다.
  - 내부 split: `teamkit.py split --train teamkit/validation/val_train.csv --out data/inner/` (키트 수정 없음, 같은 해시 방식). inner_train으로 기본 모델 학습 → inner_truth를 라벨로 reranker 학습.
  - 평가: val_train으로 학습한 기본 모델의 후보·피처 → 학습된 reranker 적용 → 키트 score.
  - 최종: train.csv 전체로 학습한 기본 모델 후보·피처 → 같은 reranker 적용. reranker 재학습 없음.
- 후보: 각 모델 top-100 합집합 (kNN, VBPR, FREEDOM, timepop, EASE/RP3β/iALS 중 성공한 것).
- 피처: 모델별 rank·score(유저별 정규화), 인기도, 기간 인기도, 유저 이력과의 콘텐츠 유사도(MiniLM, CLIP 평균/최대), 유저 활동량, 아이템 store·카테고리 일치율, 가격, 아이템 timestamp 범위와 유저 기간 겹침.
- 비교 대상 / 베이스라인: 그 시점 최고 RRF 블렌드 (cf-models 결과, 최소 v3 0.05603)
- 지표: Recall@20, 키트 score.
- 성공 기준 (결과 보기 전 고정): reranker ≥ 최고 RRF 블렌드 + 0.002
- 필요: ds 환경에 lightgbm 설치 (base 아님)
- 예상 시간: 6~10시간 (inner split용 기본 모델 재학습 포함, VBPR·FREEDOM 3seed씩)
- GPU: 사용 (VBPR, FREEDOM, iALS 재학습)
