# 계획: 팀원 B·C 모델 이식 (D036–D038)

- 질문: 팀원 코드를 그대로 옮겨 내 split 3개(val/final/inner)에서 top-100을 만들 수 있나? val 점수가 팀원 보고와 맞나?
- 코드: BM25는 B 노트북(B051-B154) 셀 3·4 로직을 `src/cf.py bm25`로 GPU 이식. C는 원문을 `src/ext/`에 그대로 두고 `src/team_models.py`가 호출.
- 차이: C 원본 val은 val_truth로 best epoch 가중치를 골랐다. 여기서는 val/final/inner 모두 고정 epoch(LightGCN 575, SimpleX 650), val_truth 미사용.
- 베이스라인: B061 0.042914 (hits 5881), C001 0.036055, C011 0.037113
- 성공 기준: BM25 hits 5881 정확히 일치 / LightGCN val ≥ 0.0350 / SimpleX val ≥ 0.0361
- 지표: 키트 score. GPU 사용. 예상 1시간.
