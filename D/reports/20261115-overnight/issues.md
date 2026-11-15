# 야간 실행 중 이슈 기록

(실험은 멈추지 않고 계속 진행. 아침에 확인할 것)

시각은 어림값이라 03:30 이후 항목은 실제보다 늦게 적힌 것이 있다(순서는 맞음).
- D051_D052.log의 ISSUE(run_content OOM)는 중단한 옛 스크립트 로그다. 실제 D051·D052는 queue.log(run_queue.sh)에서 돌았고 ISSUE 0건.

- 03:00 tmux가 설치되어 있지 않음 → nohup 백그라운드 실행 + run.log로 대체.
- 03:00 팀원 C의 val 결과(C001, C011)는 val_truth로 best epoch 가중치를 고른 값이다. 내 이식본은 val_truth 없이 고정 epoch로 학습하므로 점수가 조금 낮을 수 있다. 문제라기보다 참고 사항.
- 03:00 A 폴더에 코드가 없어 A 실험(A047 등)은 이식하지 못함.
- 03:20 D042 grid 스크립트의 b 변형이 이름에 b를 넣지 않아 기본 BM25 val 파일을 덮어쓸 뻔함 → 실행 전 중단, cf.py bm25 이름에 b·k1(기본값 아닐 때) 추가 후 재시작. k500은 이미 끝나 기록됨. 덮어쓰기는 일어나지 않음(k500 이후 단계 전 중단).
- 03:25 D039·D040 대기 스크립트가 죽은 것으로 잘못 판단함 (내 ps 확인 패턴이 run.sh를 빠뜨림). setsid로 다시 띄워 잠깐 중복 실행됨 → 원래 프로세스(시작 13초 경과)를 kill하고 새로 띄운 것만 남김. 결과 파일은 새 프로세스가 처음부터 다시 씀.
- 03:25 D042 재시작 명령은 실제로 실행되지 않았음(도구 타임아웃으로 대기 루프가 정리됨) → setsid로 재실행.
- 03:30 reviewer 지적: src/evaluate.py가 그룹별 recall을 위해 val_truth를 직접 읽어 teamkit.recall_at_20을 부른다 (기존 코드, 주 점수는 키트 score). CLAUDE.md 규칙 3("직접 Recall 계산 금지")에 글자 그대로는 걸릴 수 있음 → 아침에 판단 필요. 이번에 고치지는 않음.
- 03:30 이번 야간 코드(src/cf.py, blend.py, rerank.py, common.py, team_models.py, src/ext/)는 아직 커밋 안 됨. 커밋은 사용자 확인 후.
- 04:10 D048 두 번째 변형(D048_rerank_v4models_2split) 피처 생성 중 CUDA OOM. 원인: 같은 GPU에서 D047 CatBoost(17GB) + iALS d1024(4.7GB) + SGL이 동시에 돌았음. 첫 변형(team3_2split)은 정상 완료. → CatBoost 끝난 뒤 재실행 (run_D048b.sh).
- 04:10 D046 iALS d1024는 reg 100에서만 높고(0.04240) 50·130에서는 크게 떨어짐 → reg에 매우 민감. reg 200 추가 확인 중.
- 04:20 D047 CatBoost: 학습은 끝났으나 get_feature_importance()가 랭킹 손실에서 학습 데이터를 요구해 예외 → val 예측 전에 종료. rerank.py에서 중요도 계산을 try로 감싸고 D047만 재실행 (run_D047b.sh).
- 04:25 GPU OOM 두 번째: D045·D048b·D049 rerank·D051 base가 동시에 GPU를 씀 (rerank 피처 생성이 프로세스당 약 7.7GB). D051 run_content 실패. → 대기 중이던 D049(rerank 단계)·D050·D047b·D051/D052 스크립트를 멈추고 run_queue.sh 하나로 순차 실행하도록 재구성. D049 rerank는 시작 10초 만에 중단했으므로 결과 없음, 큐에서 다시 실행. D051 inner3 실패분(content)도 큐에서 처음부터 다시 생성.
- 04:25 D046 iALS d1024 reg 200: 0.02935. reg 100에서만 높음.
- 05:00 reviewer 2차: split 여러 개 학습 시 early stopping 검증 유저가 실제 유저 기준으로 분리되지 않음 → rerank.py 수정 (split 1개는 동작 동일). D048 결과는 수정 전 조건. D048c 재실행 + D053 5-seed 대기열 추가.
- 05:00 reviewer 2차: 팀원 모델 epoch(575/650/525/425)·BM25 pop .4는 팀원이 공통 val_truth로 고른 하이퍼파라미터. 학습 코드는 val_truth를 읽지 않지만 CLAUDE.md 규칙 1 취지에 간접적으로 걸릴 수 있음 → 아침에 판단 필요 (대안: inner split에서 epoch을 다시 고르기).
