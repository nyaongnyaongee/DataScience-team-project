# 실행 기록: reranker

- 코드: src/rerank.py (LightGBM 4.7.0, ds 환경에 설치). seed 0 (LightGBM bagging + early stopping용 유저 10% 분리).
- 내부 split: `cd teamkit && python teamkit.py split --train validation/val_train.csv --out ../data/inner --seed 7`
  → split_id b45fa59b9ee68d8a, fit 368,997 / truth 109,904 / users 36,043. 키트 파일 수정 없음, data/ 는 git 제외.
- inner 기본 모델 (val에서 고른 설정 그대로): inner_base.sh, inner_cf.log
  - run_content.py fuse_1_0.5_1 k10 pop7, vbpr.py seed 0·1·2 e200, freedom.py seed 0·1·2 e100, 각 3-seed RRF
  - cf.py ease λ500, rp3b α0.4 β0 k200, ials d128 α10 reg100 seed 0·1·2 → ials3seed
  - blend6_e1_r0_i0 (= kNN 1 : VBPR3 0.7 : FREEDOM3 1.3 : EASE 1):
    `blend.py --mode inner --inputs content_fuse_1_0.5_1_k10_pop7 vbpr3seed_e200 freedom3seed_e100 ease_l500 rp3b_a0.4_b0_k200 ials3seed --weights 1 0.7 1.3 1 0 0 --name blend6_e1_r0_i0` (출력은 버려서 로그 없음)
- final 기본 모델: final_cf.log (CF 모델과 blend6를 train.csv 전체로). kNN/VBPR/FREEDOM final은 v3 때 만든 것 사용.
- reranker 실행:
  - 1차: `rerank.py --models <7개> --topn 50 --apply val --name rerank_v1` (lambdarank lr0.05 leaves63 minleaf100)
  - 튜닝 (rerank_tune.log): lambdarank lr0.01/l31/m500, lambdarank lr0.02/l15/m2000, binary lr0.02/l31/m500
  - 최종: `rerank.py --models <7개> --lr 0.02 --leaves 31 --min-leaf 500 --objective binary --apply val final --name rerank_v4`
    한 번 학습한 모델을 val 채점과 final 적용에 함께 사용. 모델: results/rerank_v4_lgb.txt
- 모델 7개: content_fuse_1_0.5_1_k10_pop7 vbpr3seed_e200 freedom3seed_e100 ease_l500 rp3b_a0.4_b0_k200 ials3seed blend6_e1_r0_i0
- 제출 파일: submission/submission_v4.csv = results/rerank_v4_final.csv, teamkit check valid true (36043 users, 20 items)
- 코드 변경 시점: rerank.py는 튜닝 뒤 v4 실행 직전에 --apply를 여러 모드로 받도록만 바꿨다 (학습 부분 동일, val 출력이 튜닝 실행과 바이트 단위로 같음).
- cf.py iALS solve를 행 블록 방식으로 바꾼 시점이 inner/val ials3seed 생성 뒤, final 생성 전이다. 수학적으로 같은 계산이며 d128 seed 0은 같은 점수로 재현됐다 (seed 1·2는 재현 확인 안 함). blend6에서 iALS 가중치는 0이라 reranker의 iALS 피처에만 영향.
