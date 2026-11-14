# 실행 기록: reranker seed 앙상블

- 명령 (src/): `rerank.py --models content_fuse_1_0.5_1_k10_pop7 vbpr3seed_e200 freedom3seed_e100 ease_l500 rp3b_a0.4_b0_k200 ials3seed blend6_e1_r0_i0 --lr 0.02 --leaves 31 --min-leaf 500 --objective binary --seeds 0 1 2 3 4 --apply val --name rerank_seeds5`
- seed는 LightGBM bagging/feature_fraction과 early stopping용 inner 유저 10% 분리에 쓰인다. 피처와 후보는 seed와 무관.
- 단일 seed 결과는 rerank_seeds5_s{0..4}, 예측 확률 평균은 rerank_seeds5 로 기록.
- 로그: run.log
