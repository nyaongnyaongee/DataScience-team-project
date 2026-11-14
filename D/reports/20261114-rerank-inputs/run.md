# 실행 기록: reranker 입력 보강

- GPU 학습 (gpu.sh, gpu.log): inner·final 모드에서 VBPR seed 3·4 (e200), FREEDOM seed 3·4 (e100), iALS d512 α10 reg100 seed 0.
  이어서 vbpr5seed_e200, freedom5seed_e100 (inner·final). val 모드 결과는 small-tweaks·cf-models 실험에서 이미 있음.
- 블렌드 입력 (blend.log): `blend.py --mode {inner,final} --inputs content_fuse_1_0.5_1_k10_pop7 vbpr5seed_e200 freedom5seed_e100 ease_l500 --weights 1 0.7 1.3 1 --name blend4_5seed`
- reranker (src/): `rerank.py --models content_fuse_1_0.5_1_k10_pop7 vbpr5seed_e200 freedom5seed_e100 ease_l500 rp3b_a0.4_b0_k200 ials_d512_a10_r100_it15_s0 blend4_5seed --lr 0.02 --leaves 31 --min-leaf 500 --objective binary --extra --apply val --name rerank_inputs`
  `--extra`: 유저 최근 64개 이력과의 최대 cosine, 최근 5개 평균 cosine (MiniLM, CLIP 이미지)
- 로그: run.log
