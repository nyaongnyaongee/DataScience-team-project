# 실행 기록: 분포 차이에 강한 피처

- 입력 모델 7개 (rerank_v4와 같음): content_fuse_1_0.5_1_k10_pop7 vbpr3seed_e200 freedom3seed_e100 ease_l500 rp3b_a0.4_b0_k200 ials3seed blend6_e1_r0_i0
- 공통 설정: --lr 0.02 --leaves 31 --min-leaf 500 --objective binary, seed 0
- 1차 (run.log, v5.log): `rerank.py --models <위 7개> ... --robust --apply val --name rerank_robust`
  - user_n → fit 안 유저 활동량 백분위 (user_n_pct)
  - user_span → fit 전체 기간 대비 비율 (user_span_frac)
  - 같은 설정으로 `--apply val final --name rerank_v5 --no-log` 실행 (val 결과 rerank_robust와 바이트 단위 동일)
- 검토에서 user_span_frac의 분모(fit 전체 기간)가 inner·val·final 모두 6174.4일로 같아 효과가 없음이 드러남.
- 2차 (run2.log): user_span도 fit 안 유저 백분위(user_span_pct)로 바꿔
  `rerank.py --models <위 7개> ... --robust --apply val final --name rerank_robust2`
  → submission/submission_v5.csv = results/rerank_robust2_final.csv (1차 v5 파일을 대체)
- 주의: results/rerank_v5_final.csv 는 1차(user_span_frac) 파일이며 제출 파일이 아니다. 제출 파일은 rerank_robust2_final.csv 복사본.
- 2차 검토: reviewer가 코드·숫자·파일 일치를 다시 확인, 문제 없음.
