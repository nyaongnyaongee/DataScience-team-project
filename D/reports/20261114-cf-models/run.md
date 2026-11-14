# 실행 기록: CF 모델

- 코드: src/cf.py (EASE, RP3β 결정적 / iALS seed 지정). 블렌드: src/blend.py (RRF c=60).
- 데이터: val 모드 = teamkit/validation/val_train.csv. 채점: src/evaluate.py → teamkit.py score.
- GPU 0. tmux가 서버에 없어 bash 백그라운드로 실행 (queue.sh).
- 명령:
  - EASE: `python src/cf.py ease --lam {50,100,200,300,500,700,1000,2000,5000}` (ease_grid.sh, queue.sh)
  - RP3β: `python src/cf.py rp3b --alpha {0.5,0.75,1} --beta {0,0.2,0.4,0.6} --k 200`, 확장 `--alpha {0.25,0.4} --beta 0`, `--k {100,500}`, `--beta 0.1` (ext.log)
  - iALS: `python src/cf.py ials --d {64,128} --alpha {1,10,40} --reg {1,10,100} --iters 15 --seed 0`,
    확장 `--d 128 --alpha 10 --reg {150,200,300,1000,3000}`, `--alpha 20 --reg 300`, d=256은 처음에 GPU 메모리 부족(OOM)으로 실패했고, solve를 행 블록 방식으로 고친 뒤 다시 실행했다.
    seed 1, 2 추가 후 `blend.py --inputs ials_..._s0 _s1 _s2 --name ials3seed`
  - 블렌드: `blend.py --inputs content_fuse_1_0.5_1_k10_pop7 vbpr3seed_e200 freedom3seed_e100 ease_l500 rp3b_a0.4_b0_k200 ials3seed --weights 1 0.7 1.3 e r i` (blend.log)
  - iALS 추가 (ials_d256.log): 행 블록 solve 검증 `ials --d 128 --alpha 10 --reg 100 --seed 0 --no-log`, `--d 256 --reg {50,100,150}`, `--d 512 --reg {100,150}`.
    로그의 OOM traceback 2개는 외적 버퍼 크기 제한을 넣기 전 d=512 첫 시도이며, 고친 뒤 다시 실행한 결과가 아래 행이다.
  - `blend.py --inputs content_fuse_1_0.5_1_k10_pop7 vbpr3seed_e200 freedom3seed_e100 ease_l500 ials_d512_a10_r100_it15_s0 --weights 1 0.7 1.3 1 {0.25,0.5} --name blend5_ials512_{w}`
