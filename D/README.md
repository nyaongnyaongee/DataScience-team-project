# D: 멀티모달·콘텐츠 (+ CF, 블렌드, reranker)

모든 val 점수는 팀 공통 split `f0dfad2f4431a876`에서 키트 `teamkit.py score`로 잰 micro Recall@20(분모 137,041)입니다. test는 실제로 제출한 것만 적었습니다.

- 실험 461줄 전체 기록과 실험 ID 대응: `experiments_D.csv`
- 제출 기록 상세와 팀원용 앙상블 파일: `SUBMISSIONS.md` (Drive 링크 포함)
- 실험별 계획·결과·실행 기록: `reports/*/` (2026-11-14 이후 실험)

## 실험 표

| experiment_id | model_or_combination | parameters | val_recall@20 | test |
|---|---|---|---|---|
| D001 | Popularity | 키트 예제 | 0.02284 | |
| D002 | Content kNN (메타데이터 one-hot) | 카테고리·store·가격·dept, k ∈ {2,5,10,20,50,100,200,500} → 최고 k=10 | 0.02340 | |
| D003 | Content kNN (TF-IDF) | 제목·store·카테고리·features·설명, k ∈ {1…500} × 인기도 보정 β ∈ {0,0.1…100} → 최고 k=10, β=5 | 0.03605 | |
| D004 | Content kNN (MiniLM, 전체 텍스트) | all-MiniLM-L6-v2, k ∈ {2,5,10,20} × β ∈ {0,3,5} → 최고 k=10, β=5 | 0.03849 | |
| D005 | Content kNN (MiniLM, 제목·store·카테고리) | k ∈ {2,5,7,10,15,20} × β ∈ {0,3,5,7,10} → 최고 k=10, β=7 | 0.04054 | |
| D006 | Content kNN (MPNet) | all-mpnet-base-v2, k ∈ {2,5,10,20} × β ∈ {0,3,5} → 최고 k=10, β=5 | 0.03867 | |
| D007 | Content kNN (CLIP 이미지) | ViT-L/14, k ∈ {5,10,20} × β ∈ {0,5,7,10} → 최고 k=10, β=10 | 0.02779 | |
| D008 | Content kNN (CLIP 텍스트) | ViT-L/14, k ∈ {5,10,20} × β ∈ {0,5,7,10} → 최고 k=10, β=7 | 0.04004 | |
| D009 | Content kNN (MiniLM + CLIP 이미지 + CLIP 텍스트 융합) | 가중치 1:0:1 / 1:0.2:0 / 1:0.2:1 / 1:0.5:1 / 1:1:1 × β ∈ {5,7,10}, k=10 → **채택 1:0.5:1, β=7** | 0.04205 | |
| D010 | VBPR 1차 | feature fuse / none(=BPR-MF), d=64, lr ∈ {1e-3,5e-3}, reg 1e-4, ≤100 epoch → 최고 fuse lr 1e-3 e20 | 0.02879 | |
| D011 | VBPR reg·lr·d grid | reg ∈ {1e-3,5e-3,1e-2,2e-2,3e-2,0.1}, lr ∈ {1e-3,3e-3}, d ∈ {64,128}, none reg 1e-2 → 최고 reg 5e-3 e100 | 0.03269 | |
| D012 | VBPR 긴 학습 × seed | fuse, d=64, lr 1e-3, reg 5e-3, epoch 50~400, seed 0·1·2 → 최고 e350 s2, **채택 e200** (s0 0.03362) | 0.03522 | |
| D013 | VBPR 3-seed RRF | seed 0·1·2, epoch ∈ {200,300,400} → **채택 e200** | 0.03794 | |
| D014 | **v1** RRF(콘텐츠 kNN + VBPR s0) | RRF 가중치 0.3~2, minmax 비교 → **채택 1:1, VBPR e200 s0** (최고 0.7 s2 0.04578) | 0.04573 | **0.05081** |
| D015 | **v2** RRF(콘텐츠 kNN + VBPR 3-seed) | VBPR 가중치 0.7~1.3, e200 / e400 → **채택 1:1, e200** | 0.04803 | **0.05268** |
| D016 | FREEDOM dropout × epoch | d=64, LightGCN 2층, item kNN k=10, 이미지 0.1 / 텍스트 0.9, reg_weight 1e-4, lr 1e-3, dropout ∈ {0.5,0.8,0.9}, epoch 25~200, seed 0 → 최고 dropout 0.5 e75 | 0.04708 | |
| D017 | FREEDOM seed·reg·이미지 가중치 | dropout 0.8, seed 1·2, reg_weight 1e-3, 이미지 가중치 0.3, ≤125 epoch → 최고 이미지 0.3 e100 | 0.04734 | |
| D018 | FREEDOM 3-seed RRF | dropout 0.8, e100, seed 0·1·2 | 0.05169 | |
| D019 | RRF(kNN + VBPR3 + FREEDOM s0) | 가중치 6조합 → 최고 1:1:1 | 0.05453 | |
| D020 | **v3** RRF(kNN + VBPR3 + FREEDOM3) | 가중치 6조합 → **채택 1:0.7:1.3** (최고 1:0.7:1.6 0.05609) | 0.05603 | **0.06068** |
| D021 | TimePop (유저 활동 기간 인기도) | bin 30일, 가우시안 σ ∈ {0.5…16}, window pad, γ, bin 7일 → 최고 σ=8 | 0.02430 | |
| D022 | v3 + TimePop | TimePop 가중치 ∈ {0.2,0.5,1} → 모두 하락 | 0.05578 | |
| D023 | EASE | λ ∈ {50,100,200,300,500,700,1000,2000,5000} → 최고 λ=500 | 0.04345 | |
| D024 | RP3β | α ∈ {0.25,0.4,0.5,0.75,1} × β ∈ {0,0.1,0.2,0.4,0.6} × k ∈ {100,200,500} → 최고 α=0.4, β=0, k=200 | 0.04232 | |
| D025 | iALS | d ∈ {64,128,256,512} × α ∈ {1,10,20,40} × reg ∈ {1…3000}, 15 iter → 최고 d=512, α=10, reg=100 (d128 3-seed RRF 0.03971) | 0.04197 | |
| D026 | v3 + EASE / RP3β / iALS3 | 가중치 13조합 → 최고 EASE 1, RP3β 0, iALS 0 (= blend6, reranker 입력) | 0.05744 | |
| D027 | seed 3 → 5개 | VBPR 5-seed 0.03880, FREEDOM 5-seed 0.05263, blend6의 5-seed판 | 0.05778 | |
| D028 | 활동량 그룹별 블렌드 가중치 | kNN ∈ {0.5,1,1.5} × FREEDOM ∈ {0.8,1.3,2}, 그룹 ≤10 / 11–14 / ≥15 | 0.05779 | |
| D029 | blend6 + iALS d512 | iALS 가중치 ∈ {0.25,0.5} → 하락 | 0.05722 | |
| D030 | Reranker (LightGBM LambdaRank) | 후보 = 모델 7개 top-50 합집합, (lr, leaves, min_leaf) ∈ {(0.05,63,100),(0.01,31,500),(0.02,15,2000)} → 최고 마지막 | 0.06011 | |
| D031 | **v4** Reranker (LightGBM binary) | lr 0.02, leaves 31, min_leaf 500, 274 rounds, seed 0, 내부 split(seed 7) 학습. 피처: 모델별 rank·score, 인기도, TimePop, 콘텐츠 cosine, store·카테고리 일치율, 가격, 유저 활동량 | 0.06082 | **0.06705** |
| D032 | Reranker LightGBM seed 5개 평균 | seed 0~4 (단일 평균 0.06059 ± 0.00026) | 0.06093 | |
| D033 | Reranker 개수형 피처 → 백분위 | user_n, user_span → fit 내 백분위 (v5 파일, 미제출) | 0.06064 | |
| D034 | Reranker 입력 보강 | 5-seed 입력, iALS d512, 이력 최대·최근 cosine 피처 | 0.06114 | |
| D035 | 팀원 C LightGCN을 RRF로 추가 | v3 / blend6에 LightGCN 가중치 ∈ {0.3,0.5,1} → 모두 하락 | 0.05679 | |

## 코드

| 파일 | 실험 ID | 내용 |
|---|---|---|
| `D001_Popularity.sh` | D001 | 키트 예제 |
| `D002-D009_ContentKNN.sh` | D002–D009 | `src/run_content.py` |
| `D010-D013_VBPR.sh` | D010–D013 | `src/vbpr.py` |
| `D014-D015_Blend_v1_v2.sh` | D014–D015 | `src/blend.py` |
| `D016-D018_FREEDOM.sh` | D016–D018 | `src/freedom.py` |
| `D019-D020_Blend_v3.sh` | D019–D020 | `src/blend.py` |
| `D021-D022_TimePop.sh` | D021–D022 | `src/timepop.py` |
| `D023-D026_D029_CF.sh` | D023–D026, D029 | `src/cf.py` (EASE, RP3β, iALS) |
| `D027-D028_SeedsGroupWeights.sh`, `D028_group_combo.py` | D027–D028 | seed 추가, 그룹별 가중치 |
| `D030-D034_Reranker.sh` | D030–D034 | `src/rerank.py` |
| `D035_Blend_with_C_LightGCN.sh` | D035 | 팀원 C 파일 블렌드 |

`src/`의 공통 모듈: `common.py`(데이터·추천·저장), `features.py`(MiniLM·CLIP·TF-IDF 특성), `evaluate.py`(키트 score 호출 + 기록), `download_images.py`(item_meta의 이미지 URL 다운로드).

각 `.sh` 파일에는 그 ID들의 val grid 명령과, 채택한 설정의 final·inner 명령이 들어 있습니다. 한 줄씩 골라 실행해도 됩니다.

### 실행 준비

```
D/
├── teamkit/        ← 팀 공통 키트 압축을 그대로 풀기 (validation/*.csv 포함)
├── data/           ← Kaggle 원본: train.csv, test.csv, item_meta.jsonl, (images/)
├── results/        ← 예측 출력 (자동 생성)
└── src/
```

- 환경: Python 3.12, torch 2.6 (CUDA), numpy, pandas, scipy, scikit-learn, sentence-transformers, transformers, pillow, lightgbm 4.7
- 이미지: `cd src && python download_images.py` → `data/images/{item_id}.jpg`
- 실행: `bash D002-D009_ContentKNN.sh` (`PYTHON=...`으로 인터프리터 지정 가능)
- 출력: `results/{이름}_{val|final|inner}_top100.csv.gz` (`user_id,item_id,score,rank`), val은 `experiments.csv`에 자동 기록

### 재현 시 주의
- 결정적 모델(콘텐츠 kNN, EASE, RP3β, TimePop, 블렌드)은 점수가 정확히 같아야 합니다.
- VBPR, FREEDOM, iALS, reranker는 GPU 비결정성 때문에 seed가 같아도 조금씩 다를 수 있습니다.
- 초기 실험(D002–D013)의 일부 명령은 기록된 결과 이름에서 다시 구성한 것입니다. 실행했던 코드도 그 뒤에 일부 고쳐졌습니다(예: 인기도 보정 옵션 추가). 그래서 재실행했을 때 몇몇 grid 점은 값이 조금 다를 수 있습니다. 2026-11-14 이후 실험(D021–D035)은 `reports/*/run.md`에 실제 실행한 명령이 그대로 남아 있습니다.
