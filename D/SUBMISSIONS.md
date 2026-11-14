# D(멀티모달·콘텐츠) 제출 기록과 앙상블용 파일

작성 2026-11-14. 모든 로컬 점수는 공통 split `f0dfad2f4431a876`에서 키트 `teamkit.py score`로 잰 micro Recall@20(분모 137,041).

## 1. 제출 요약

| 버전 | 구성 | 로컬 val R@20 (hits) | LB R@20 | 상태 |
|---|---|---|---|---|
| v1 | RRF(콘텐츠 kNN 1 : VBPR seed0 1) | 0.04573 (6267) | 0.05081 | 2026-11-11 제출 |
| v2 | RRF(콘텐츠 kNN 1 : VBPR 3-seed 1) | 0.04803 (6582) | 0.05268 | 2026-11-11 제출 |
| v3 | RRF(콘텐츠 kNN 1 : VBPR 3-seed 0.7 : FREEDOM 3-seed 1.3) | 0.05603 (7678) | 0.06068 | 2026-11-12 제출 |
| v4 | LightGBM reranker (모델 7개 후보 + 피처) | 0.06082 (8335) | 0.06705 | 2026-11-14 제출 |

- 모든 제출은 val에서 설정·가중치를 정한 뒤 원본 `train.csv` 전체로 다시 학습해서 만들었습니다. reranker 자체는 다시 학습하지 않았습니다.
- 로컬 순위와 LB 순위가 같습니다(v1 < v2 < v3 < v4). LB가 로컬보다 v1~v3는 0.0047~0.0051, v4는 0.0062 높습니다.
- RRF 식은 `Σ w_m / (60 + rank_m)`이고 각 모델의 top-100을 씁니다. 리스트에 없는 아이템은 그 모델에서 0점입니다.

## 2. 버전별 상세

### v1 — LB 0.05081
- **콘텐츠 kNN** (`content_fuse_1_0.5_1_k10_pop7`, 단독 0.04205)
  - 아이템 표현: MiniLM(all-MiniLM-L6-v2, 제목·브랜드·카테고리) + 0.5 × CLIP 이미지 + CLIP 텍스트
  - 유저가 산 아이템마다 이웃 k=10개를 모으고, 인기도 보정 β=7을 적용
- **VBPR** (`vbpr_fuse_d64v64_lr0.001_reg0.005_e200_s0`, 단독 0.03362)
  - 아이템 feature는 위와 같은 fuse feature. d=64, 시각 부분 64차원, lr 1e-3, reg 5e-3, 200 epoch, seed 0
- 두 모델을 1:1 RRF로 합침

### v2 — LB 0.05268
- VBPR을 seed 0·1·2로 학습해서 RRF로 합침 (`vbpr3seed_e200`, 단독 0.03794)
- 콘텐츠 kNN과 1:1 RRF. seed를 3개로 늘린 것만으로 로컬 +0.0023

### v3 — LB 0.06068
- **FREEDOM** 추가 (`freedom3seed_e100`, 단독 0.05169로 단일 모델 중 가장 높음)
  - d=64, LightGCN 2층, item-item kNN 그래프 k=10 (이미지 0.1 / 텍스트 0.9)
  - edge dropout 0.8, reg_weight 1e-4, lr 1e-3, 100 epoch, seed 0·1·2 RRF
  - 텍스트 feature = MiniLM + CLIP 텍스트, 이미지 feature = CLIP 이미지
- 가중치는 kNN 1 : VBPR 0.7 : FREEDOM 1.3 (val에서 소규모 grid로 고름)

### v4 — LB 0.06705 (로컬 0.06082)
- **후보**: 아래 7개 모델에서 각각 top-50을 뽑아 합집합 (유저당 약 180개)
  - 콘텐츠 kNN, VBPR 3-seed, FREEDOM 3-seed, EASE λ=500, RP3β α=0.4 β=0 k=200, iALS d=128 3-seed
  - RRF 블렌드 `blend6_e1_r0_i0` (kNN 1 : VBPR 0.7 : FREEDOM 1.3 : EASE 1, 로컬 0.05744)
- **피처**: 모델별 rank·score, 인기도, 유저 활동 시기 인기도(timepop), 시간 차이, 콘텐츠 cosine(MiniLM, CLIP), store·카테고리 일치율, 가격, 유저 활동량
- **모델**: LightGBM binary, lr 0.02, leaves 31, min_leaf 500, 274 rounds
- **학습 데이터**: `val_train.csv`를 키트 split으로 한 번 더 나눈 내부 split(seed 7, split id `b45fa59b9ee68d8a`)에서 학습했습니다. `val_truth`는 쓰지 않았습니다.
- **LB 확인 결과**: 설정을 val로 고른 낙관 편향(보수적 추정 약 0.0599)과 inner → final 피처 범위 차이를 걱정했지만, LB는 v3보다 +0.00637 올랐습니다. 로컬 상승폭(+0.00479)보다도 큽니다.
- 자세한 내용: `reports/20261114-reranker/result.md`, 재생성 방법: `reports/20261114-reranker/run.md`

## 3. 팀원이 앙상블에 쓸 수 있는 파일

**다운로드: [Google Drive 폴더](https://drive.google.com/drive/u/0/folders/1ClCxmI5NouNw_UgxuVwhnY5rh0dHr9vI)** (팀원만 접근 가능)

- 들어 있는 파일: 아래 표에서 O 표시한 23개, 합계 약 712MB
- 파일 이름: `{이름}_{mode}_top100.csv.gz`, 각각 약 25~33MB
- 받은 파일은 `results/`에 넣으면 `src/blend.py`가 바로 읽습니다.
- seed별 파일은 Drive에 올리지 않았습니다. 필요하면 요청해 주세요.

- **형식**: `user_id,item_id,score,rank`. 유저마다 100행이고 rank는 1~100입니다. 학습 때 본 아이템은 이미 빠져 있습니다.
- **mode**
  - `val`: `val_train.csv`로 학습해서 val 유저를 예측한 결과. 앙상블 가중치를 정하고 점수를 잴 때 씁니다.
  - `final`: 원본 `train.csv` 전체로 학습해서 test 유저를 예측한 결과. 제출 파일을 만들 때 씁니다.
  - `inner`: 내부 split(`b45fa59b9ee68d8a`)으로 학습한 결과. reranker를 학습할 때만 씁니다.

### 단일 모델 (앙상블 재료로 추천)

| 이름 | 모델 | 단독 val R@20 | val | final | inner |
|---|---|---|---|---|---|
| `freedom3seed_e100` | FREEDOM 3-seed | 0.05169 | O | O | O |
| `ease_l500` | EASE λ=500 | 0.04345 | O | O | O |
| `rp3b_a0.4_b0_k200` | RP3β | 0.04232 | O | O | O |
| `content_fuse_1_0.5_1_k10_pop7` | 콘텐츠 kNN (MiniLM+CLIP) | 0.04205 | O | O | O |
| `ials3seed` | iALS d128 3-seed | 0.03971 | O | O | O |
| `vbpr3seed_e200` | VBPR 3-seed | 0.03794 | O | O | O |

seed별 파일(`freedom_..._s0/s1/s2`, `vbpr_fuse_..._s0/s1/s2`, `ials_d128_..._s0/s1/s2`)도 val과 final이 다 있지만 Drive에는 올리지 않았습니다.

### 블렌드·reranker 결과

| 이름 | 내용 | val R@20 | val | final | inner |
|---|---|---|---|---|---|
| `rerank_v4` | v4 reranker | 0.06082 | O | O | - |
| `blend6_e1_r0_i0` | kNN 1 : VBPR 0.7 : FREEDOM 1.3 : EASE 1 | 0.05744 | O | O | O |

v1~v3 블렌드 파일(`blend3s_k1_v0.7_f1.3`, `blend_knn_vbpr3seed_e200_w1`, `blend_rrf_knn_vbpr200_w1_s0`)은 위 단일 모델로 다시 만들 수 있어서 Drive에 올리지 않았습니다.

팀원 모델을 붙일 때는 블렌드 결과 하나보다 **단일 모델 파일을 각각** 넣는 게 낫습니다. 그래야 가중치를 모델마다 따로 조정할 수 있습니다. 내 모델끼리 합쳐 둔 결과를 기준선으로 쓰고 싶으면 `blend6_e1_r0_i0`을 쓰면 됩니다.

## 4. 팀원 파일과 합치는 방법

### 팀원이 줘야 하는 것
1. **val top-100**: `val_train.csv`로 학습해서 저장. 형식은 위와 같습니다(`user_id,item_id,score,rank`).
2. **final top-100**: 같은 설정으로 원본 `train.csv` 전체로 학습해서 저장. **이게 없으면 제출 파일을 만들 수 없습니다.**
3. (reranker에 넣을 경우) **inner top-100**: 아래 명령으로 만든 내부 split의 `val_train.csv`로 학습해서 저장
   ```bash
   cd teamkit && python teamkit.py split --train validation/val_train.csv --out ../data/inner --seed 7
   # split_id b45fa59b9ee68d8a, fit 368,997 / truth 109,904 가 나와야 같은 split
   ```

받은 val 파일은 먼저 top-20만 잘라서 키트 `score`로 채점합니다. 팀원이 보고한 점수와 같아야 같은 split에서 만든 파일입니다.

### 합치는 명령
`src/blend.py`는 RRF(rank만 사용)와 minmax(score 사용)를 지원합니다. 파일을 `results/`에 같은 이름 규칙으로 넣고 실행합니다.

```bash
# val에서 가중치 확인
python src/blend.py --mode val --inputs content_fuse_1_0.5_1_k10_pop7 vbpr3seed_e200 freedom3seed_e100 ease_l500 <팀원모델> \
    --weights 1 0.7 1.3 1 0.5 --method rrf --name blend_team_test
# 같은 이름·가중치로 final 생성 → teamkit check
python src/blend.py --mode final ... --name blend_team_test
```

### top-20만 있을 때 (2026-11-14 시뮬레이션)
내 모델 결과를 top-20으로 잘라서 v3 블렌드에 붙여 비교했습니다. EASE를 팀원 모델 대신으로 썼습니다.

| 경우 | val R@20 | v3 대비 |
|---|---|---|
| v3 + EASE top-100 (가중치 1) | 0.05744 | +0.00141 |
| v3 + EASE top-20 (가중치 0.5) | 0.05718 | +0.00115 |
| FREEDOM만 top-20으로 자름 | 0.05441 | −0.00162 |
| 3개 모두 top-20으로 자름 | 0.05250 | −0.00353 |

- 보조 모델 하나를 붙이는 경우에는 top-20으로도 top-100 이득의 약 75~80%가 나옵니다. 앙상블에 도움이 되는지 1차로 걸러내는 데는 쓸 만합니다.
- 점수가 높고 가중치를 크게 줄 모델은 top-100이 있어야 합니다.
- top-20 CSV에는 점수가 없으므로 RRF만 됩니다. minmax 가중합과 reranker 피처로는 쓸 수 없습니다.
