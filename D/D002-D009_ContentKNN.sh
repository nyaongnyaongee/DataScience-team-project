#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/src"
P=${PYTHON:-python}
K=content_fuse_1_0.5_1_k10_pop7
V=vbpr_fuse_d64v64_lr0.001_reg0.005
FR=freedom_d64_L2_k10_wi0.1_do0.8_rw0.0001_wd0_lr0.001
# 콘텐츠 item-kNN: 아이템 특성 cosine → 이웃 k개 → 유저 점수 합, --pop-beta 는 인기도 보정 (1e-6 = 보정 없음)
# D002 메타데이터 one-hot (카테고리·store·가격·dept)
$P run_content.py --features meta --k 2 5 10 20 50 100 200 500
# D003 TF-IDF (title·store·categories·features·description)
$P run_content.py --features tfidf --k 2 5 10 20 50 100 200 500
$P run_content.py --features tfidf --k 1 2 5 20 --pop-beta 0.1 0.3 1 3
$P run_content.py --features tfidf --k 5 10 20 50 --pop-beta 10 30 100
$P run_content.py --features tfidf --k 5 10 --pop-beta 2 3 5
# D004 MiniLM (전체 텍스트)
$P run_content.py --features minilm --k 2 5 10 20 --pop-beta 1e-6 3 5
# D005 MiniLM (title·store·categories)
$P run_content.py --features minilm_title --k 2 5 10 20 --pop-beta 1e-6 3 5
$P run_content.py --features minilm_title --k 7 10 15 --pop-beta 5 7 10
# D006 MPNet
$P run_content.py --features mpnet --k 2 5 10 20 --pop-beta 1e-6 3 5
# D007 CLIP 이미지 (ViT-L/14, data/images 필요: python download_images.py)
$P run_content.py --features clip_img --k 5 10 20 --pop-beta 1e-6 5 7 10
# D008 CLIP 텍스트
$P run_content.py --features clip_txt --k 5 10 20 --pop-beta 1e-6 5 7 10
# D009 융합 fuse_<MiniLM>_<CLIP이미지>_<CLIP텍스트> 가중치  → 채택: fuse_1_0.5_1 k10 pop7
for f in fuse_1_0_1 fuse_1_0.2_0 fuse_1_0.2_1 fuse_1_0.5_1 fuse_1_1_1; do $P run_content.py --features $f --k 10 --pop-beta 5 7 10; done
# 채택 설정 final / inner
for m in final inner; do $P run_content.py --features fuse_1_0.5_1 --k 10 --pop-beta 7 --mode $m; done
