#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/src"
P=${PYTHON:-python}
# D036 팀원 B BM25-Shrink 이식 (item IDF, k 2000, shrink 50, power 0.5, pop 0.4) — val/final/inner top-100
for m in val final inner; do $P cf.py bm25 --k 2000 --pop 0.4 --mode $m; done
# D042 BM25 grid (val) → 최고 k 4000, b 0.25 (0.04359)
for k in 500 1000 4000; do $P cf.py bm25 --k $k --pop 0.4; done
for p in 0 0.2 0.5; do $P cf.py bm25 --k 2000 --pop $p; done
for sh in 10 200; do $P cf.py bm25 --k 2000 --pop 0.4 --shrink $sh; done
for pw in 0.75 1; do $P cf.py bm25 --k 2000 --pop 0.4 --power $pw; done
for b in 0 0.25 0.5 1.0; do $P cf.py bm25 --k 2000 --pop 0.4 --b $b; done
$P cf.py bm25 --k 200 --pop 0.4 --b 0
$P cf.py bm25 --k 4000 --pop 0.4 --b 0.5; $P cf.py bm25 --k 4000 --pop 0.2 --b 0.5; $P cf.py bm25 --k 4000 --pop 0.4 --b 0.25
for m in final inner inner2; do $P cf.py bm25 --k 4000 --pop 0.4 --b 0.25 --mode $m; done   # D050·D052 입력
