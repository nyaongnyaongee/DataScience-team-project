#!/bin/bash
# D042 BM25 grid (val)
cd "$(dirname "$0")/../../src"
P=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=0
for k in 1000 4000; do $P cf.py bm25 --k $k --pop 0.4 || echo "ISSUE D042 k$k"; done
for p in 0 0.2 0.5; do $P cf.py bm25 --k 2000 --pop $p || echo "ISSUE D042 p$p"; done
for sh in 10 200; do $P cf.py bm25 --k 2000 --pop 0.4 --shrink $sh || echo "ISSUE D042 sh$sh"; done
for pw in 0.75 1; do $P cf.py bm25 --k 2000 --pop 0.4 --power $pw || echo "ISSUE D042 pw$pw"; done
for b in 0.5 1.0; do $P cf.py bm25 --k 2000 --pop 0.4 --b $b || echo "ISSUE D042 b$b"; done
echo DONE
