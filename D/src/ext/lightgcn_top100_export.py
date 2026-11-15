# 팀원 C 코드 원문 (C/C001_LightGCN_Top100_export.ipynb 셀 4). 수정하지 않음.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate separate validation/final scored Top-100 files with the C group's LightGCN.

This script uses the *original* graph_models.py LightGCN, BPR loss, edge sampler,
normalized graph, and data ID handling (no substitute algorithm). The original
predict_top20 routine writes only 20 IDs without scores; this exporter streams
100 **scored** recommendations per target user instead.

Run (Kaggle working directory, with graph_models.py in the same directory):
  python lightgcn_top100_export.py \
    --validation-dir /kaggle/input/datasets/swaidataset/amazon-topk-teamkit/amazon_topk_teamkit/validation \
    --full-train /kaggle/input/competitions/ssu-data-science-amazon-top-k-recommendation/train.csv \
    --test-users /kaggle/input/competitions/ssu-data-science-amazon-top-k-recommendation/test.csv \
    --output-dir /kaggle/working/C_lightgcn_top100

Optional: --seeds 42,2026,3407 combines multiple seeds' **scores**. Single
seed 42 is the default original-best single-run configuration, not a measured
multi-seed result.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import json
import math
import random
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

try:
    import graph_models as gm
except ImportError as exc:
    raise SystemExit('Missing graph_models.py. Put the original C model implementation beside this script.') from exc

SPLIT_ID = 'f0dfad2f4431a876'
TOPK = 100


def load_truth(path: Path) -> dict[str, set[str]]:
    return {u: s for u, s in gm.load_truth(path, list(read_users(path))).items()}


def read_users(path: Path) -> list[str]:
    # For val_truth.csv, we use unique user IDs; for val_users/test.csv, one row/user is expected.
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        rdr = csv.DictReader(f)
        if not rdr.fieldnames or 'user_id' not in rdr.fieldnames:
            raise ValueError(f'{path} needs user_id column')
        return list(dict.fromkeys(row['user_id'] for row in rdr))


def check_validation_split(validation: Path):
    manifest = validation / 'manifest.json'
    if not manifest.exists():
        raise FileNotFoundError(f'Missing validation manifest: {manifest}')
    data = json.loads(manifest.read_text(encoding='utf8'))
    if data.get('split_id') != SPLIT_ID:
        raise ValueError(f'Unexpected validation split: {data.get("split_id")}')
    try:
        import teamkit
        result = teamkit.verify_split(validation)
        if result.get('split_id') != SPLIT_ID:
            raise ValueError('teamkit.verify_split split ID mismatch')
    except ImportError:
        print('WARNING: teamkit.py unavailable: checked manifest split ID only, not dataset hashes.', flush=True)
    print('Validation split:', SPLIT_ID, flush=True)


def train_one(data: gm.InteractionData, graph: torch.Tensor, device: torch.device,
              *, seed: int, epochs: int, dim: int, layers: int, lr: float,
              reg: float, samples_per_epoch: int, truth_file: Path | None,
              eval_every: int, eval_batch: int):
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)
    model = gm.LightGCN(data.num_users, data.num_items, dim, layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_epoch, best_state = float('-inf'), 0, None
    truth = gm.load_truth(truth_file, data.target_users) if truth_file else None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sampled = gm.sample_triples(data, samples_per_epoch, rng, device)
        uv, iv = model(graph)
        loss, bpr, regularization = gm._training_loss(model, uv, iv, sampled, reg)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print(json.dumps({'seed':seed, 'epoch':epoch, 'epochs':epochs,
                'loss':round(float(loss.detach().cpu()),6)}), flush=True)
        del sampled, uv, iv, loss, bpr, regularization
        if truth is not None and (epoch % eval_every == 0 or epoch == epochs):
            pred = gm.predict_top20(model, graph, data, eval_batch)
            result = gm.recall_at_20(pred, truth)
            metric = result['recall_at_20']
            if metric > best:
                best, best_epoch, best_state = metric, epoch, copy.deepcopy(model.state_dict())
            print(json.dumps({'seed':seed,'epoch':epoch, 'val_recall_at_20':metric,
                              'best_val_recall':best, 'best_epoch':best_epoch}),flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        uv, iv = model(graph)
    print(json.dumps({'seed':seed,'best_epoch':best_epoch or epochs,
                      'best_recall':best if truth else None}),flush=True)
    return uv.detach().cpu(), iv.detach().cpu(), {'seed':seed, 'best_epoch':best_epoch or epochs,
                                                    'best_val_recall':best if truth else None}


def predict_and_write(data: gm.InteractionData, embeddings: list[tuple[torch.Tensor,torch.Tensor]],
                      out: Path, device: torch.device, batch_size: int, topk:int = TOPK):
    # Use raw dot product scores, average multiple seeds, and mask every seen item.
    if data.num_items < topk+1:
        raise ValueError('Need at least 101 catalog items for Top100')
    out.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(out, 'wt', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['user_id','item_id','score','rank'])
        n = len(data.target_users)
        for start in range(0,n,batch_size):
            users = data.target_users[start:start+batch_size]
            idx = torch.tensor([data.user_index[u] for u in users],device=device,dtype=torch.long)
            scores = None
            for uv,iv in embeddings:
                uv_batch = uv.to(device)[idx] if uv.device == device else uv[idx.cpu()].to(device)
                iv_gpu = iv.to(device)
                cur = uv_batch @ iv_gpu.t()
                scores = cur if scores is None else scores + cur
            scores /= len(embeddings)
            for ridx,user in enumerate(users):
                seen = list(data.seen_items[data.user_index[user]])
                scores[ridx,seen] = -torch.inf
            values, indices = torch.topk(scores,topk,dim=1)
            val_rows = values.cpu().tolist()
            ind_rows = indices.cpu().tolist()
            for user,vals,inds in zip(users,val_rows,ind_rows):
                if not all(math.isfinite(v) for v in vals):
                    raise ValueError(f'User {user} has <{topk} unseen items')
                for rank,(val,item_idx) in enumerate(zip(vals,inds),1):
                    writer.writerow([user,data.item_ids[item_idx],format(val,'.9g'),rank])
            if (start//batch_size)%100 == 0:
                print(f'{out.name}: {min(start+batch_size,n):,}/{n:,} users',flush=True)
    return out


def check_output(path:Path, users:list[str], topk:int=TOPK):
    # Streaming validation: no huge DataFrame is created.
    rows=0; last=None; rank=0; user_count=0; seen_items=set(); prev_score=float('inf')
    with gzip.open(path,'rt',encoding='utf8',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames != ['user_id','item_id','score','rank']:
            raise ValueError('Bad output header')
        for row in reader:
            u=row['user_id']; item=row['item_id']; value=float(row['score']); r=int(row['rank'])
            if u!=last:
                if last is not None and rank!=topk:
                    raise ValueError(f'User {last} has only {rank} items')
                if user_count >= len(users) or u != users[user_count]:
                    raise ValueError('User order mismatch / extra user')
                last=u;rank=0;seen_items=set();prev_score=float('inf');user_count+=1
            rank+=1
            if r!=rank or item in seen_items or value>prev_score+1e-6:
                raise ValueError(f'Invalid Top100 rank/duplicates/order for {u}')
            prev_score=value;seen_items.add(item);rows+=1
    if user_count!=len(users) or rank!=topk or rows!=len(users)*topk:
        raise ValueError('Output row count mismatch')
    print(f'✅ {path.name}: {rows:,} rows; {user_count:,} users, exactly Top{topk}',flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--validation-dir',type=Path,required=True,
                    help='Folder with val_train.csv, val_users.csv, val_truth.csv, manifest.json')
    ap.add_argument('--full-train',type=Path,required=True)
    ap.add_argument('--test-users',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,default=Path('/kaggle/working/C_lightgcn_top100'))
    ap.add_argument('--epochs',type=int,default=575)
    ap.add_argument('--full-epochs',type=int,default=575,
                    help='Train full data for this many epochs (not using validation truth)')
    ap.add_argument('--dim',type=int,default=128)
    ap.add_argument('--layers',type=int,default=3)
    ap.add_argument('--lr',type=float,default=0.003)
    ap.add_argument('--reg',type=float,default=0.0001)
    ap.add_argument('--samples-per-epoch',type=int,default=100000)
    ap.add_argument('--seeds',default='42',help='Comma-separated integer seeds, e.g. 42,2026,3407')
    ap.add_argument('--eval-every',type=int,default=25)
    ap.add_argument('--batch-size',type=int,default=128)
    ap.add_argument('--device',default='auto')
    ap.add_argument('--skip-val',action='store_true',help='Only generate final file')
    ap.add_argument('--skip-final',action='store_true',help='Only generate validation file')
    args=ap.parse_args()
    if args.skip_val and args.skip_final:ap.error('Nothing to do')
    if args.epochs<1 or args.full_epochs<1 or args.samples_per_epoch<1 or args.batch_size<1:
        ap.error('epochs, full-epochs, samples-per-epoch, batch-size must be positive')
    seeds=[int(x) for x in args.seeds.split(',') if x.strip()]
    if not seeds or len(seeds)!=len(set(seeds)):ap.error('Provide one or more distinct seeds')
    device=torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    check_validation_split(args.validation_dir)
    metadata={'model':'LightGCN','default_best_single_config':{'dim':128,'layers':3,'lr':0.003,'epochs':575},
              'config':{'dim':args.dim,'layers':args.layers,'lr':args.lr,'reg':args.reg,
                        'epochs':args.epochs,'full_epochs':args.full_epochs,'samples_per_epoch':args.samples_per_epoch,
                        'seeds':seeds,'topk':TOPK},'validation_split':SPLIT_ID,'results':{}}
    for stage in ('val','final'):
        if (stage=='val' and args.skip_val) or (stage=='final' and args.skip_final): continue
        fit=args.validation_dir/'val_train.csv' if stage=='val' else args.full_train
        users=args.validation_dir/'val_users.csv' if stage=='val' else args.test_users
        truth=args.validation_dir/'val_truth.csv' if stage=='val' else None
        epochs=args.epochs if stage=='val' else args.full_epochs
        data=gm.load_data(fit,users)
        for user in data.target_users:
            if data.num_items-len(data.seen_items[data.user_index[user]])<TOPK:
                raise ValueError(f'Not enough unseen items for user {user}')
        graph=gm.normalized_graph(data,device)
        all_embeddings=[];stats=[]
        for seed in seeds:
            uv,iv,meta=train_one(data,graph,device,seed=seed,epochs=epochs,dim=args.dim,
                   layers=args.layers,lr=args.lr,reg=args.reg,samples_per_epoch=args.samples_per_epoch,
                   truth_file=truth,eval_every=args.eval_every,eval_batch=args.batch_size)
            all_embeddings.append((uv,iv));stats.append(meta)
        path=args.output_dir/f'C_lightgcn_{stage}_top100.csv.gz'
        predict_and_write(data,all_embeddings,path,device,args.batch_size)
        check_output(path,data.target_users)
        metadata['results'][stage]={'path':str(path),'users':len(data.target_users),'rows':len(data.target_users)*TOPK,
                                    'trained_seeds':stats}
        (args.output_dir/'run_metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf8')
        del graph,data,all_embeddings
    print('✅ All requested Top100 files generated successfully.',flush=True)

if __name__=='__main__':main()
