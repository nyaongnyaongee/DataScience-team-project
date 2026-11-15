# 팀원 C 코드 원문 (C/C009-C012_C014-C016_UltraGCN_SimpleX_RRF.ipynb 셀 7). 수정하지 않음.
"""UltraGCN-style and SimpleX-style implicit top-k recommendation.

Designed for the team's provided split and single CUDA GPU (also CPU testable).
Faithfulness notes:
- UltraGCN-style: direct user/item embeddings, degree-aware weighted BCE,
  and item-item co-occurrence constraint. This is a streamlined approximation
  of UltraGCN, not a bit-for-bit reproduction of the published implementation.
- SimpleX-style: user ID + mean item-history tower, cosine contrastive loss,
  leave-one-out history for positive training samples. This is a compact
  implementation of the SimpleX family, not the official codebase.
- Validation positives must NEVER be included in fit history or item neighbors.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
import torch.nn.functional as F

from graph_models import load_data, load_truth, recall_at_20, write_predictions


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_interaction_matrix(data):
    rows = np.asarray(data.positive_users, dtype=np.int64)
    cols = np.asarray(data.positive_items, dtype=np.int64)
    mat = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                        shape=(data.num_users, data.num_items)).tocsr()
    mat.sum_duplicates()
    mat.data[:] = 1.0
    return mat


def ii_neighbors(mat, k: int):
    """Top-k co-occurrence cosine neighbors from FIT ONLY; padded weights=0."""
    n_items = mat.shape[1]
    ids = np.zeros((n_items, k), dtype=np.int64)
    weights = np.zeros((n_items, k), dtype=np.float32)
    deg = np.maximum(np.asarray(mat.getnnz(axis=0), dtype=np.float32), 1)
    co = (mat.T @ mat).tocsr()
    inv = np.sqrt(deg)
    for i in range(n_items):
        start, end = co.indptr[i], co.indptr[i+1]
        nbrs = co.indices[start:end]
        vals = co.data[start:end]
        if not len(nbrs):
            continue
        scores = vals / (inv[i] * inv[nbrs])
        keep = nbrs != i
        nbrs, scores = nbrs[keep], scores[keep]
        if not len(nbrs):
            continue
        take = min(k, len(nbrs))
        if take < len(nbrs):
            pos = np.argpartition(scores, -take)[-take:]
        else:
            pos = np.arange(len(nbrs))
        pos = pos[np.argsort(-scores[pos], kind='stable')]
        ids[i, :take] = nbrs[pos]
        values = scores[pos].astype(np.float32)
        weights[i, :take] = values / max(values.sum(), 1e-9)
    return ids, weights


class UltraStyle(nn.Module):
    def __init__(self, n_users, n_items, dim):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=.05)
        nn.init.normal_(self.item_emb.weight, std=.05)

    def vectors(self, mat=None):
        return self.user_emb.weight, self.item_emb.weight


class SimpleXStyle(nn.Module):
    def __init__(self, n_users, n_items, dim, gamma):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.gamma = gamma
        nn.init.normal_(self.user_emb.weight, std=.05)
        nn.init.normal_(self.item_emb.weight, std=.05)

    def vectors(self, mat):
        # CSR FIT-history aggregation only; do not include held-out labels.
        history = torch.sparse_csr_tensor(
            torch.from_numpy(mat.indptr.astype(np.int64)).to(self.item_emb.weight.device),
            torch.from_numpy(mat.indices.astype(np.int64)).to(self.item_emb.weight.device),
            torch.from_numpy(mat.data.astype(np.float32)).to(self.item_emb.weight.device),
            mat.shape, device=self.item_emb.weight.device)
        lengths = torch.tensor(np.maximum(mat.getnnz(axis=1), 1),
                               dtype=torch.float32, device=self.item_emb.weight.device)
        agg = torch.sparse.mm(history, self.item_emb.weight) / lengths[:, None]
        return (F.normalize(self.user_emb.weight + self.gamma * agg, dim=-1),
                F.normalize(self.item_emb.weight, dim=-1))


def history_sum_for_batch(item_emb, flat_hist, rowptr, users):
    """Differentiable ragged history sum, for selected users only."""
    starts = rowptr[users]
    lengths = rowptr[users + 1] - starts
    offsets = torch.cat([torch.zeros((1,), device=users.device, dtype=torch.long),
                         lengths.cumsum(0)[:-1]])
    owner = torch.repeat_interleave(torch.arange(len(users), device=users.device), lengths)
    flat_positions = torch.repeat_interleave(starts, lengths) + (
        torch.arange(owner.numel(), device=users.device) - offsets[owner]
    )
    bag_idx = flat_hist[flat_positions]
    sums = F.embedding_bag(bag_idx, item_emb.weight, offsets,
                           mode='sum', include_last_offset=False)
    return sums, lengths


def sample_batch(users_all, items_all, n_items, observed_codes, batch, n_negs):
    dev = users_all.device
    sampled = torch.randint(len(users_all), (batch,), device=dev)
    users = users_all[sampled]
    positives = items_all[sampled]
    negatives = torch.randint(n_items, (batch, n_negs), device=dev)
    for _ in range(15):
        ids = users[:, None] * n_items + negatives
        pos = torch.searchsorted(observed_codes, ids.reshape(-1)).clamp_max(len(observed_codes)-1)
        mask = (observed_codes[pos].reshape_as(ids) == ids)
        if not mask.any().item():
            break
        negatives[mask] = torch.randint(n_items, (int(mask.sum().item()),), device=dev)
    else:
        raise RuntimeError('Unable to sample unobserved negatives')
    return users, positives, negatives


def evaluate(model, data, mat, truth, device, batch_size=128):
    model.eval()
    predictions = {}
    with torch.no_grad():
        user_vectors, item_vectors = model.vectors(mat)
        item_trans = item_vectors.T.contiguous()
        for start in range(0, len(data.target_users), batch_size):
            batch_users = data.target_users[start:start+batch_size]
            uidx = [data.user_index[u] for u in batch_users]
            scores = user_vectors[torch.tensor(uidx, device=device)] @ item_trans
            for r, ui in enumerate(uidx):
                seen = data.seen_items.get(ui, set())
                if seen:
                    scores[r, torch.tensor(list(seen), dtype=torch.long,device=device)] = -float('inf')
            top = scores.topk(20, dim=1, sorted=True).indices.cpu().tolist()
            for user, idxs in zip(batch_users, top):
                predictions[user] = [data.item_ids[i] for i in idxs]
        score = recall_at_20(predictions, truth) if truth else None
    return predictions, score


def run(a):
    assert a.epochs > 0 and a.eval_every > 0 and a.batch_size > 0 and a.steps_per_epoch > 0
    assert a.negatives > 0 and a.lr > 0 and a.reg >= 0
    device = torch.device(a.device)
    seed_all(a.seed)
    data = load_data(a.fit, a.users)
    truth = load_truth(a.truth, data.target_users) if a.truth else None
    mat = get_interaction_matrix(data)
    if a.model == 'ultragcn':
        model = UltraStyle(data.num_users, data.num_items, a.dim).to(device)
        user_degree = torch.tensor(np.maximum(mat.getnnz(axis=1), 1), dtype=torch.float32, device=device)
        item_degree = torch.tensor(np.maximum(mat.getnnz(axis=0), 1), dtype=torch.float32, device=device)
        if a.ii_weight > 0:
            ii_id, ii_wt = ii_neighbors(mat, a.ii_topk)
            ii_id = torch.from_numpy(ii_id).to(device)
            ii_wt = torch.from_numpy(ii_wt).to(device)
        else:
            ii_id, ii_wt = None, None
    elif a.model == 'simplex':
        model = SimpleXStyle(data.num_users, data.num_items, a.dim, a.gamma).to(device)
        rowptr = torch.tensor(mat.indptr.astype(np.int64), device=device)
        flat_history = torch.tensor(mat.indices.astype(np.int64), device=device)
    else:
        raise ValueError('Unexpected model')
    optim = torch.optim.Adam(model.parameters(), lr=a.lr)
    u_edges = torch.tensor(data.positive_users, dtype=torch.long, device=device)
    i_edges = torch.tensor(data.positive_items, dtype=torch.long, device=device)
    observed = (u_edges * data.num_items + i_edges).sort().values.unique(sorted=True)

    best_epoch = None
    best_score = -1.0
    best_state = None
    stale = 0
    last_loss = None
    for epoch in range(1, a.epochs+1):
        begin = time.time()
        model.train()
        loss_acc = 0.0
        for _ in range(a.steps_per_epoch):
            u, i, j = sample_batch(u_edges, i_edges, data.num_items, observed,
                                   a.batch_size, a.negatives)
            optim.zero_grad(set_to_none=True)
            if a.model == 'ultragcn':
                uv = model.user_emb(u)
                pv = model.item_emb(i)
                nv = model.item_emb(j)
                pos = (uv * pv).sum(-1)
                neg = (uv[:,None,:] * nv).sum(-1)
                # Degree-aware confidence for user-item constraint.
                confidence = (user_degree[u] / item_degree[i]).sqrt().clamp(0.25, 4.0)
                bce_pos = (1.0 + a.beta * confidence) * F.softplus(-pos)
                bce_neg = F.softplus(neg).mean(-1) * a.neg_weight
                loss = (bce_pos + bce_neg).mean()
                if ii_id is not None:
                    nbr_emb = model.item_emb(ii_id[i])
                    nbr_s = (uv[:,None,:] * nbr_emb).sum(-1)
                    item_loss = (F.softplus(-nbr_s) * ii_wt[i]).sum(-1).mean()
                    loss = loss + a.ii_weight * item_loss
                reg = .5*(uv.square().sum(-1).mean()+pv.square().sum(-1).mean()+nv.square().sum(-1).mean())
            else:
                # Leave the current target item OUT of the user's training history.
                history_sum, lengths = history_sum_for_batch(model.item_emb, flat_history, rowptr, u)
                agg_loo = (history_sum - model.item_emb(i)) / (lengths-1).clamp_min(1)[:,None]
                user_r = F.normalize(model.user_emb(u) + model.gamma * agg_loo, dim=-1)
                p = F.normalize(model.item_emb(i), dim=-1)
                n = F.normalize(model.item_emb(j), dim=-1)
                pos_s = (user_r * p).sum(-1)
                neg_s = (user_r[:,None,:] * n).sum(-1)
                loss = ((1.0 - pos_s) + a.neg_weight * F.relu(neg_s-a.margin).square().mean(-1)).mean()
                reg = .5*(model.user_emb(u).square().sum(-1).mean()+model.item_emb(i).square().sum(-1).mean()+model.item_emb(j).square().sum(-1).mean())
            loss = loss + a.reg * reg
            if not torch.isfinite(loss).item():
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            optim.step()
            loss_acc += loss.item()
        last_loss = loss_acc / a.steps_per_epoch
        if (epoch % a.eval_every == 0 or epoch == a.epochs or epoch == 1) and truth:
            preds, result = evaluate(model, data, mat, truth, device, a.eval_batch)
            best = result['recall_at_20'] > best_score
            if best:
                best_score = result['recall_at_20']
                best_epoch = epoch
                best_state = copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
                stale = 0
            else:
                stale += 1
            print(json.dumps({'epoch':epoch,'loss':last_loss,'recall_at_20':result['recall_at_20'],
                              'hits':result['hits'],'best':best,'seconds':round(time.time()-begin,2)}),flush=True)
            if a.patience>0 and stale>=a.patience:
                break
        elif epoch % max(a.eval_every, 10) == 0 or epoch == a.epochs:
            print(json.dumps({'epoch':epoch,'loss':last_loss,'seconds':round(time.time()-begin,2)}),flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    predictions, result = evaluate(model, data, mat, truth, device, a.eval_batch)
    write_predictions(a.out, data.target_users, predictions)
    output = dict(model=a.model, out=a.out, fit=a.fit, best_epoch=best_epoch,
                  trained_epochs=epoch, epochs_requested=a.epochs, last_loss=last_loss,
                  dim=a.dim, lr=a.lr, negatives=a.negatives, reg=a.reg,
                  steps_per_epoch=a.steps_per_epoch, batch_size=a.batch_size,
                  ii_weight=a.ii_weight, beta=a.beta, gamma=a.gamma,
                  margin=a.margin, neg_weight=a.neg_weight, seed=a.seed)
    if result:
        output.update(result)
    if a.metrics:
        Path(a.metrics).write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps({'final':output}), flush=True)
    return output


def main():
    p=argparse.ArgumentParser(description='UltraGCN-style and SimpleX-style collaborative filtering')
    p.add_argument('--model', choices=['ultragcn','simplex'], required=True)
    p.add_argument('--fit', required=True)
    p.add_argument('--users', required=True)
    p.add_argument('--truth', default=None)
    p.add_argument('--out', required=True)
    p.add_argument('--metrics', default=None)
    p.add_argument('--epochs',type=int,default=70)
    p.add_argument('--eval-every',type=int,default=10)
    p.add_argument('--patience',type=int,default=5)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--dim',type=int,default=128)
    p.add_argument('--batch-size',type=int,default=2048)
    p.add_argument('--steps-per-epoch',type=int,default=40)
    p.add_argument('--negatives',type=int,default=10)
    p.add_argument('--lr',type=float,default=.002)
    p.add_argument('--reg',type=float,default=1e-5)
    p.add_argument('--ii-weight',type=float,default=.1)
    p.add_argument('--ii-topk',type=int,default=10)
    p.add_argument('--beta',type=float,default=.1)
    p.add_argument('--neg-weight',type=float,default=4.)
    p.add_argument('--gamma',type=float,default=1.)
    p.add_argument('--margin',type=float,default=.2)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--eval-batch',type=int,default=128)
    a=p.parse_args()
    run(a)


if __name__=='__main__':
    main()
