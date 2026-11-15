# 팀원 C 코드 원문 (C009-C016 노트북 셀 6). 수정하지 않음.
"""LightGCN backbone with SGL-ED or NCL-style SSL, for the existing team kit.

Faithfulness note:
- SGL-ED: independently edge-subsampled, re-normalized graph views, InfoNCE.
- NCL: layer-0 versus 2h-hop structural InfoNCE, user/item prototypes from
  detached spherical torch k-means; not bit-for-bit identical to RecBole/FAISS.
Validation truth is used strictly for model selection and reporting.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from graph_models import (
    LightGCN, _training_loss, load_data, load_truth, normalized_graph,
    predict_top20, recall_at_20, sample_triples, write_predictions,
)


def edge_view(positive_users, positive_items, n_users, n_items, drop, device):
    """Independent sampled bipartite graph, symmetric, degree re-normalized."""
    if not 0 <= drop < 1:
        raise ValueError('drop ratio must be in [0,1)')
    keep = torch.rand(positive_users.numel(), device=device) >= drop
    u, i = positive_users[keep], positive_items[keep] + n_users
    row = torch.cat((u, i))
    col = torch.cat((i, u))
    counts = torch.bincount(row, minlength=n_users + n_items).float()
    values = counts[row].clamp_min(1).rsqrt() * counts[col].clamp_min(1).rsqrt()
    return torch.sparse_coo_tensor(
        torch.stack((row, col)), values,
        (n_users + n_items, n_users + n_items), device=device
    ).coalesce()


def propagation(model, graph, depth):
    curr = torch.cat((model.user_embedding.weight, model.item_embedding.weight), dim=0)
    steps = [curr]
    for _ in range(depth):
        curr = torch.sparse.mm(graph, curr)
        steps.append(curr)
    mean = torch.stack(steps[:model.layers + 1], 0).mean(0)
    n_users = model.user_embedding.num_embeddings
    return mean[:n_users], mean[n_users:], steps


def nce_full(anchor, positive, candidate_all, tau):
    """Sampled anchor nodes, ALL same-type nodes used in InfoNCE denominator."""
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    candidate_all = F.normalize(candidate_all, dim=-1)
    logits = anchor @ candidate_all.T / tau
    pos = (anchor * positive).sum(-1) / tau
    return (torch.logsumexp(logits, dim=1) - pos).mean()


def sgl_info_nce(view1, view2, sampled_users, sampled_items, n_users, tau):
    loss_u = nce_full(
        view1[0][sampled_users], view2[0][sampled_users], view2[0], tau
    )
    loss_i = nce_full(
        view1[1][sampled_items], view2[1][sampled_items], view2[1], tau
    )
    return loss_u + loss_i


@torch.no_grad()
def spherical_kmeans(x, n_clusters, iterations=5):
    """GPU spherical k-means to avoid requiring faiss-gpu/RecBole."""
    x = F.normalize(x.detach(), dim=1)
    n = x.size(0)
    if n_clusters > n:
        raise ValueError('Clusters exceed node count')
    centers = x[torch.randperm(n, device=x.device)[:n_clusters]].clone()
    for _ in range(iterations):
        labels = (x @ centers.T).argmax(dim=1)
        sums = torch.zeros_like(centers)
        sums.index_add_(0, labels, x)
        counts = torch.bincount(labels, minlength=n_clusters)
        empty = torch.where(counts == 0)[0]
        if empty.numel():
            sums[empty] = x[torch.randint(n, (empty.numel(),), device=x.device)]
        centers = F.normalize(sums, dim=1)
    labels = (x @ centers.T).argmax(dim=1)
    return centers.detach(), labels.detach()


def prototype_nce(raw_users, raw_items, sample_users, sample_items,
                  user_clusters, item_clusters, tau):
    cent_u, lab_u = user_clusters
    cent_i, lab_i = item_clusters
    u = F.normalize(raw_users[sample_users], dim=-1)
    i = F.normalize(raw_items[sample_items], dim=-1)
    lu = F.cross_entropy(u @ cent_u.T / tau, lab_u[sample_users])
    li = F.cross_entropy(i @ cent_i.T / tau, lab_i[sample_items])
    return lu + li


def evaluate(model, graph, data, truth, eval_batch):
    predictions = predict_top20(model, graph, data, eval_batch)
    return recall_at_20(predictions, truth)


def train_run(a):
    if a.epochs < 1 or a.layers < 1 or a.dim < 1 or a.samples_per_epoch < 1:
        raise ValueError('epochs/layers/dim/samples_per_epoch must be positive')
    if a.ssl_nodes < 1 or a.eval_every < 1 or a.tau <= 0 or a.lr <= 0:
        raise ValueError('ssl_nodes/eval_every/tau/lr must be positive')
    if a.model == 'ncl' and (a.hyper_layers < 1 or a.clusters < 2 or a.cluster_every < 1):
        raise ValueError('invalid NCL clustering/hyper-layer configuration')
    if a.ssl_weight < 0 or a.proto_weight < 0 or a.reg < 0:
        raise ValueError('loss weights must be nonnegative')
    if a.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    device = torch.device(a.device)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(a.seed)
    rng = random.Random(a.seed)
    data = load_data(a.fit, a.users)
    truth = load_truth(a.truth, data.target_users) if a.truth else None
    graph = normalized_graph(data, device)
    edge_u = torch.tensor(data.positive_users, dtype=torch.long, device=device)
    edge_i = torch.tensor(data.positive_items, dtype=torch.long, device=device)
    model = LightGCN(data.num_users, data.num_items, a.dim, a.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    print(json.dumps({'info':'START', 'model':a.model, 'users':data.num_users, 'items':data.num_items,
                      'fit_edges':data.num_edges,'layers':a.layers,'dim':a.dim,
                      'device':str(device),'epochs':a.epochs, 'ssl_weight':a.ssl_weight,
                      'proto_weight':a.proto_weight}), flush=True)
    best_score = -1.0
    best_epoch = None
    best_state = None
    stale_evals = 0
    current_clusters = None
    last_loss = None
    trained_epochs = 0
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, a.epochs + 1):
        tick = time.monotonic()
        trained_epochs = epoch
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sampled = sample_triples(data, a.samples_per_epoch, rng, device)
        anchor_u = torch.unique(sampled[0][:a.ssl_nodes])
        anchor_i = torch.unique(sampled[1][:a.ssl_nodes])
        if a.model == 'ncl':
            u_vec, i_vec, steps = propagation(model, graph, max(a.layers, 2 * a.hyper_layers))
        else:
            u_vec, i_vec = model(graph)
            steps = None
        bpr_reg, bpr, reg = _training_loss(model, u_vec, i_vec, sampled, a.reg)
        ssl = bpr_reg.new_zeros(())
        proto = bpr_reg.new_zeros(())
        if a.model == 'sgl' and a.ssl_weight > 0:
            g1 = edge_view(edge_u, edge_i, data.num_users, data.num_items, a.drop, device)
            g2 = edge_view(edge_u, edge_i, data.num_users, data.num_items, a.drop, device)
            e1 = model(g1)
            e2 = model(g2)
            ssl = sgl_info_nce(e1, e2, anchor_u, anchor_i, data.num_users, a.tau)
        elif a.model == 'ncl':
            raw = steps[0]
            context = steps[2 * a.hyper_layers]
            raw_user = raw[:data.num_users]
            raw_item = raw[data.num_users:]
            ctx_user = context[:data.num_users]
            ctx_item = context[data.num_users:]
            if a.ssl_weight > 0:
                ssl = (nce_full(ctx_user[anchor_u], raw_user[anchor_u], raw_user, a.tau)
                       + a.alpha * nce_full(ctx_item[anchor_i], raw_item[anchor_i], raw_item, a.tau))
            if a.proto_weight > 0 and epoch >= a.warmup:
                if current_clusters is None or (epoch - a.warmup) % a.cluster_every == 0:
                    current_clusters = (
                        spherical_kmeans(raw_user, a.clusters),
                        spherical_kmeans(raw_item, a.clusters),
                    )
                proto = prototype_nce(raw_user, raw_item, anchor_u, anchor_i,
                                      current_clusters[0], current_clusters[1], a.tau)
        loss = bpr_reg + a.ssl_weight * ssl + a.proto_weight * proto
        if not torch.isfinite(loss):
            raise RuntimeError(f'Loss non-finite at epoch {epoch}: {float(loss)}')
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())
        record = {'epoch':epoch, 'loss':round(last_loss,6),
                  'bpr':round(float(bpr.detach()),6),
                  'reg':round(float(reg.detach()),6),
                  'ssl':round(float(ssl.detach()),6),
                  'proto':round(float(proto.detach()),6),
                  'seconds':round(time.monotonic()-tick,3)}
        del sampled, u_vec, i_vec, steps, loss, bpr_reg, bpr, reg, ssl, proto
        should_eval = truth is not None and (epoch == 1 or epoch % a.eval_every == 0 or epoch == a.epochs)
        if should_eval:
            score = evaluate(model, graph, data, truth, a.eval_batch)
            record.update(score)
            if score['recall_at_20'] > best_score:
                best_score = score['recall_at_20']
                best_epoch = epoch
                best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                stale_evals = 0
                record['best'] = True
            else:
                stale_evals += 1
                record['best'] = False
        elif epoch % 50 != 0:
            continue  # emit concise logs only, not 700+ lines per run
        print(json.dumps(record), flush=True)
        if should_eval and a.patience > 0 and stale_evals >= a.patience:
            print(json.dumps({'early_stop_epoch':epoch}),flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    predictions = predict_top20(model, graph, data, a.eval_batch)
    write_predictions(out_path, data.target_users, predictions)
    result = {
        'model':a.model, 'output':str(out_path), 'best_epoch':best_epoch,
        'trained_epochs':trained_epochs,'epochs_requested':a.epochs,'device':str(device),
        'layers':a.layers,'dim':a.dim,'lr':a.lr,'reg':a.reg,'ssl_weight':a.ssl_weight,
        'proto_weight':a.proto_weight,'drop':a.drop,'tau':a.tau,'clusters':a.clusters,
        'warmup':a.warmup,'hyper_layers':a.hyper_layers,'alpha':a.alpha,
        'samples_per_epoch':a.samples_per_epoch,'seed':a.seed,'last_loss':last_loss,
    }
    if truth is not None:
        result.update(recall_at_20(predictions,truth))
    if a.metrics:
        Path(a.metrics).parent.mkdir(parents=True,exist_ok=True)
        Path(a.metrics).write_text(json.dumps(result,indent=2),encoding='utf-8')
    if a.checkpoint:
        torch.save({'model':a.model,'weights':model.state_dict(),'config':vars(a)},a.checkpoint)
    print(json.dumps({'final':result}),flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model',choices=['sgl','ncl'],required=True)
    p.add_argument('--fit',required=True)
    p.add_argument('--users',required=True)
    p.add_argument('--truth',default=None)
    p.add_argument('--out',required=True)
    p.add_argument('--metrics',default=None)
    p.add_argument('--checkpoint',default=None)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--epochs',type=int,default=700)
    p.add_argument('--eval-every',type=int,default=25)
    p.add_argument('--patience',type=int,default=0)
    p.add_argument('--eval-batch',type=int,default=256)
    p.add_argument('--dim',type=int,default=64)
    p.add_argument('--layers',type=int,default=3)
    p.add_argument('--lr',type=float,default=.002)
    p.add_argument('--reg',type=float,default=.0001)
    p.add_argument('--samples-per-epoch',type=int,default=100000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--ssl-nodes',type=int,default=192)
    p.add_argument('--ssl-weight',type=float,default=.05)
    p.add_argument('--drop',type=float,default=.1)
    p.add_argument('--tau',type=float,default=.2)
    p.add_argument('--proto-weight',type=float,default=.005)
    p.add_argument('--clusters',type=int,default=64)
    p.add_argument('--cluster-every',type=int,default=50)
    p.add_argument('--warmup',type=int,default=50)
    p.add_argument('--hyper-layers',type=int,default=1)
    p.add_argument('--alpha',type=float,default=1.)
    a=p.parse_args()
    train_run(a)

if __name__=='__main__':
    main()