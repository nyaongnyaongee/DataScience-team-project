# 팀원 C 코드 원문 (C/C001_LightGCN_Top100_export.ipynb 셀 2). 수정하지 않음.
"""Single-GPU LightGCN / NGCF experiments for the shared Amazon Top-20 split.

Example:
    python graph_models.py --model lightgcn \
      --fit ../amazon_topk_teamkit/validation/val_train.csv \
      --users ../amazon_topk_teamkit/validation/val_users.csv \
      --truth ../amazon_topk_teamkit/validation/val_truth.csv \
      --out lightgcn_val.csv

No metadata or reviews are read. Train interactions are the only graph edges.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None


TOP_K = 20
_BaseModel = nn.Module if nn is not None else object


def _id_key(value: str):
    return (0, int(value)) if value.isdecimal() else (1, value)


def _csv_rows(path: str | Path, required: tuple[str, ...]):
    with open(path, "r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not set(required).issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain {', '.join(required)}")
        for row in reader:
            if None in row:
                raise ValueError(f"Malformed CSV row in {path}")
            yield row


def _identifier(value: str | None, column: str) -> str:
    if not value or value != value.strip() or any(char.isspace() for char in value):
        raise ValueError(f"Invalid {column}: {value!r}")
    return value


@dataclass
class InteractionData:
    user_ids: list[str]
    item_ids: list[str]
    user_index: dict[str, int]
    item_index: dict[str, int]
    positive_users: list[int]
    positive_items: list[int]
    seen_items: dict[int, set[int]]
    target_users: list[str]

    @property
    def num_users(self) -> int:
        return len(self.user_ids)

    @property
    def num_items(self) -> int:
        return len(self.item_ids)

    @property
    def num_edges(self) -> int:
        return len(self.positive_users)


def load_data(fit_path: str | Path, users_path: str | Path) -> InteractionData:
    """Index unique positive edges; do not filter interactions by rating."""
    edges: set[tuple[str, str]] = set()
    for row in _csv_rows(fit_path, ("user_id", "item_id")):
        edges.add((_identifier(row["user_id"], "user_id"), _identifier(row["item_id"], "item_id")))
    if not edges:
        raise ValueError("Empty fit interactions")

    target_users: list[str] = []
    target_set: set[str] = set()
    for row in _csv_rows(users_path, ("user_id",)):
        user_id = _identifier(row["user_id"], "user_id")
        if user_id in target_set:
            raise ValueError(f"Duplicate target user: {user_id}")
        target_set.add(user_id)
        target_users.append(user_id)

    train_users = {user for user, _ in edges}
    unseen_users = target_set - train_users
    if unseen_users:
        raise ValueError(f"{len(unseen_users)} target users have no training interaction")
    user_ids = sorted(train_users, key=_id_key)
    item_ids = sorted({item for _, item in edges}, key=_id_key)
    if len(item_ids) <= TOP_K:
        raise ValueError("Need more than 20 known items to recommend 20 unseen items")
    user_index = {value: index for index, value in enumerate(user_ids)}
    item_index = {value: index for index, value in enumerate(item_ids)}
    positive_users: list[int] = []
    positive_items: list[int] = []
    seen_items: dict[int, set[int]] = defaultdict(set)
    for user, item in sorted(edges, key=lambda pair: (_id_key(pair[0]), _id_key(pair[1]))):
        u, i = user_index[user], item_index[item]
        positive_users.append(u)
        positive_items.append(i)
        seen_items[u].add(i)
    for user in target_users:
        if len(seen_items[user_index[user]]) + TOP_K > len(item_ids):
            raise ValueError(f"User {user} has fewer than 20 unseen items")
    return InteractionData(
        user_ids, item_ids, user_index, item_index, positive_users,
        positive_items, dict(seen_items), target_users,
    )


def load_truth(path: str | Path, target_users: Sequence[str]) -> dict[str, set[str]]:
    truth: dict[str, set[str]] = defaultdict(set)
    for row in _csv_rows(path, ("user_id", "item_id")):
        user, item = _identifier(row["user_id"], "user_id"), _identifier(row["item_id"], "item_id")
        if item in truth[user]:
            raise ValueError(f"Duplicate truth pair: {user}, {item}")
        truth[user].add(item)
    if set(truth) != set(target_users):
        raise ValueError("Truth and target user sets differ")
    return dict(truth)


def recall_at_20(predictions: dict[str, list[str]], truth: dict[str, set[str]]) -> dict:
    """Pooled competition Recall@20: total hits / sum(min(20, |GT_u|))."""
    if set(predictions) != set(truth):
        raise ValueError("Prediction and truth user sets differ")
    hits = sum(len(set(items[:TOP_K]) & truth[user]) for user, items in predictions.items())
    denominator = sum(min(TOP_K, len(items)) for items in truth.values())
    if not denominator:
        raise ValueError("No truth items")
    return {"recall_at_20": hits / denominator, "hits": hits, "denominator": denominator, "users": len(truth)}


def normalized_graph(data: InteractionData, device):
    """D^-1/2 A D^-1/2 for the undirected user-item graph."""
    if torch is None:
        raise RuntimeError("PyTorch is required to build the graph")
    u = torch.tensor(data.positive_users, dtype=torch.long)
    i = torch.tensor(data.positive_items, dtype=torch.long) + data.num_users
    rows = torch.cat((u, i))
    cols = torch.cat((i, u))
    degree = torch.bincount(rows, minlength=data.num_users + data.num_items).float()
    values = (degree[rows] * degree[cols]).rsqrt()
    indices = torch.stack((rows, cols)).to(device)
    return torch.sparse_coo_tensor(
        indices, values.to(device),
        (data.num_users + data.num_items, data.num_users + data.num_items),
        device=device,
    ).coalesce()


def sparse_edge_dropout(graph, probability: float):
    if probability <= 0:
        return graph
    if not 0 <= probability < 1:
        raise ValueError("Edge dropout must be in [0, 1)")
    keep = torch.rand(graph._nnz(), device=graph.device) >= probability
    return torch.sparse_coo_tensor(
        graph.indices()[:, keep], graph.values()[keep] / (1 - probability),
        graph.shape, device=graph.device,
    ).coalesce()


class LightGCN(_BaseModel):
    def __init__(self, n_users: int, n_items: int, dim: int, layers: int):
        if nn is None:
            raise RuntimeError("PyTorch is required to train LightGCN")
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, dim)
        self.item_embedding = nn.Embedding(n_items, dim)
        self.layers = layers
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def forward(self, graph):
        current = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        layer_outputs = [current]
        for _ in range(self.layers):
            current = torch.sparse.mm(graph, current)
            layer_outputs.append(current)
        final = torch.stack(layer_outputs, dim=0).mean(dim=0)
        return final[: self.user_embedding.num_embeddings], final[self.user_embedding.num_embeddings :]


class NGCF(_BaseModel):
    def __init__(self, n_users: int, n_items: int, dim: int, layers: int, message_dropout: float):
        if nn is None:
            raise RuntimeError("PyTorch is required to train NGCF")
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, dim)
        self.item_embedding = nn.Embedding(n_items, dim)
        self.sum_linears = nn.ModuleList(nn.Linear(dim, dim) for _ in range(layers))
        self.bi_linears = nn.ModuleList(nn.Linear(dim, dim) for _ in range(layers))
        self.message_dropout = message_dropout
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        for layer in (*self.sum_linears, *self.bi_linears):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, graph):
        current = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        layer_outputs = [current]
        for sum_linear, bi_linear in zip(self.sum_linears, self.bi_linears):
            neighbors = torch.sparse.mm(graph, current)
            current = F.leaky_relu(sum_linear(current + neighbors) + bi_linear(current * neighbors), negative_slope=0.2)
            current = F.dropout(current, p=self.message_dropout, training=self.training)
            current = F.normalize(current, p=2, dim=1)
            layer_outputs.append(current)
        final = torch.cat(layer_outputs, dim=1)
        return final[: self.user_embedding.num_embeddings], final[self.user_embedding.num_embeddings :]


def sample_triples(data: InteractionData, count: int, rng: random.Random, device):
    """Sample one uniformly random unseen item per sampled positive edge."""
    if count <= 0:
        raise ValueError("samples-per-epoch must be positive")
    edge_indices = [rng.randrange(data.num_edges) for _ in range(count)]
    users = [data.positive_users[index] for index in edge_indices]
    positive = [data.positive_items[index] for index in edge_indices]
    negative = [rng.randrange(data.num_items) for _ in range(count)]
    for position, (user, item) in enumerate(zip(users, negative)):
        if item in data.seen_items[user]:
            while True:
                candidate = rng.randrange(data.num_items)
                if candidate not in data.seen_items[user]:
                    negative[position] = candidate
                    break
    return (
        torch.tensor(users, dtype=torch.long, device=device),
        torch.tensor(positive, dtype=torch.long, device=device),
        torch.tensor(negative, dtype=torch.long, device=device),
    )


@torch.no_grad() if torch is not None else (lambda function: function)
def predict_top20(model, graph, data: InteractionData, user_batch_size: int) -> dict[str, list[str]]:
    """Score all known items in bounded user batches and mask fit interactions."""
    if user_batch_size <= 0:
        raise ValueError("eval-user-batch-size must be positive")
    model.eval()
    user_vectors, item_vectors = model(graph)
    item_matrix = item_vectors.T.contiguous()
    predictions: dict[str, list[str]] = {}
    for start in range(0, len(data.target_users), user_batch_size):
        batch = data.target_users[start : start + user_batch_size]
        user_indices = torch.tensor([data.user_index[user] for user in batch], dtype=torch.long, device=graph.device)
        scores = user_vectors[user_indices] @ item_matrix
        masked_rows: list[int] = []
        masked_columns: list[int] = []
        for local_row, user_index in enumerate(user_indices.tolist()):
            for item_index in data.seen_items[user_index]:
                masked_rows.append(local_row)
                masked_columns.append(item_index)
        if masked_rows:
            row_index = torch.tensor(masked_rows, dtype=torch.long, device=graph.device)
            column_index = torch.tensor(masked_columns, dtype=torch.long, device=graph.device)
            scores[row_index, column_index] = -float("inf")
        top = torch.topk(scores, k=TOP_K, dim=1).indices.cpu().tolist()
        for user_id, item_indices in zip(batch, top):
            predictions[user_id] = [data.item_ids[index] for index in item_indices]
    return predictions


def write_predictions(path: str | Path, users: Sequence[str], predictions: dict[str, list[str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("user_id", "item_id"))
        for user in users:
            writer.writerow((user, " ".join(predictions[user])))


def _device(name: str):
    if torch is None:
        raise RuntimeError("PyTorch is not installed. Use Kaggle GPU Notebook or install torch.")
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def _training_loss(model, user_vectors, item_vectors, sampled, regularization: float):
    users, positives, negatives = sampled
    positive_scores = (user_vectors[users] * item_vectors[positives]).sum(dim=1)
    negative_scores = (user_vectors[users] * item_vectors[negatives]).sum(dim=1)
    bpr = F.softplus(negative_scores - positive_scores).mean()
    raw_u = model.user_embedding(users)
    raw_p = model.item_embedding(positives)
    raw_n = model.item_embedding(negatives)
    reg = (raw_u.square().sum(dim=1) + raw_p.square().sum(dim=1) + raw_n.square().sum(dim=1)).mean() / 2
    return bpr + regularization * reg, bpr, reg


def run(args: argparse.Namespace) -> dict:
    device = _device(args.device)
    if args.layers < 1 or args.embedding_dim < 1 or args.epochs < 1:
        raise ValueError("layers, embedding-dim and epochs must be positive")
    if not 0 <= args.edge_dropout < 1 or not 0 <= args.message_dropout < 1:
        raise ValueError("Dropout probabilities must be in [0, 1)")
    if args.reg < 0 or args.lr <= 0:
        raise ValueError("reg must be nonnegative and lr positive")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)
    data = load_data(args.fit, args.users)
    truth = load_truth(args.truth, data.target_users) if args.truth else None
    graph = normalized_graph(data, device)
    if args.model == "lightgcn":
        model = LightGCN(data.num_users, data.num_items, args.embedding_dim, args.layers)
    else:
        model = NGCF(data.num_users, data.num_items, args.embedding_dim, args.layers, args.message_dropout)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(json.dumps({
        "model": args.model, "device": str(device), "users": data.num_users,
        "items": data.num_items, "fit_edges": data.num_edges,
        "embedding_dim": args.embedding_dim, "layers": args.layers,
        "samples_per_epoch": args.samples_per_epoch,
    }), flush=True)

    best_score = -math.inf
    best_state = None
    best_epoch = None
    stale_evaluations = 0
    last_loss = None
    trained_epochs = 0
    for epoch in range(1, args.epochs + 1):
        trained_epochs = epoch
        start = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sampled = sample_triples(data, args.samples_per_epoch, rng, device)
        epoch_graph = sparse_edge_dropout(graph, args.edge_dropout) if args.model == "ngcf" else graph
        user_vectors, item_vectors = model(epoch_graph)
        loss, bpr, reg = _training_loss(model, user_vectors, item_vectors, sampled, args.reg)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        record = {
            "epoch": epoch, "loss": round(last_loss, 6),
            "bpr": round(float(bpr.detach().cpu()), 6),
            "regularization": round(float(reg.detach().cpu()), 6),
            "seconds": round(time.monotonic() - start, 2),
        }
        del sampled, user_vectors, item_vectors, loss, bpr, reg
        should_evaluate = truth is not None and (epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs)
        if should_evaluate:
            predictions = predict_top20(model, graph, data, args.eval_user_batch_size)
            metric = recall_at_20(predictions, truth)
            record.update(metric)
            if metric["recall_at_20"] > best_score:
                best_score = metric["recall_at_20"]
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                stale_evaluations = 0
                record["best"] = True
            else:
                stale_evaluations += 1
                record["best"] = False
        print(json.dumps(record), flush=True)
        if should_evaluate and args.patience > 0 and stale_evaluations >= args.patience:
            print(json.dumps({"early_stop_epoch": epoch}), flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    predictions = predict_top20(model, graph, data, args.eval_user_batch_size)
    write_predictions(args.out, data.target_users, predictions)
    result = {
        "model": args.model, "device": str(device), "fit_edges": data.num_edges,
        "users": len(data.target_users), "output": str(args.out), "last_loss": last_loss,
        "trained_epochs": trained_epochs, "best_epoch": best_epoch,
        "epochs_requested": args.epochs, "seed": args.seed,
        "embedding_dim": args.embedding_dim, "layers": args.layers,
        "samples_per_epoch": args.samples_per_epoch,
        "learning_rate": args.lr, "regularization": args.reg,
        "edge_dropout": args.edge_dropout if args.model == "ngcf" else 0.0,
        "message_dropout": args.message_dropout if args.model == "ngcf" else 0.0,
    }
    if truth is not None:
        result.update(recall_at_20(predictions, truth))
    print(json.dumps({"final": result}), flush=True)
    return result


def main(argv: Sequence[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="LightGCN / NGCF on the shared Amazon Top-20 split")
    parser.add_argument("--model", choices=("lightgcn", "ngcf"), required=True)
    parser.add_argument("--fit", required=True, help="val_train.csv or full train.csv")
    parser.add_argument("--users", required=True, help="val_users.csv or test.csv")
    parser.add_argument("--truth", help="val_truth.csv; omit for final test predictions")
    parser.add_argument("--out", required=True, help="Prediction CSV")
    parser.add_argument("--metrics-out", help="Optional JSON file for run settings and results")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--samples-per-epoch", type=int, default=100000)
    parser.add_argument("--eval-user-batch-size", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=4, help="Validation evaluations without improvement; 0 disables")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--reg", type=float, default=0.0001)
    parser.add_argument("--edge-dropout", type=float, default=0.0, help="Optional NGCF edge dropout")
    parser.add_argument("--message-dropout", type=float, default=0.1, help="NGCF only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0")
    args = parser.parse_args(argv)
    if args.eval_every < 1 or args.patience < 0:
        parser.error("eval-every must be positive and patience nonnegative")
    result = run(args)
    if args.metrics_out:
        Path(args.metrics_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
