"""Dual encoders and strictly isolated bipartite GraphSAGE inference."""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_mean_pool


class CodeEncoder(nn.Module):
    def __init__(self, vocabulary_size, hidden=128, dropout=0.3, kind="sage"):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, hidden - 1)
        conv = {"sage": SAGEConv, "gcn": GCNConv, "gat": GATConv}[kind]
        self.layers = nn.ModuleList([conv(hidden, hidden), conv(hidden, hidden)])
        self.dropout = dropout

    def forward(self, files):
        device = self.embedding.weight.device
        graphs = [g for f in files for g in f]
        batch = Batch.from_data_list(graphs).to(device)
        x = self.node_states(batch)
        partitions = global_mean_pool(x, batch.batch)
        file_ids = torch.tensor([i for i, f in enumerate(files) for _ in f], device=device)
        return global_mean_pool(partitions, file_ids)

    def node_states(self, batch):
        x = torch.cat((self.embedding(batch.token), batch.order), dim=-1)
        for layer in self.layers:
            x = F.dropout(F.relu(layer(x, batch.edge_index)), self.dropout, self.training)
        return x


class MetricEncoder(nn.Module):
    def __init__(self, features, hidden=128, dropout=0.3, graph=True):
        super().__init__()
        self.graph = graph
        conv = SAGEConv if graph else nn.Linear
        self.layers = nn.ModuleList([conv(features, hidden), conv(hidden, hidden)])
        self.dropout = dropout

    def forward(self, x, edge_index, return_layers=False):
        states = [x]
        for layer in self.layers:
            x = layer(x, edge_index) if self.graph else layer(x)
            x = F.dropout(F.relu(x), self.dropout, self.training)
            states.append(x)
        return tuple(states) if return_layers else x

    @torch.no_grad()
    def cache(self, x, edge_index):
        if self.training:
            raise ValueError("Build training caches only in evaluation mode")
        return tuple(v.detach().clone() for v in self.forward(x, edge_index, return_layers=True))

    def query(self, x, incoming_edges, cache):
        if self.training:
            raise ValueError("Query inference requires evaluation mode")
        for i, layer in enumerate(self.layers):
            x = layer((cache[i], x), incoming_edges) if self.graph else layer(x)
            x = F.relu(x)
        return x


class DualGraph(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        features,
        hidden=128,
        projection=64,
        dropout=0.3,
        metric_graph=True,
        view="both",
        code_kind="sage",
        fusion="concat",
    ):
        super().__init__()
        if view not in ("both", "code", "metric") or fusion not in ("concat", "gate"):
            raise ValueError("Unknown view/fusion")
        self.view, self.fusion = view, fusion
        self.code = (
            CodeEncoder(vocabulary_size, hidden, dropout, code_kind) if view != "metric" else None
        )
        self.metric = (
            MetricEncoder(features, hidden, dropout, metric_graph) if view != "code" else None
        )
        self.code_projection = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, projection)
        )
        self.metric_projection = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, projection)
        )
        self.gate = nn.Linear(hidden * 2, hidden) if fusion == "gate" else None
        width = hidden * 2 if view == "both" and fusion == "concat" else hidden
        self.classifier = nn.Sequential(
            nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def classify(self, code, metric):
        if self.view == "code":
            x = code
        elif self.view == "metric":
            x = metric
        elif self.fusion == "gate":
            gate = torch.sigmoid(self.gate(torch.cat((code, metric), dim=1)))
            x = gate * code + (1 - gate) * metric
        else:
            x = torch.cat((code, metric), dim=1)
        return self.classifier(x).flatten()


def masked_infonce(code, metric, allowed, temperature=0.1):
    if temperature <= 0 or code.shape != metric.shape:
        raise ValueError("Invalid temperature/projection shapes")
    masks = allowed if isinstance(allowed, tuple) else (allowed, allowed.T)
    masks = tuple(mask.clone().to(code.device) for mask in masks)
    for mask in masks:
        mask.fill_diagonal_(True)
    logits = F.normalize(code, dim=1) @ F.normalize(metric, dim=1).T / temperature
    losses, skipped, counts = [], [], []
    for scores, mask in zip((logits, logits.T), masks):
        eligible = mask.sum(dim=1) > 1
        per_anchor = -scores.diagonal() + scores.masked_fill(~mask, -torch.inf).logsumexp(dim=1)
        losses.append(per_anchor[eligible].mean() if eligible.any() else scores.sum() * 0)
        skipped.append((~eligible).float().mean())
        counts.append((mask.sum(dim=1) - 1).float().mean())
    return sum(losses) / 2, {
        "skipped_fraction": float(torch.stack(skipped).mean()),
        "effective_negatives": float(torch.stack(counts).mean()),
    }
