"""Training-only corruption and explicitly labelled SSL adaptations."""

import torch
from torch.nn import functional as F
from torch_geometric.data import Batch

from .models import masked_infonce


def corrupt_files(files, rate=0.1):
    result = []
    for file in files:
        copies = []
        for graph in file:
            copy = graph.clone()
            mask = torch.rand(len(copy.token), device=copy.token.device) < rate
            copy.token[mask] = 0
            copies.append(copy)
        result.append(copies)
    return result


def graphcl_alignment(model, files, temperature, rate=0.1):
    first = model.code_projection(model.code(corrupt_files(files, rate)))
    second = model.code_projection(model.code(corrupt_files(files, rate)))
    return masked_infonce(
        first, second, torch.ones((len(files), len(files)), dtype=torch.bool), temperature
    )[0]


def pretrain(model, files, xt, edges, allowed, cfg, optimizer, generator):
    """Returns a history; every tensor/sample is supplied from training only."""
    history = []
    kind = cfg.get("pretrain_kind")
    if not kind:
        return history
    device = xt.device
    for epoch in range(cfg.get("pretrain_epochs", 50)):
        model.train()
        values = []
        for batch in torch.randperm(len(files), generator=generator).split(cfg["batch_size"]):
            indices = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            if kind == "contrastive":
                code = model.code_projection(model.code([files[i] for i in batch.tolist()]))
                metric = model.metric_projection(model.metric(xt, edges)[indices])
                loss, _ = masked_infonce(
                    code, metric, allowed[indices][:, indices], cfg["temperature"]
                )
            elif kind == "graphmae_adaptation":
                graphs = Batch.from_data_list([g for i in batch.tolist() for g in files[i]]).to(
                    device
                )
                target = graphs.token.clone()
                mask = torch.rand(len(target), device=device) < 0.1
                if not mask.any():
                    mask[0] = True
                graphs.token[mask] = 0
                decoded = model.code_decoder(model.code.node_states(graphs))
                code_loss = F.cross_entropy(decoded[mask], target[mask])
                metric_mask = torch.rand_like(xt) < 0.1
                if not metric_mask.any():
                    metric_mask[0, 0] = True
                corrupted = xt.masked_fill(metric_mask, 0)
                recovered = model.metric_decoder(model.metric(corrupted, edges))
                metric_loss = F.mse_loss(recovered[metric_mask], xt[metric_mask])
                loss = code_loss + metric_loss
            else:
                raise ValueError(f"Unknown pretraining kind: {kind}")
            loss.backward()
            optimizer.step()
            values.append(float(loss.detach()))
        history.append({"epoch": epoch + 1, "loss": sum(values) / len(values), "kind": kind})
    return history
