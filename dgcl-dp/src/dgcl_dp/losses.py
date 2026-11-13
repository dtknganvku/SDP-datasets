"""Controlled supervised and ProGCL-style cross-view loss adaptations."""

import numpy as np
import torch
from scipy.special import betaln, logsumexp
from torch.nn import functional as F


def supcon(code, metric, labels, temperature=0.1, balanced=False):
    logits = F.normalize(code, dim=1) @ F.normalize(metric, dim=1).T / temperature
    positives = labels[:, None] == labels[None, :]
    weights = torch.ones(len(labels), device=code.device)
    if balanced:
        weights = torch.stack([1 / (labels == y).sum() for y in labels])
    losses = []
    for scores in (logits, logits.T):
        loss = scores.logsumexp(1) - (scores * positives).sum(1) / positives.sum(1)
        losses.append((loss * weights).sum() / weights.sum())
    return sum(losses) / 2


def _true_negative_probability(similarity):
    """Two beta components; lower-mean component estimates true negatives.

    This refits on each training minibatch. It is an adaptation of ProGCL-weight,
    not the author's fixed-epoch global estimator or within-view objective.
    """
    values = np.clip(similarity.detach().cpu().double().numpy(), 1e-5, 1 - 1e-5)
    off_diagonal = ~np.eye(len(values), dtype=bool)
    x = values[off_diagonal]
    if len(x) < 4 or np.std(x) < 1e-5:
        return torch.ones_like(similarity)
    alpha, beta, prior = np.array([1.0, 2.0]), np.array([2.0, 1.0]), np.array([0.5, 0.5])
    for _ in range(10):
        logp = (
            (alpha[:, None] - 1) * np.log(x)
            + (beta[:, None] - 1) * np.log1p(-x)
            - betaln(alpha, beta)[:, None]
            + np.log(prior)[:, None]
        )
        r = np.exp(logp - logsumexp(logp, axis=0))
        mass = np.maximum(r.sum(1), 1e-8)
        mean = np.clip((r * x).sum(1) / mass, 1e-4, 1 - 1e-4)
        variance = np.maximum((r * (x - mean[:, None]) ** 2).sum(1) / mass, 1e-8)
        concentration = np.clip(mean * (1 - mean) / variance - 1, 0.02, 1e4)
        alpha, beta = mean * concentration, (1 - mean) * concentration
        prior = np.clip(mass / len(x), 1e-5, 1.0)
        prior /= prior.sum()
    low = np.argmin(alpha / (alpha + beta))
    flat = values.ravel()
    logp = (
        (alpha[:, None] - 1) * np.log(flat)
        + (beta[:, None] - 1) * np.log1p(-flat)
        - betaln(alpha, beta)[:, None]
        + np.log(prior)[:, None]
    )
    posterior = np.exp(logp[low] - logsumexp(logp, axis=0)).reshape(values.shape)
    weight = np.maximum(posterior * values, 1e-8)
    return torch.tensor(weight, device=similarity.device, dtype=similarity.dtype)


def progcl_weighted(code, metric, temperature=0.1):
    similarity = F.normalize(code, dim=1) @ F.normalize(metric, dim=1).T
    weights = _true_negative_probability((similarity + 1) / 2)
    diagonal = torch.eye(len(code), dtype=torch.bool, device=code.device)
    logits, losses = similarity / temperature, []
    if len(code) == 1:
        return logits.sum() * 0
    for scores, weight in ((logits, weights), (logits.T, weights.T)):
        row_mean = weight.masked_fill(diagonal, 0).sum(1) / (len(code) - 1)
        log_weights = (weight / row_mean[:, None]).log().masked_fill(diagonal, 0)
        losses.append((-scores.diagonal() + (scores + log_weights).logsumexp(1)).mean())
    return sum(losses) / 2
