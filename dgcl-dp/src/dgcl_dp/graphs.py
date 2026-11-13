"""Deterministic training k-NN, incoming query edges, and receptive fields."""

import numpy as np
import torch


def nearest(reference, queries, k, reference_ids, exclude_self=False):
    if k < 1 or len(reference) != len(reference_ids) or len(set(reference_ids)) != len(reference):
        raise ValueError("Invalid k/reference IDs")
    result = []
    for q, x in enumerate(queries):
        distances = np.square(reference - x).sum(axis=1)
        candidates = [i for i in range(len(reference)) if not exclude_self or i != q]
        candidates.sort(key=lambda i: (float(distances[i]), reference_ids[i]))
        result.append(candidates[:k])
    return result


def training_graph(x, ids, k=5, mode="knn", seed=0):
    neighbours = nearest(x, x, k, ids, exclude_self=True)
    edges = {(j, i) for i, ns in enumerate(neighbours) for j in ns}
    edges |= {(j, i) for i, j in edges}
    if mode == "shuffled":
        permutation = np.random.default_rng(seed).permutation(len(x))
        edges = {(int(permutation[i]), int(permutation[j])) for i, j in edges}
    elif mode == "none":
        edges = set()
    elif mode != "knn":
        raise ValueError(f"Unsupported graph control: {mode}")
    return torch.tensor(sorted(edges), dtype=torch.long).reshape(-1, 2).T.contiguous()


def query_graph(reference, queries, reference_ids, k=5):
    edges = [
        (j, i) for i, ns in enumerate(nearest(reference, queries, k, reference_ids)) for j in ns
    ]
    return torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T.contiguous()


def receptive_fields(edge_index, n, layers=2):
    adjacent = [set() for _ in range(n)]
    for a, b in edge_index.T.tolist():
        adjacent[b].add(a)
    fields = [{i} for i in range(n)]
    for _ in range(layers):
        fields = [fields[i] | set().union(*(fields[j] for j in adjacent[i])) for i in range(n)]
    return adjacent, fields


def negative_mask(edge_index, clone_groups, mode="overlap", overlap=0.5, seed=0):
    """True entries are eligible denominator pairs, including the diagonal."""
    n = len(clone_groups)
    allowed = torch.ones((n, n), dtype=torch.bool)
    if mode == "all":
        return allowed
    adjacent, fields = receptive_fields(edge_index, n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            clone = clone_groups[i] == clone_groups[j]
            structural = j in adjacent[i] or (
                len(fields[i] & fields[j]) / len(fields[i] | fields[j]) >= overlap
            )
            if clone or (mode in ("overlap", "random") and structural):
                allowed[i, j] = False
    if mode == "random":
        rng = np.random.default_rng(seed)
        for i in range(n):
            count = int((~allowed[i]).sum())
            allowed[i] = True
            candidates = [j for j in range(n) if j != i]
            if count:
                allowed[i, rng.choice(candidates, count, replace=False)] = False
    elif mode not in ("clone", "overlap"):
        raise ValueError(f"Unsupported negative control: {mode}")
    return allowed


def matched_random_masks(allowed, rng):
    """Match each anchor's negative count in each direction within the actual batch."""
    masks = []
    for target in (allowed, allowed.T):
        mask = torch.ones_like(target)
        for i in range(len(target)):
            count = int((~target[i]).sum())
            candidates = [j for j in range(len(target)) if j != i]
            if count:
                mask[i, rng.choice(candidates, count, replace=False)] = False
        masks.append(mask)
    return tuple(masks)
