import copy

import numpy as np
import pytest
import torch

from dgcl_dp.graphs import matched_random_masks, negative_mask, query_graph, training_graph
from dgcl_dp.models import MetricEncoder, masked_infonce
from dgcl_dp.training import fit, predict


def test_frozen_metric_query_contract():
    torch.manual_seed(4)
    encoder = MetricEncoder(3, hidden=8, dropout=0.3).eval()
    x = torch.randn(6, 3)
    ids = [f"train-{i}" for i in range(6)]
    edges = training_graph(x.numpy(), ids, 2)
    cache = encoder.cache(x, edges)
    before = [v.clone() for v in cache]
    queries = torch.randn(4, 3)

    def infer(q):
        return encoder.query(q, query_graph(x.numpy(), q.numpy(), ids, 2), cache).detach()

    alone = infer(queries[:1])[0]
    torch.testing.assert_close(alone, infer(queries)[0], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(alone, infer(queries[[3, 0, 1]])[1], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(alone, infer(queries[[0, 2]])[0], rtol=1e-6, atol=1e-7)
    for a, b in zip(before, cache):
        assert torch.equal(a, b)
    assert edges.max() < len(x)


def test_fitting_has_no_test_label_dependency(task, config):
    args = task["fit_args"]
    cfg = {**config["model"], "name": "dgcl", "max_epochs": 2}
    first = fit(cfg, **args, seed=7)
    for i in task["split"]["test"]:
        task["labels"][i]["label"] = 1 - task["labels"][i]["label"]
    second = fit(cfg, **args, seed=7)
    assert first.transform.state() == second.transform.state()
    assert first.vocabulary.tokens == second.vocabulary.tokens
    assert first.threshold == second.threshold
    for key, value in first.model.state_dict().items():
        assert torch.equal(value, second.model.state_dict()[key])
    for a, b in zip(first.cache, second.cache):
        assert torch.equal(a, b)
    ids = task["split"]["test"]
    before = copy.deepcopy(first.model.state_dict())
    cached = [v.clone() for v in first.cache]
    all_p = predict(first, ids, task["features"], task["parsed"], task["samples"])
    single = predict(first, ids[:1], task["features"], task["parsed"], task["samples"])
    perm = predict(first, ids[::-1], task["features"], task["parsed"], task["samples"])
    np.testing.assert_allclose(all_p[:1], single, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(all_p, perm[::-1], rtol=1e-6, atol=1e-7)
    assert set(first.training_ids) == set(task["split"]["train"])
    for key, value in first.model.state_dict().items():
        assert torch.equal(value, before[key])
    assert all(torch.equal(a, b) for a, b in zip(cached, first.cache))


def test_overlap_clone_masks_and_no_negative_gradient():
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    mask = negative_mask(edges, ["a", "b", "c", "a"], mode="overlap")
    assert not mask[0, 1] and not mask[0, 3]
    assert torch.diagonal(mask).all()
    code, metric = torch.randn(4, 8, requires_grad=True), torch.randn(4, 8, requires_grad=True)
    loss, stats = masked_infonce(code, metric, torch.eye(4, dtype=torch.bool))
    assert loss.item() == 0
    assert stats["skipped_fraction"] == 1
    loss.backward()
    assert torch.isfinite(code.grad).all()
    assert torch.equal(code.grad, torch.zeros_like(code))


def test_matched_random_preserves_each_direction_batch_counts():
    mask = torch.tensor([[1, 0, 1, 0], [0, 1, 1, 1], [1, 1, 1, 0], [0, 1, 0, 1]], dtype=torch.bool)
    forward, reverse = matched_random_masks(mask, np.random.default_rng(10))
    assert torch.equal(forward.sum(1), mask.sum(1))
    assert torch.equal(reverse.sum(1), mask.T.sum(1))
    assert forward.diagonal().all() and reverse.diagonal().all()


def test_train_cache_requires_eval():
    with pytest.raises(ValueError, match="evaluation"):
        MetricEncoder(2).cache(torch.randn(2, 2), torch.empty((2, 0), dtype=torch.long))
