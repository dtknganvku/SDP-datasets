import torch

from dgcl_dp.losses import progcl_weighted, supcon


def test_control_losses_are_finite_and_train_both_views():
    torch.manual_seed(0)
    for kind in ("supcon", "balanced", "progcl"):
        code = torch.randn(8, 4, requires_grad=True)
        metric = torch.randn(8, 4, requires_grad=True)
        labels = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1])
        loss = (
            progcl_weighted(code, metric)
            if kind == "progcl"
            else supcon(code, metric, labels, balanced=kind == "balanced")
        )
        loss.backward()
        assert (
            torch.isfinite(loss)
            and torch.isfinite(code.grad).all()
            and torch.isfinite(metric.grad).all()
        )
        assert code.grad.abs().sum() > 0 and metric.grad.abs().sum() > 0
