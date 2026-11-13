"""Training/validation operations, deliberately independent of test outcomes."""

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC
from scipy.special import expit
from torch.nn import functional as F

from .evaluation import choose_threshold
from .features import MetricTransform
from .graphs import matched_random_masks, negative_mask, query_graph, training_graph
from .io import digest
from .losses import progcl_weighted, supcon
from .models import DualGraph, masked_infonce
from .runtime import deterministic
from .source import Vocabulary
from .sequences import TokenEncoder, TokenVocabulary
from .ssl import graphcl_alignment, pretrain


CLASSICAL = {"majority", "loc", "inverse_loc", "lr", "rf", "svm", "hgb", "dbn"}


@dataclass
class Fitted:
    model: object
    transform: MetricTransform
    vocabulary: Vocabulary
    training_ids: list
    training_x: np.ndarray
    edges: torch.Tensor
    cache: tuple
    threshold: float
    metadata: dict
    config: dict


def _records(features, ids):
    return [features[i] for i in ids]


def _pr_auc(y, p):
    if len(set(y)) < 2:
        raise ValueError("Validation PR-AUC undefined for a single-class partition")
    return float(average_precision_score(y, p))


def predict(fitted, ids, features, parsed, samples):
    """Receives no labels; every query independently aggregates frozen references."""
    cfg, model = fitted.config, fitted.model
    if cfg["name"] == "inverse_loc":
        return np.asarray([1 / max(1, samples[i]["loc"]) for i in ids], dtype=float)
    x = fitted.transform.transform(_records(features, ids))
    if cfg["name"] in CLASSICAL:
        if cfg["name"] == "loc":
            x = np.log1p([[samples[i]["loc"]] for i in ids])
        probability = model.predict_proba(x)
        return (
            probability[:, list(model.classes_).index(1)]
            if 1 in model.classes_
            else np.zeros(len(x))
        )
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        metric = torch.zeros((len(ids), cfg["hidden"]), device=device)
        if cfg.get("view", "both") != "code":
            incoming = query_graph(fitted.training_x, x, fitted.training_ids, cfg["k"]).to(device)
            if cfg.get("graph_control") == "none":
                incoming = incoming[:, :0]
            metric = model.metric.query(torch.tensor(x, device=device), incoming, fitted.cache)
        probabilities = []
        for start in range(0, len(ids), cfg["batch_size"]):
            batch_ids = ids[start : start + cfg["batch_size"]]
            code = torch.zeros((len(batch_ids), cfg["hidden"]), device=device)
            if cfg.get("view", "both") != "metric":
                code = model.code([fitted.vocabulary.graphs(parsed[i]) for i in batch_ids])
            logits = model.classify(code, metric[start : start + len(batch_ids)])
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
        return np.asarray(probabilities)


def fit(
    config,
    train_ids,
    validation_ids,
    features,
    parsed,
    samples,
    train_labels,
    validation_labels,
    schema,
    seed=0,
):
    deterministic(seed)
    cfg = dict(config)
    train_ids, validation_ids = sorted(train_ids), sorted(validation_ids)
    if len(train_ids) != len(set(train_ids)) or set(train_ids) & set(validation_ids):
        raise ValueError("Training/validation IDs must be unique and disjoint")
    y = np.asarray([train_labels[i] for i in train_ids], dtype=float)
    if len(set(y)) < 2 and cfg["name"] not in ("majority", "inverse_loc"):
        raise ValueError("Single-class training is ineligible; use majority baseline")
    vy = [validation_labels[i] for i in validation_ids]
    if len(set(vy)) < 2:
        raise ValueError("Single-class validation is ineligible for PR-AUC model selection")
    transform = MetricTransform(schema).fit(_records(features, train_ids))
    x = transform.transform(_records(features, train_ids))
    vocabulary_cls = TokenVocabulary if cfg.get("sequence_kind") else Vocabulary
    vocab = vocabulary_cls.fit([parsed[i] for i in train_ids])
    if cfg.get("pretrained_key"):
        from .pretrained import PretrainedVocabulary

        vocab = PretrainedVocabulary(cfg["pretrained_key"])
    edges = training_graph(x, train_ids, cfg["k"], cfg.get("graph_control", "knn"), seed)
    metadata = {
        "seed": seed,
        "training_ids": train_ids,
        "preprocessing": transform.state(),
        "vocabulary": vocab.tokens,
        "config_hash": digest(cfg),
        "epochs": 0,
        "parameters": 0,
        "alignment": [],
    }
    fitted = Fitted(None, transform, vocab, train_ids, x, edges, (), 0.5, metadata, cfg)
    start = time.perf_counter()
    if cfg["name"] in CLASSICAL:
        models = {
            "majority": lambda: DummyClassifier(strategy="most_frequent"),
            "loc": lambda: LogisticRegression(
                class_weight="balanced", random_state=seed, max_iter=2000
            ),
            "lr": lambda: LogisticRegression(
                C=cfg.get("C", 1.0), class_weight="balanced", random_state=seed, max_iter=2000
            ),
            "rf": lambda: RandomForestClassifier(
                n_estimators=cfg.get("n_estimators", 300),
                max_depth=cfg.get("max_depth"),
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            ),
            "svm": lambda: SVC(
                C=cfg.get("C", 1.0), class_weight="balanced", probability=True, random_state=seed
            ),
            "hgb": lambda: HistGradientBoostingClassifier(
                learning_rate=cfg.get("lr", 0.1), class_weight="balanced", random_state=seed
            ),
            "inverse_loc": lambda: DummyClassifier(strategy="prior"),
            "dbn": lambda: make_pipeline(
                FunctionTransformer(expit),
                BernoulliRBM(
                    n_components=cfg["hidden"],
                    learning_rate=cfg.get("rbm_lr", 0.01),
                    n_iter=cfg.get("rbm_epochs", 20),
                    batch_size=10,
                    random_state=seed,
                ),
                BernoulliRBM(
                    n_components=cfg["hidden"],
                    learning_rate=cfg.get("rbm_lr", 0.01),
                    n_iter=cfg.get("rbm_epochs", 20),
                    batch_size=10,
                    random_state=seed + 1,
                ),
                LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
            ),
        }
        model = models[cfg["name"]]()
        fit_x = np.log1p([[samples[i]["loc"]] for i in train_ids]) if cfg["name"] == "loc" else x
        model.fit(fit_x, y)
        fitted.model = model
    else:
        if len(set(y)) < 2:
            raise ValueError("Single-class neural training is ineligible; use majority baseline")
        device = torch.device(cfg.get("device", "cpu"))
        model = DualGraph(
            len(vocab.tokens),
            x.shape[1],
            cfg["hidden"],
            cfg["projection"],
            cfg["dropout"],
            cfg.get("metric_graph", True),
            cfg.get("view", "both"),
            cfg.get("code_kind", "sage"),
            cfg.get("fusion", "concat"),
        ).to(device)
        if cfg.get("sequence_kind"):
            model.code = TokenEncoder(
                len(vocab.tokens), cfg["hidden"], cfg["dropout"], cfg["sequence_kind"]
            ).to(device)
        if cfg.get("pretrained_key"):
            from .pretrained import PretrainedEncoder

            model.code = PretrainedEncoder(vocab, cfg["hidden"], cfg["dropout"]).to(device)
            metadata["external_checkpoint"] = vocab.spec
            metadata["pretraining_exposure"] = "Unresolved external-data condition"
        fitted.model = model
        if not cfg["contrastive_weight"] and cfg.get("pretrain_kind") != "contrastive":
            model.code_projection = None
            model.metric_projection = None
        if cfg.get("pretrain_kind") == "graphmae_adaptation":
            model.code_decoder = torch.nn.Linear(cfg["hidden"], len(vocab.tokens)).to(device)
            model.metric_decoder = torch.nn.Linear(cfg["hidden"], x.shape[1]).to(device)
        metadata["parameters"] = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
        xt = torch.tensor(x, device=device)
        yt = torch.tensor(y, dtype=torch.float32, device=device)
        edges = edges.to(device)
        mask_mode = "overlap" if cfg["negative_mode"] == "random" else cfg["negative_mode"]
        allowed = negative_mask(
            edges, [samples[i]["clone_group"] for i in train_ids], mask_mode, seed=seed
        ).to(device)
        masking_rng = np.random.default_rng(seed)
        files = [vocab.graphs(parsed[i]) for i in train_ids]
        generator = torch.Generator().manual_seed(seed)
        metadata["pretraining"] = pretrain(
            model, files, xt, edges, allowed, cfg, optimizer, generator
        )
        best, best_state, wait = -float("inf"), None, 0
        for epoch in range(cfg["max_epochs"]):
            model.train()
            diagnostics = []
            for batch in torch.randperm(len(train_ids), generator=generator).split(
                cfg["batch_size"]
            ):
                indices = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                # Recompute the full metric graph for each optimizer step; no stale gradients.
                metric = torch.zeros((len(batch), cfg["hidden"]), device=device)
                code = torch.zeros_like(metric)
                if cfg.get("view", "both") != "code":
                    metric = model.metric(xt, edges)[indices]
                if cfg.get("view", "both") != "metric":
                    code = model.code([files[i] for i in batch.tolist()])
                logits = model.classify(code, metric)
                per_sample = F.binary_cross_entropy_with_logits(
                    logits, yt[indices], pos_weight=(1 - yt.mean()) / yt.mean(), reduction="none"
                )
                if cfg.get("classification_loss") == "focal":
                    pt = torch.where(
                        yt[indices] == 1, torch.sigmoid(logits), torch.sigmoid(-logits)
                    )
                    per_sample = per_sample * (1 - pt).square()
                cls = per_sample.mean()
                if cfg["contrastive_weight"] and cfg.get("view", "both") == "both":
                    submask = allowed[indices][:, indices]
                    if cfg["negative_mode"] == "random":
                        submask = matched_random_masks(submask, masking_rng)
                    alignment, diagnostic = masked_infonce(
                        model.code_projection(code),
                        model.metric_projection(metric),
                        submask,
                        cfg["temperature"],
                    )
                    loss_kind = cfg.get("contrastive_loss", "infonce")
                    if loss_kind in ("supcon", "balanced_supcon"):
                        alignment = supcon(
                            model.code_projection(code),
                            model.metric_projection(metric),
                            yt[indices],
                            cfg["temperature"],
                            loss_kind == "balanced_supcon",
                        )
                    elif loss_kind == "progcl_weight_adaptation":
                        alignment = progcl_weighted(
                            model.code_projection(code),
                            model.metric_projection(metric),
                            cfg["temperature"],
                        )
                    if loss_kind == "graphcl_adaptation":
                        alignment = graphcl_alignment(
                            model, [files[i] for i in batch.tolist()], cfg["temperature"]
                        )
                    if loss_kind != "infonce":
                        diagnostic = {
                            "skipped_fraction": float(len(batch) < 2),
                            "effective_negatives": float(max(0, len(batch) - 1)),
                        }
                    loss = cls + cfg["contrastive_weight"] * alignment
                    diagnostics.append(diagnostic)
                else:
                    loss = cls
                loss.backward()
                optimizer.step()
            model.eval()
            fitted.cache = (
                model.metric.cache(xt, edges) if cfg.get("view", "both") != "code" else ()
            )
            vp = predict(fitted, validation_ids, features, parsed, samples)
            score = _pr_auc(vy, vp)
            if score > best:
                best, best_state, wait = score, copy.deepcopy(model.state_dict()), 0
                metadata["best_epoch"] = epoch + 1
            else:
                wait += 1
            metadata["epochs"] = epoch + 1
            metadata["alignment"].append(
                {
                    "epoch": epoch + 1,
                    "validation_pr_auc": score,
                    **{
                        key: float(np.mean([d[key] for d in diagnostics]))
                        for key in ("skipped_fraction", "effective_negatives")
                        if diagnostics
                    },
                }
            )
            if wait >= cfg["patience"]:
                break
        if best_state is None:
            raise ValueError("Training produced no checkpoint")
        model.load_state_dict(best_state)
        model.eval()
        fitted.cache = model.metric.cache(xt, edges) if cfg.get("view", "both") != "code" else ()
    vp = predict(fitted, validation_ids, features, parsed, samples)
    fitted.threshold = choose_threshold(vy, vp)
    metadata.update(
        {
            "validation_pr_auc": _pr_auc(vy, vp),
            "threshold": fitted.threshold,
            "training_seconds": time.perf_counter() - start,
            "peak_vram_bytes": torch.cuda.max_memory_allocated()
            if torch.cuda.is_available()
            else 0,
        }
    )
    return fitted
