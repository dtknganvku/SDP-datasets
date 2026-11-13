"""Pinned external encoder file-window adaptations.

These are file-level adaptations, not exact reproductions of published DP methods.
GraphCodeBERT is intentionally excluded until its data-flow input is implemented.
"""

import re
from importlib.resources import files

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

from .io import read_json


def checkpoint_spec(name):
    spec = read_json(files("dgcl_dp.configs") / "pretrained.json")["models"][name]
    if not re.fullmatch(r"[0-9a-f]{40}", spec["revision"]):
        raise ValueError("Pretrained checkpoint requires a full immutable commit revision")
    if spec.get("requires_data_flow"):
        raise ValueError("GraphCodeBERT requires an audited data-flow input adapter")
    return spec


class PretrainedVocabulary:
    def __init__(self, name, tokenizer=None):
        self.name, self.spec = name, checkpoint_spec(name)
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                self.spec["repository"], revision=self.spec["revision"], trust_remote_code=False
            )
        self.tokenizer = tokenizer
        self.tokens = ["<external-checkpoint-tokenizer>"]

    def graphs(self, parsed):
        tokenizer = self.tokenizer
        values = tokenizer.encode(" ".join(parsed.lexical), add_special_tokens=False)
        window = 508 if self.name == "unixcoder" else 510
        chunks = [values[i : i + window] for i in range(0, len(values), window)] or [[]]
        if self.name == "unixcoder":
            marker = tokenizer.convert_tokens_to_ids("<encoder-only>")
            if marker == tokenizer.unk_token_id:
                raise ValueError("UniXcoder encoder-only marker missing")
            chunks = [
                [tokenizer.cls_token_id, marker, tokenizer.sep_token_id, *c, tokenizer.sep_token_id]
                for c in chunks
            ]
        else:
            chunks = [tokenizer.build_inputs_with_special_tokens(c) for c in chunks]
        return [Data(token=torch.tensor(c, dtype=torch.long)) for c in chunks]


class PretrainedEncoder(nn.Module):
    def __init__(self, vocabulary, hidden=128, dropout=0.3, backbone=None):
        super().__init__()
        spec = vocabulary.spec
        if backbone is None:
            from transformers import AutoModel

            backbone = AutoModel.from_pretrained(
                spec["repository"], revision=spec["revision"], trust_remote_code=False
            )
        self.backbone = backbone
        if hasattr(self.backbone, "get_encoder"):
            self.backbone = self.backbone.get_encoder()
        width = getattr(self.backbone.config, "hidden_size", None) or self.backbone.config.d_model
        self.projection = nn.Linear(width, hidden)
        self.dropout = nn.Dropout(dropout)
        self.pad_token_id = vocabulary.tokenizer.pad_token_id
        if self.pad_token_id is None:
            raise ValueError("Checkpoint tokenizer must declare a padding token")

    def forward(self, files):
        device = self.projection.weight.device
        sequences = [g.token.to(device) for f in files for g in f]
        lengths = torch.tensor([len(s) for s in sequences], device=device)
        tokens = pad_sequence(sequences, batch_first=True, padding_value=self.pad_token_id)
        mask = torch.arange(tokens.shape[1], device=device)[None] < lengths[:, None]
        states = self.backbone(input_ids=tokens, attention_mask=mask.long()).last_hidden_state
        mean = (states * mask[:, :, None]).sum(1) / lengths[:, None]
        file_ids = torch.tensor([i for i, f in enumerate(files) for _ in f], device=device)
        return self.dropout(self.projection(global_mean_pool(mean, file_ids)))
