"""Training-only Java token windows and historical sequence encoders."""

from collections import Counter

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool


class TokenVocabulary:
    def __init__(self, tokens, window=512):
        self.tokens = ["<UNK>", "<PAD>"] + list(tokens)
        self.lookup = {t: i for i, t in enumerate(self.tokens)}
        self.window = window

    @classmethod
    def fit(cls, parsed, window=512, maximum=50000):
        counts = Counter(t for f in parsed for t in f.lexical)
        tokens = sorted(counts, key=lambda t: (-counts[t], t))[: maximum - 2]
        return cls(tokens, window)

    def graphs(self, parsed):
        values = [self.lookup.get(t, 0) for t in parsed.lexical] or [0]
        return [
            Data(token=torch.tensor(values[start : start + self.window], dtype=torch.long))
            for start in range(0, len(values), self.window)
        ]


class TokenEncoder(nn.Module):
    def __init__(self, vocabulary_size, hidden=128, dropout=0.3, kind="cnn"):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, hidden, padding_idx=1)
        self.kind, self.dropout = kind, dropout
        if kind == "cnn":
            self.convolutions = nn.ModuleList(
                [nn.Conv1d(hidden, hidden, k, padding=k // 2) for k in (3, 5, 7)]
            )
            self.output = nn.Linear(hidden * 3, hidden)
        elif kind == "bilstm":
            self.rnn = nn.LSTM(hidden, hidden // 2, batch_first=True, bidirectional=True)
        else:
            raise ValueError("Sequence encoder must be cnn or bilstm")

    def forward(self, files):
        device = self.embedding.weight.device
        sequences = [g.token.to(device) for f in files for g in f]
        lengths = torch.tensor([len(s) for s in sequences], device=device)
        padded = pad_sequence(sequences, batch_first=True, padding_value=1)
        x = self.embedding(padded)
        if self.kind == "cnn":
            mask = torch.arange(padded.shape[1], device=device)[None] >= lengths[:, None]
            pooled = [
                F.relu(conv(x.transpose(1, 2))).masked_fill(mask[:, None], -torch.inf).amax(2)
                for conv in self.convolutions
            ]
            representation = self.output(torch.cat(pooled, dim=1))
        else:
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (h, _) = self.rnn(packed)
            representation = torch.cat((h[-2], h[-1]), dim=1)
        file_ids = torch.tensor([i for i, f in enumerate(files) for _ in f], device=device)
        pooled = global_mean_pool(representation, file_ids)
        return F.dropout(pooled, self.dropout, self.training)
