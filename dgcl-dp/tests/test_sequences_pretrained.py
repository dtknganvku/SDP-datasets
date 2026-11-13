from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dgcl_dp.pretrained import PretrainedEncoder, PretrainedVocabulary, checkpoint_spec
from dgcl_dp.sequences import TokenEncoder, TokenVocabulary
from dgcl_dp.source import parse_java


def test_token_encoders_pool_windows_without_batch_padding_leakage():
    files = [parse_java(b"class A { int f(){return 1;} }"), parse_java(b"class B {}")]
    vocabulary = TokenVocabulary.fit(files[:1], window=8)
    assert "B" not in vocabulary.lookup
    graphs = [vocabulary.graphs(f) for f in files]
    for kind in ("cnn", "bilstm"):
        model = TokenEncoder(len(vocabulary.tokens), hidden=8, kind=kind).eval()
        torch.testing.assert_close(model(graphs)[0], model(graphs[:1])[0], rtol=1e-5, atol=1e-6)


class StubTokenizer:
    pad_token_id = 1
    cls_token_id = 0
    sep_token_id = 2
    unk_token_id = 3

    def encode(self, text, add_special_tokens=False):
        return [4] * max(1, len(text.split()))

    def build_inputs_with_special_tokens(self, values):
        return [0, *values, 2]

    def convert_tokens_to_ids(self, marker):
        return 5


class StubBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(8, 8)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_pinned_pretrained_window_adapter_with_local_stub_no_download():
    parsed = parse_java(b"class A { int f(){return 1;} }")
    vocabulary = PretrainedVocabulary("codebert", tokenizer=StubTokenizer())
    graphs = vocabulary.graphs(parsed)
    model = PretrainedEncoder(vocabulary, hidden=4, backbone=StubBackbone()).eval()
    assert model([graphs]).shape == (1, 4)
    assert len(vocabulary.spec["revision"]) == 40
    unix = PretrainedVocabulary("unixcoder", tokenizer=StubTokenizer())
    assert unix.graphs(parsed)[0].token[:3].tolist() == [0, 5, 2]
    with pytest.raises(ValueError, match="data-flow"):
        checkpoint_spec("graphcodebert")
