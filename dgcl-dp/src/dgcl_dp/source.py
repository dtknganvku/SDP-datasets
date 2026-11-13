"""Named-node Java graphs and label-independent clone fingerprints."""

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import tree_sitter_java
from torch_geometric.data import Data
from tree_sitter import Language, Parser


LANGUAGE = Language(tree_sitter_java.language())
METHODS = {"method_declaration", "constructor_declaration", "compact_constructor_declaration"}
COMMENTS = {"line_comment", "block_comment"}


def walk(root):
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type in COMMENTS:
            continue
        yield node
        stack.extend(reversed(node.named_children))


def lexical_tokens(source):
    tree = Parser(LANGUAGE).parse(source)
    stack, tokens = [tree.root_node], []
    while stack:
        node = stack.pop()
        if node.type in COMMENTS:
            continue
        if node.children:
            stack.extend(reversed(node.children))
        else:
            tokens.append(source[node.start_byte : node.end_byte].decode("utf-8", errors="replace"))
    return tokens


def fingerprint(source):
    tokens = lexical_tokens(source)
    grams = {tuple(tokens[i : i + 5]) for i in range(max(0, len(tokens) - 4))}
    return {
        "raw_hash": hashlib.sha256(source).hexdigest(),
        "token_hash": hashlib.sha256("\0".join(tokens).encode()).hexdigest(),
        "tokens": len(tokens),
        "grams": grams,
    }


def clone_families(sources):
    """Exact hashes plus fixed 5-gram Jaccard >= .9, length ratio >= .8."""
    ids = sorted(sources)
    fp = {i: fingerprint(sources[i]) for i in ids}
    parent = {i: i for i in ids}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    inverted, hashes = defaultdict(set), {}
    for i in ids:
        f = fp[i]
        candidates = set()
        for gram in f["grams"]:
            candidates.update(inverted[gram])
        for key in ("raw_hash", "token_hash"):
            if f[key] in hashes:
                candidates.add(hashes[f[key]])
            hashes[f[key]] = i
        for j in candidates:
            g = fp[j]
            exact = f["token_hash"] == g["token_hash"] or f["raw_hash"] == g["raw_hash"]
            union = f["grams"] | g["grams"]
            similar = (
                min(f["tokens"], g["tokens"]) / max(1, f["tokens"], g["tokens"]) >= 0.8
                and bool(union)
                and len(f["grams"] & g["grams"]) / len(union) >= 0.9
            )
            if exact or similar:
                a, b = sorted((root(i), root(j)))
                parent[b] = a
        for gram in f["grams"]:
            inverted[gram].add(i)
    return {i: root(i) for i in ids}


@dataclass
class ParsedFile:
    partitions: list
    records: list
    top_level_classes: int
    status: str
    primary_eligible: bool = False
    lexical: list = field(default_factory=list)


def parse_java(source, max_nodes=2000, first_nodes=None):
    if max_nodes < 1:
        raise ValueError("max_nodes must be positive")
    tree = Parser(LANGUAGE).parse(source)
    root = tree.root_node
    declarations = [
        c
        for c in root.named_children
        if c.type
        in {
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "record_declaration",
            "annotation_type_declaration",
        }
    ]
    top = len(declarations)
    if root.has_error:
        return ParsedFile([], [], top, "parse_failure")
    nodes = list(walk(root))
    groups, records = [], []
    if first_nodes is not None:
        groups = [nodes[:first_nodes]]
        records.append({"policy": "first_nodes", "dropped": max(0, len(nodes) - first_nodes)})
    elif len(nodes) <= max_nodes:
        groups = [nodes]
        records.append({"policy": "whole", "dropped": 0})
    else:
        # Choose outermost method roots; nested/local-class methods stay inside them.
        methods, covered = [], set()
        for node in nodes:
            if node.type in METHODS and node.id not in covered:
                subtree = list(walk(node))
                covered.update(n.id for n in subtree)
                methods.append(subtree)
        skeleton = [n for n in nodes if n.id not in covered]
        for start in range(0, len(skeleton), max_nodes):
            groups.append(skeleton[start : start + max_nodes])
            records.append({"policy": "scaffold", "dropped": 0})
        for method in methods:
            groups.append(method[:max_nodes])
            records.append(
                {
                    "policy": "method",
                    "start_byte": method[0].start_byte,
                    "end_byte": method[0].end_byte,
                    "dropped": max(0, len(method) - max_nodes),
                }
            )
    partitions = []
    for group, record in zip(groups, records):
        indices = {node.id: i for i, node in enumerate(group)}
        types, order, edges, attrs = [], [], [], []
        for node in group:
            types.append(node.type)
            siblings = (
                [c for c in node.parent.named_children if c.type not in COMMENTS]
                if (node.parent is not None)
                else [node]
            )
            ordinal = next(i for i, c in enumerate(siblings) if c.id == node.id)
            order.append(ordinal / max(1, len(siblings) - 1))
            if node.parent is not None and node.parent.id in indices:
                p, c = indices[node.parent.id], indices[node.id]
                edges.extend(((p, c), (c, p)))
                attrs.extend(((1, ordinal), (-1, ordinal)))
        record["nodes"] = len(group)
        record["node_byte_ranges"] = [[n.start_byte, n.end_byte] for n in group]
        partitions.append({"types": types, "order": order, "edges": edges, "edge_attrs": attrs})
    eligible = len(declarations) == 1 and declarations[0].type == "class_declaration"
    return ParsedFile(partitions, records, top, "ok", eligible, lexical_tokens(source))


class Vocabulary:
    def __init__(self, tokens=()):
        self.tokens = ["<UNK>"] + sorted(set(tokens) - {"<UNK>"})
        self.lookup = {token: i for i, token in enumerate(self.tokens)}

    @classmethod
    def fit(cls, files):
        return cls(t for f in files for p in f.partitions for t in p["types"])

    def graphs(self, parsed):
        if parsed.status != "ok":
            raise ValueError("Parse failures must be excluded consistently across models")
        return [
            Data(
                token=torch.tensor([self.lookup.get(t, 0) for t in p["types"]], dtype=torch.long),
                order=torch.tensor(p["order"], dtype=torch.float32).reshape(-1, 1),
                edge_index=torch.tensor(p["edges"], dtype=torch.long).reshape(-1, 2).T.contiguous(),
                edge_attr=torch.tensor(p["edge_attrs"], dtype=torch.long).reshape(-1, 2),
                num_nodes=len(p["types"]),
            )
            for p in parsed.partitions
        ]
