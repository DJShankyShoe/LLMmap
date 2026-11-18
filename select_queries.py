#!/usr/bin/env python3
"""Pick the most discriminative subset of a query pool (greedy forward selection).

Every query sent to the target is a chance to be noticed, so it pays to send as
few as possible. Collect the dataset once with the full candidate pool
(e.g. confs/queries/soc_pool.json), then run this script to find the k queries
that best separate the LLMs, and train with `"query_subset": [...]` in the
training conf (no need to re-query any model).

Scoring proxy (fast, no training): each trace is represented by the concatenation
of the L2-normalised embeddings of its answers to the selected queries; a test
trace is assigned to the nearest per-LLM centroid computed on the train split.
The proxy correlates well with the trained model's accuracy and is cheap enough
to evaluate thousands of subsets.

    python select_queries.py confs/soc.json -k 8 --write confs/queries/soc_selected.json
"""
import argparse
import json

import numpy as np
import torch

from LLMmap.dataset_maker import read_dataset
from LLMmap.dataset import build_cache
from LLMmap.utility import read_conf_file


def build_tensors(entries, cache, llm_map):
    norm = cache.normalizer
    X = torch.stack([torch.stack([cache(norm(o)) for _, o in e['traces']]) for e in entries]).numpy()  # (N, Q, E)
    X = X / np.linalg.norm(X, axis=-1, keepdims=True).clip(1e-12)
    y = np.array([llm_map[e['llm']] for e in entries])
    return X, y


def score(subset, Xtr, ytr, Xte, yte, n_classes):
    ftr = Xtr[:, subset].reshape(len(Xtr), -1)
    fte = Xte[:, subset].reshape(len(Xte), -1)
    C = np.zeros((n_classes, ftr.shape[1]), dtype=np.float32)
    for c in range(n_classes):
        m = ytr == c
        if m.any():
            C[c] = ftr[m].mean(0)
    sims = fte @ C.T - 0.5 * (C ** 2).sum(1)[None, :]  # argmax == argmin euclidean distance
    order = np.argsort(-sims, axis=1)
    top1 = float((order[:, 0] == yte).mean())
    top3 = float((order[:, :3] == yte[:, None]).any(1).mean())
    return top1, top3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('conf', help='training conf (uses dataset_path, embedding model, normalisation, embedding_cache_path)')
    ap.add_argument('-k', type=int, default=8, help='number of queries to select')
    ap.add_argument('--must', nargs='*', type=int, default=[], help='query indices that must be included')
    ap.add_argument('--never', nargs='*', type=int, default=[], help='query indices to exclude (e.g. ones you judge too risky)')
    ap.add_argument('--write', help='write the selected queries (strings) to this JSON file')
    args = ap.parse_args()

    conf = read_conf_file(args.conf)
    conf.pop('query_subset', None)
    train, test = read_dataset(conf['dataset_path'], seed=conf.get('seed', 0))
    if conf.get('num_istances_dataset'):
        train, test = train[:conf['num_istances_dataset'][0]], test[:conf['num_istances_dataset'][1]]
    cache, _ = build_cache(conf, train, test)
    queries = cache.queries
    llm_map = cache.llms_map
    Xtr, ytr = build_tensors(train, cache, llm_map)
    Xte, yte = build_tensors(test, cache, llm_map)
    n_classes = len(llm_map)
    print(f"{len(queries)} candidate queries, {n_classes} LLMs, {len(train)} train / {len(test)} test traces\n")

    print("Single-query accuracy (top-1 / top-3):")
    solo = []
    for i, q in enumerate(queries):
        t1, t3 = score([i], Xtr, ytr, Xte, yte, n_classes)
        solo.append(t1)
        print(f"  [{i:2d}] {t1:.3f} / {t3:.3f}   {q[:90]!r}")

    selected = list(args.must)
    remaining = [i for i in range(len(queries)) if i not in selected and i not in args.never]
    print("\nGreedy forward selection:")
    while len(selected) < args.k and remaining:
        best = max(remaining, key=lambda i: score(selected + [i], Xtr, ytr, Xte, yte, n_classes))
        selected.append(best)
        remaining.remove(best)
        t1, t3 = score(selected, Xtr, ytr, Xte, yte, n_classes)
        print(f"  +[{best:2d}] -> {len(selected)} queries: top-1 {t1:.3f}  top-3 {t3:.3f}   {queries[best][:70]!r}")

    t1_all, t3_all = score(list(range(len(queries))), Xtr, ytr, Xte, yte, n_classes)
    print(f"\nAll {len(queries)} queries: top-1 {t1_all:.3f}  top-3 {t3_all:.3f}")
    print(f"Selected indices (put in the training conf as \"query_subset\"): {sorted(selected)}")
    if args.write:
        with open(args.write, 'w') as f:
            json.dump([queries[i] for i in sorted(selected)], f, indent=2, ensure_ascii=False)
        print(f"Wrote {args.write}")


if __name__ == '__main__':
    main()
