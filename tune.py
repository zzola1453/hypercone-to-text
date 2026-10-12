"""Hyper-parameter search without touching the test set.

Within the known classes of a split, 25% are held out as pseudo-unknown: models train on the train
samples of the remaining classes and are scored on the validation samples of all known classes.

python -m its.tune --datasets clinc banking --ratio 0.75 --seeds 0 1
"""
import argparse
import itertools
import json
import os

import numpy as np
import torch

from its.data import load, open_split
from its.embed import DEFAULT_MODEL, embed
from its.methods import cone_predict, train_cone
from its.metrics import evaluate


def grid():
    yield {"outliers": "none"}
    for space, beta, gamma in itertools.product(["feature", "input"], [1.0, 3.0], [0.0, 1.0]):
        yield {"outliers": "gauss", "space": space, "beta": beta, "gamma": gamma}
        for delta in [0.05, 0.15]:
            yield {"outliers": "tvmf", "space": space, "beta": beta, "gamma": gamma, "delta": delta}


def pseudo_split(emb, labels, names, oos, ratio, seed, dev):
    known, sp = open_split(labels, len(names), oos, ratio, seed)
    K = len(known)
    rng = np.random.RandomState(seed + 100)
    held = np.zeros(K, dtype=bool)
    held[rng.choice(K, K // 4, replace=False)] = True
    K2 = int((~held).sum())
    remap = np.full(K, K2)
    remap[~held] = np.arange(K2)
    itr, ytr = sp["train"]
    keep = ~held[ytr]
    iva, yva = sp["validation"]
    xtr = emb["train"][itr[keep]].to(dev)
    ytr = torch.as_tensor(remap[ytr[keep]], device=dev)
    xva = emb["validation"][iva].to(dev)
    return xtr, ytr, xva, remap[yva], K2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["clinc", "banking"])
    p.add_argument("--ratio", type=float, default=0.75)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--tag", default="tune")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    configs = list(grid())
    scores = {i: [] for i in range(len(configs))}
    for ds in a.datasets:
        data, names, oos = load(ds)
        emb = embed(ds, data, DEFAULT_MODEL)
        labels = {s: y for s, (_, y) in data.items()}
        for seed in a.seeds:
            xtr, ytr, xva, yva, K = pseudo_split(emb, labels, names, oos, a.ratio, seed, dev)
            for i, cfg in enumerate(configs):
                net, theta = train_cone(xtr, ytr, K, a.epochs, seed, **cfg)
                r = evaluate(*(t.cpu() for t in cone_predict(net, theta, xva, K)), yva, K)
                scores[i].append(r)
                print(ds, seed, cfg, f"auroc={100 * r['auroc']:.2f} fpr95={100 * r['fpr95']:.2f}", flush=True)
    table = []
    for i, cfg in enumerate(configs):
        table.append({"cfg": cfg, **{c: float(np.mean([s[c] for s in scores[i]])) for c in ("auroc", "fpr95")}})
    table.sort(key=lambda t: -t["auroc"])
    os.makedirs("its/results", exist_ok=True)
    with open(f"its/results/{a.tag}.json", "w") as f:
        json.dump({"args": vars(a), "table": table}, f, indent=1)
    for t in table:
        print(f"auroc={100 * t['auroc']:.2f} fpr95={100 * t['fpr95']:.2f} {t['cfg']}")


if __name__ == "__main__":
    main()
