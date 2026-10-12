"""Run the open-set benchmark.

python run.py --datasets clinc banking --ratios 0.25 0.5 0.75 --seeds 0 1 2
Results go to results/<tag>.json (one row per dataset/ratio/seed/method) and <tag>.md (mean ± std).
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from data import load, open_split
from embed import DEFAULT_MODEL, embed
from methods import (adb, cone_predict, cosine_msp, energy, knn, mahalanobis, msp, train_cone,
                     train_softmax)
from metrics import evaluate

HERE = os.path.dirname(os.path.abspath(__file__))

METHODS = ["msp", "adb", "cone_q", "cone_gauss", "cone_vmf", "cone_tvmf"]
COLS = ["acc", "f1_all", "f1_known", "f1_unknown", "auroc", "fpr95"]


def run_one(method, xtr, ytr, xva, xte, K, a, seed, cache):
    """Returns {row name: (pred, score)}. Cone methods give a raw (score >= 0) and a calibrated row."""
    if method in ("msp", "adb", "energy", "maha", "knn"):
        if "softmax" not in cache:
            cache["softmax"] = train_softmax(xtr, ytr, K, a.epochs, seed)
        net = cache["softmax"]
        if method == "msp":
            return {"msp": msp(net, xva, xte, K)}
        if method == "energy":
            return {"energy": energy(net, xva, xte, K)}
        if method == "maha":
            return {"maha": mahalanobis(net, xtr, ytr, xva, xte, K)}
        if method == "knn":
            return {"knn": knn(net, xtr, xva, xte, K)}
        return {"adb": adb(net, xtr, ytr, xte, K, seed=seed, xval=xva), "adb_raw": adb(net, xtr, ytr, xte, K, seed=seed)}
    if method.startswith("cq_"):                      # other decision rules on the cone_q (cosine-softmax) network
        if "coneq" not in cache:
            cache["coneq"] = train_cone(xtr, ytr, K, a.epochs, seed, outliers="none", q=a.q)
        net = cache["coneq"][0]
        cos_logits = lambda x: net.feats(x) @ net.protos().T
        rule = method[3:]
        if rule == "msp":
            return {method: cosine_msp(net, xva, xte, K)}
        if rule == "maha":
            return {method: mahalanobis(net, xtr, ytr, xva, xte, K, feat=net.feats)}
        if rule == "maha_raw":
            return {method: mahalanobis(net, xtr, ytr, xva, xte, K, feat=net.body)}
        if rule == "knn":
            return {method: knn(net, xtr, xva, xte, K, feat=net.feats, logits=cos_logits)}
        raise ValueError(method)
    kind = method.split("_", 1)[1]
    net, theta = train_cone(xtr, ytr, K, a.epochs, seed, outliers="none" if kind == "q" else kind,
                            space=a.space, beta=a.beta, gamma=a.gamma, delta=a.delta, q=a.q)
    return {method: cone_predict(net, theta, xte, K, xva), method + "_raw": cone_predict(net, theta, xte, K)}


def summarize(rows):
    lines = []
    for ds in dict.fromkeys(r["dataset"] for r in rows):
        for ratio in dict.fromkeys(r["ratio"] for r in rows if r["dataset"] == ds):
            lines += [f"\n### {ds}, known {int(ratio * 100)}%\n", "| method | " + " | ".join(COLS) + " |",
                      "|---|" + "---|" * len(COLS)]
            for m in dict.fromkeys(r["method"] for r in rows):
                sel = [r for r in rows if r["dataset"] == ds and r["ratio"] == ratio and r["method"] == m]
                if sel:
                    cells = [f"{100 * np.mean([r[c] for r in sel]):.2f} ± {100 * np.std([r[c] for r in sel]):.2f}"
                             for c in COLS]
                    lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["clinc", "banking"])
    p.add_argument("--ratios", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--methods", nargs="+", default=METHODS)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--delta", type=float, default=0.1)
    p.add_argument("--q", type=float, default=0.95)
    p.add_argument("--space", default="feature", choices=["feature", "input"])
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tag", default="main")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(f"{HERE}/results", exist_ok=True)
    rows = []
    for ds in a.datasets:
        data, names, oos = load(ds)
        emb = embed(ds, data, a.model)
        labels = {s: y for s, (_, y) in data.items()}
        for ratio in a.ratios:
            for seed in a.seeds:
                known, sp = open_split(labels, len(names), oos, ratio, seed)
                K = len(known)
                (itr, ytr), (iva, _), (ite, yte) = sp["train"], sp["validation"], sp["test"]
                xtr, xva, xte = (emb[s][i].to(dev) for s, i in (("train", itr), ("validation", iva), ("test", ite)))
                ytr = torch.as_tensor(ytr, device=dev)
                cache = {}
                for m in a.methods:
                    t0 = time.time()
                    for name, (pred, score) in run_one(m, xtr, ytr, xva, xte, K, a, seed, cache).items():
                        r = evaluate(pred.cpu(), score.cpu(), yte, K)
                        rows.append({"dataset": ds, "ratio": ratio, "seed": seed, "method": name, "K": K, **r})
                        print(f"{ds} {ratio} s{seed} {name:15s} " + " ".join(f"{c}={100 * r[c]:.2f}" for c in COLS)
                              + f" ({time.time() - t0:.0f}s)", flush=True)
                with open(f"{HERE}/results/{a.tag}.json", "w") as f:
                    json.dump({"args": vars(a), "rows": rows}, f, indent=1)
    md = summarize(rows)
    with open(f"{HERE}/results/{a.tag}.md", "w", encoding="utf-8") as f:
        f.write(f"args: {vars(a)}\n" + md)
    print(md)


if __name__ == "__main__":
    main()
