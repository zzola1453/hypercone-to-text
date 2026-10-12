"""Qualitative results: which test sentences each model accepts or rejects, plus dataset samples.

Trains MSP and the hypercone + input-space tvMF model (raw cone decision, as in results/input.md)
on one split and writes its/results/examples.md.

python -m its.examples --dataset clinc --ratio 0.75 --seed 0
"""
import argparse
import os

import numpy as np
import torch

from its.data import load, open_split
from its.embed import embed
from its.methods import cone_predict, msp, train_cone, train_softmax


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c).replace("|", "/") for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="clinc")
    p.add_argument("--ratio", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--out", default="its/results/examples.md")
    a = p.parse_args()
    rng = np.random.RandomState(0)
    data, names, oos = load(a.dataset)
    emb = embed(a.dataset, data)
    labels = {s: y for s, (_, y) in data.items()}
    known, sp = open_split(labels, len(names), oos, a.ratio, a.seed)
    K = len(known)
    (itr, ytr), (iva, _), (ite, yte) = sp["train"], sp["validation"], sp["test"]
    xtr, xva, xte = emb["train"][itr].cuda(), emb["validation"][iva].cuda(), emb["test"][ite].cuda()
    ytr = torch.as_tensor(ytr).cuda()

    soft = train_softmax(xtr, ytr, K, 30, a.seed)
    p_msp, _ = msp(soft, xva, xte, K)
    net, theta = train_cone(xtr, ytr, K, 30, a.seed, outliers="tvmf", space="input", delta=0.15)
    p_cone, _ = cone_predict(net, theta, xte, K)
    with torch.no_grad():
        ang = net.angles(net.feats(xte))
        near = (theta - ang).argmax(1)                         # the cone the decision rule uses
        deg = torch.rad2deg(ang.gather(1, near[:, None]).squeeze(1))
        th_deg = torch.rad2deg(theta[near])
    p_msp, p_cone, near, deg, th_deg = (t.cpu().numpy() for t in (p_msp, p_cone, near, deg, th_deg))

    texts = np.array(data["test"][0], dtype=object)
    orig = labels["test"]
    kname = lambda i: names[known[i]] if i < K else "거부(unknown)"
    true = lambda j: names[orig[j]] + ("" if yte[j] < K else " *")

    def pick(mask, n):
        idx = np.flatnonzero(mask)
        return rng.choice(idx, min(n, len(idx)), replace=False) if len(idx) else idx

    def rows(idx):
        return [[texts[j], true(j), kname(p_msp[j]), kname(p_cone[j]),
                 f"{deg[j]:.1f}° / {th_deg[j]:.1f}° ({names[known[near[j]]]})"] for j in idx]

    hdr = ["문장", "정답 (* = 학습 안 한 의도)", "MSP 판정", "Hypercone+tvMF 판정", "판정 원뿔까지 각도 / 그 원뿔 반지름"]
    unk, kn = yte == K, yte < K
    cases = [
        ("A. 관련 없는 문장: MSP는 받아들였지만 Hypercone은 거부 (개선 사례)", unk & (p_msp < K) & (p_cone == K)),
        ("B. 관련 없는 문장: 둘 다 거부 (쉬운 사례)", unk & (p_msp == K) & (p_cone == K)),
        ("C. 관련 없는 문장: Hypercone도 받아들임 (한계 사례, near-OOD)", unk & (p_cone < K)),
        ("D. 학습한 의도: Hypercone이 잘못 거부", kn & (p_cone == K)),
        ("E. 학습한 의도: Hypercone이 정확히 분류", kn & (p_cone == yte)),
        ("F. 관련 없는 문장: MSP는 거부했지만 Hypercone은 받아들임 (역전 사례)", unk & (p_msp == K) & (p_cone < K)),
    ]
    count = lambda m: f"{int(m.sum())}개 ({m.mean() * 100:.1f}%)"
    summary = [
        ["학습한 의도 문장 중 정확히 분류", count((p_msp == yte)[kn]), count((p_cone == yte)[kn])],
        ["학습한 의도 문장 중 잘못 거부", count((p_msp == K)[kn]), count((p_cone == K)[kn])],
        ["관련 없는 문장 중 거부", count((p_msp == K)[unk]), count((p_cone == K)[unk])],
        ["관련 없는 문장 중 받아들임 (새어 들어옴)", count((p_msp < K)[unk]), count((p_cone < K)[unk])],
    ]
    lines = [f"# 실제 문장 판정 사례: {a.dataset}, known {int(a.ratio * 100)}%, seed {a.seed}\n",
             f"학습한 의도 {K}개, 테스트 문장 {len(yte)}개 (학습한 의도 {int(kn.sum())}개, 관련 없는 문장 {int(unk.sum())}개). "
             "Hypercone+tvMF는 입력 공간 tvMF 가상 이상치(δ=0.15)로 학습하고 원뿔 경계 그대로 판정, MSP는 validation known 95% 통과 기준.\n",
             "## 요약\n", table(summary, ["", "MSP", "Hypercone+tvMF"]), ""]
    for title, m in cases:
        lines += [f"\n## {title}\n", f"전체 {int(m.sum())}개 중 무작위 {min(a.n, int(m.sum()))}개\n", table(rows(pick(m, a.n)), hdr)]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:6]))
    print("saved", a.out)


if __name__ == "__main__":
    main()
