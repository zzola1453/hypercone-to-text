"""Summary figures and tables for the professor briefing, from its/results/{main,input,baselines,cosine_rules}.json.

figures/summary_overall.png:  every method on AUROC / FPR95 / F1, 18-setting mean
figures/summary_paired.png:   per-setting hypercone vs MSP / kNN / Mahalanobis (AUROC, FPR95)
figures/summary_gain.png:     MSP -> cosine MSP -> hypercone on the same network (where the AUROC gain comes from)
figures/summary_margin.png:   score distributions of known vs unknown test sentences, MSP vs hypercone (one split per
                              dataset; trains the two models, needs the embedding cache)
results/summary.md:           the same numbers as tables, with paired win counts and Wilcoxon tests

python -m its.plot_summary            # everything
python -m its.plot_summary --no_margin  # skip the figure that trains models
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from its.plot_results import GRID, INK, INK2, SERIES, SURFACE, style

MUTED = "#a9a7a1"
DATASETS = [("clinc", "CLINC150"), ("banking", "BANKING77")]
RATIOS = [0.25, 0.5, 0.75]
# (key, label, family): family "ours" is drawn in color, the rest in gray
OVERALL = [("msp", "Softmax 확신도 (MSP)", "base"), ("energy", "Energy", "base"), ("adb", "ADB (재구현)", "base"),
           ("knn", "kNN (k=10)", "base"), ("maha", "Mahalanobis", "base"),
           ("cone_q", "Hypercone", "ours"), ("in_tvmf", "Hypercone + tvMF", "ours")]
METRICS = [("auroc", "AUROC", "높을수록 좋음", True), ("fpr95", "FPR95", "낮을수록 좋음", False),
           ("f1_all", "F1 전체", "높을수록 좋음", True)]


def load_rows():
    R = {}
    for tag, prefix in (("main", ""), ("baselines", ""), ("cosine_rules", ""), ("input", "in_")):
        for r in json.load(open(f"its/results/{tag}.json"))["rows"]:
            m = r["method"].replace("cone_", prefix) if prefix and r["method"].startswith("cone_") else r["method"]
            R[(m, r["dataset"], r["ratio"], r["seed"])] = r     # msp / cone_q repeat across files with identical values
    return R


def values(R, m, c, ds=None):
    """Per-setting values (%) in a fixed (dataset, ratio, seed) order, so two methods line up as pairs."""
    keys = [(d, q, s) for d, _ in DATASETS for q in RATIOS for s in (0, 1, 2) if ds in (None, d)]
    return np.array([R[(m, *k)][c] * 100 for k in keys])


def wins(a, b, higher):
    return int(((a > b) if higher else (a < b)).sum())


# ---------------------------------------------------------------- figures

def overall_figure(R):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), facecolor=SURFACE, sharey=True)
    y = np.arange(len(OVERALL))[::-1]
    for ax, (c, name, hint, higher) in zip(axes, METRICS):
        style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, lw=0.6)
        # AUROC / FPR95 ignore the threshold, so the raw cone row equals the calibrated one; F1 uses the raw rows
        keys = [k + "_raw" if c == "f1_all" and fam == "ours" else k for k, _, fam in OVERALL]
        v = [values(R, k, c).mean() for k in keys]
        lo, hi = min(v), max(v)
        pad = (hi - lo) * 0.25
        left = lo - pad
        for yi, vi, (_, _, fam) in zip(y, v, OVERALL):
            col = SERIES[0] if fam == "ours" else MUTED
            ax.hlines(yi, left, vi, color=GRID, lw=1)
            ax.scatter(vi, yi, s=90, color=col, edgecolors=SURFACE, linewidths=2, zorder=3)
            ax.text(vi + (pad if higher else -pad) * 0.12, yi, f"{vi:.1f}", va="center", fontsize=10,
                    color=INK if fam == "ours" else INK2, fontweight="bold" if fam == "ours" else "normal")
        if c == "f1_all":                                   # calibrated hypercone F1 for comparison
            vc = values(R, "cone_q", c).mean()
            ax.scatter(vc, y[5], s=70, facecolors="none", edgecolors=SERIES[0], linewidths=1.6, zorder=3)
            ax.annotate(f"보정 시 {vc:.1f}", (vc, y[5]), xytext=(0, -15), textcoords="offset points",
                        ha="center", fontsize=8, color=INK2)
        ax.set_xlim(left, hi + pad)
        if not higher:
            ax.invert_xaxis()
        ax.set_title(f"{name}  ({hint}, → 쪽이 좋음)", loc="left", fontsize=11, color=INK)
    axes[0].set_yticks(y, [lab for _, lab, _ in OVERALL], fontsize=10, color=INK)
    fig.text(0.01, 0.97, "전체 비교: 18개 설정 평균 (2 데이터셋 × 학습 의도 25/50/75% × seed 3)", fontsize=14,
             color=INK, va="top")
    fig.text(0.01, 0.015, "FPR95 축은 뒤집어 그려서 세 패널 모두 오른쪽이 좋은 쪽. 기준 방법은 validation known 95% 통과로 기준값 보정, "
             "Hypercone 계열 F1은 보정 없이 원뿔 경계 그대로 판정 (빈 원 = 같은 모델을 보정했을 때).",
             fontsize=8.5, color=INK2)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.83, bottom=0.12, wspace=0.12)
    fig.savefig("its/figures/summary_overall.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def paired_figure(R):
    rivals = [("msp", "Softmax 확신도 (MSP)"), ("knn", "kNN (k=10)"), ("maha", "Mahalanobis")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6), facecolor=SURFACE)
    for i, (c, name, _, higher) in enumerate(METRICS[:2]):
        for j, (m, label) in enumerate(rivals):
            ax = axes[i, j]
            style(ax)
            ax.grid(axis="x", color=GRID, lw=0.6)
            ours_all, theirs_all = values(R, "cone_q", c), values(R, m, c)
            lo = min(ours_all.min(), theirs_all.min()) - 1.5
            hi = max(ours_all.max(), theirs_all.max()) + 1.5
            # shade the half where the hypercone is better
            if higher:
                ax.fill_between([lo, hi], [lo, hi], hi, color=SERIES[0], alpha=0.06, lw=0)
            else:
                ax.fill_between([lo, hi], lo, [lo, hi], color=SERIES[0], alpha=0.06, lw=0)
            ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls=(0, (4, 3)))
            for k, (ds, dname) in enumerate(DATASETS):
                ax.scatter(values(R, m, c, ds), values(R, "cone_q", c, ds), s=55, color=SERIES[k],
                           marker="os"[k], edgecolors=SURFACE, linewidths=1.5, zorder=3, label=dname)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            w = wins(ours_all, theirs_all, higher)
            ax.set_title(f"vs {label}: Hypercone 우세 {w}/18", loc="left", fontsize=11, color=INK)
            ax.set_xlabel(f"{label} {name}", fontsize=9, color=INK2)
            if j == 0:
                ax.set_ylabel(f"Hypercone {name}", fontsize=10, color=INK)
            corner = (0.04, 0.93) if higher else (0.96, 0.06)
            ax.text(*corner, "Hypercone이 나은 쪽", transform=ax.transAxes, fontsize=8.5, color=SERIES[0],
                    ha="left" if higher else "right", va="top" if higher else "bottom")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.98, 0.985))
    fig.text(0.01, 0.975, "설정 18개 하나하나에서의 비교 (점 하나 = 데이터셋·비율·seed 하나)", fontsize=14, color=INK, va="top")
    fig.text(0.01, 0.012, "윗줄 AUROC(높을수록 좋음), 아랫줄 FPR95(낮을수록 좋음). 점선은 두 방법이 같은 선, 파란 영역에 있으면 "
             "Hypercone이 나은 설정. 기준 방법은 모두 같은 임베딩·같은 크기 MLP.", fontsize=8.5, color=INK2)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.08, hspace=0.32, wspace=0.25)
    fig.savefig("its/figures/summary_paired.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def gain_figure(R):
    steps = [("msp", "Softmax 확신도 (MSP)", "일반 softmax로 학습"),
             ("cq_msp", "cosine MSP", "같은 판정, 학습만 cosine softmax로"),
             ("cone_q", "Hypercone", "같은 신경망, 판정만 원뿔로")]
    v = [values(R, m, "auroc").mean() for m, _, _ in steps]
    w = [None, wins(values(R, "cq_msp", "auroc"), values(R, "msp", "auroc"), True),
         wins(values(R, "cone_q", "auroc"), values(R, "cq_msp", "auroc"), True)]
    fig, ax = plt.subplots(figsize=(9.5, 4.4), facecolor=SURFACE)
    style(ax)
    x = np.arange(3)
    ax.plot(x, v, color=SERIES[0], lw=2, zorder=2)
    ax.scatter(x, v, s=110, color=[MUTED, SERIES[0], SERIES[0]], edgecolors=SURFACE, linewidths=2, zorder=3)
    for xi, vi in zip(x, v):
        ax.text(xi, vi + 0.25, f"{vi:.2f}", ha="center", fontsize=11, color=INK)
    for i in (1, 2):
        ax.text(i - 0.5 + 0.08, (v[i - 1] + v[i]) / 2 - 0.35, f"+{v[i] - v[i - 1]:.2f}%p\n({w[i]}/18 설정)",
                va="top", ha="left", fontsize=10, color=SERIES[0], fontweight="bold")
    ax.set_xticks(x, [f"{name}\n{desc}" for _, name, desc in steps], fontsize=9.5, color=INK)
    ax.set_xlim(-0.4, 2.4)
    ax.set_ylim(min(v) - 1.2, max(v) + 1.0)
    ax.set_ylabel("AUROC, 18개 설정 평균 (%)", fontsize=9, color=INK2)
    fig.text(0.02, 0.96, f"MSP 대비 AUROC +{v[2] - v[0]:.2f}%p는 어디서 오는가", fontsize=13, color=INK, va="top")
    fig.text(0.02, 0.885, "한 번에 하나씩만 바꿔 가며 비교. 학습 방식 변경과 원뿔 판정이 이득을 나눠 만든다.",
             fontsize=9, color=INK2, va="top")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.8, bottom=0.2)
    fig.savefig("its/figures/summary_gain.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def margin_figure(ratio=0.75, seed=0):
    import torch
    import torch.nn.functional as F

    from its.data import load, open_split
    from its.embed import embed
    from its.methods import train_cone, train_softmax

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.8), facecolor=SURFACE)
    stats = []
    for i, (ds, dname) in enumerate(DATASETS):
        data, names, oos = load(ds)
        emb = embed(ds, data)
        labels = {s: y for s, (_, y) in data.items()}
        known, sp = open_split(labels, len(names), oos, ratio, seed)
        K = len(known)
        (itr, ytr), (iva, _), (ite, yte) = sp["train"], sp["validation"], sp["test"]
        xtr, xva, xte = (emb[s][j].to(dev) for s, j in (("train", itr), ("validation", iva), ("test", ite)))
        ytr = torch.as_tensor(ytr, device=dev)
        is_known = np.asarray(yte) < K
        soft = train_softmax(xtr, ytr, K, 30, seed)
        net, theta = train_cone(xtr, ytr, K, 30, seed, outliers="none")
        with torch.no_grad():
            p_te = F.softmax(soft(xte), 1).max(1).values.cpu().numpy()
            tau = F.softmax(soft(xva), 1).max(1).values.quantile(0.05).item()
            m_te = torch.rad2deg((theta - net.angles(net.feats(xte))).max(1).values).cpu().numpy()
        panels = [(p_te, tau, "Softmax 확신도 (MSP): 최대 확률", "최대 softmax 확률", np.linspace(0, 1, 41)),
                  (m_te, 0.0, "Hypercone: 가장 가까운 원뿔 안쪽 여유 m = θ - ∠(z, μ)", "m (도), 0보다 크면 원뿔 안",
                   np.arange(np.floor(m_te.min() / 2) * 2, np.ceil(m_te.max() / 2) * 2 + 2, 2))]
        for j, (s, thr, title, xlabel, bins) in enumerate(panels):
            ax = axes[i, j]
            style(ax)
            for mask, col, lab in ((is_known, SERIES[0], "학습한 의도의 문장"), (~is_known, SERIES[1], "관련 없는 문장")):
                ax.hist(s[mask], bins=bins, color=col, alpha=0.55, label=lab, weights=np.full(mask.sum(), 100 / mask.sum()))
            ax.axvline(thr, color=INK, lw=1.4)
            leak = 100 * (s[~is_known] >= thr).mean()
            drop = 100 * (s[is_known] < thr).mean()
            ax.text(0.02 if j == 0 else 0.98, 0.95, f"새어 들어온 관련 없는 문장 {leak:.1f}%\n잘못 거부한 학습 문장 {drop:.1f}%",
                    transform=ax.transAxes, fontsize=9.5, color=INK, va="top", ha="left" if j == 0 else "right",
                    bbox=dict(facecolor=SURFACE, edgecolor=GRID, boxstyle="round,pad=0.4"))
            ax.set_title(f"{dname} · {title}", loc="left", fontsize=10.5, color=INK)
            ax.set_xlabel(xlabel + (" (세로선 = validation 기준값)" if j == 0 else " (세로선 = 원뿔 경계, 보정 없음)"),
                          fontsize=9, color=INK2)
            if j == 0:
                ax.set_ylabel("비율 (%)", fontsize=9, color=INK2)
            stats.append((dname, ["MSP", "Hypercone"][j], leak, drop))
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.98, 0.985))
    fig.text(0.01, 0.975, f"테스트 문장의 점수 분포 (known {int(ratio * 100)}%, seed {seed})", fontsize=14, color=INK, va="top")
    fig.text(0.01, 0.012, "세로선 오른쪽은 수용, 왼쪽은 거부. MSP는 관련 없는 문장도 확률 1 근처에 몰려 두 분포가 겹치고, "
             "원뿔은 관련 없는 문장 대부분이 경계(0°) 바깥에 놓인다. 각 색의 막대 합은 100%.", fontsize=8.5, color=INK2)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.09, hspace=0.38, wspace=0.15)
    fig.savefig("its/figures/summary_margin.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return stats


# ---------------------------------------------------------------- tables

def md_table(header, rows, left=1):
    """Markdown table; the first `left` columns are text (left-aligned), the rest numbers (right-aligned)."""
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" if i < left else "---:" for i in range(len(header))) + "|"]
    return "\n".join(out + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def summary_md(R, margin_stats):
    rows_all = [("msp", "Softmax 확신도 (MSP)", "보정"), ("energy", "Energy", "보정"), ("knn", "kNN (k=10)", "보정"),
                ("adb", "ADB (재구현)", "보정"), ("maha", "Mahalanobis", "보정"), ("cone_q", "Hypercone", "보정"),
                ("cone_q_raw", "Hypercone", "경계 그대로"), ("in_tvmf", "Hypercone + tvMF (입력 공간)", "보정"),
                ("in_tvmf_raw", "Hypercone + tvMF (입력 공간)", "경계 그대로")]
    cols = [("auroc", True), ("fpr95", False), ("f1_all", True), ("f1_known", True), ("f1_unknown", True)]

    def fmt(m, c, ds=None):
        v = values(R, m, c, ds)
        return f"{v.mean():.2f} ± {v.std():.2f}"

    def best(c, higher, ds=None):
        v = [values(R, m, c, ds).mean() for m, _, _ in rows_all]
        return max(v) if higher else min(v)

    def cell(m, c, higher, ds=None):
        s = fmt(m, c, ds)
        return f"**{s}**" if abs(values(R, m, c, ds).mean() - best(c, higher, ds)) < 1e-9 else s

    out = ["# 결과 요약표", "",
           "`python -m its.plot_summary`로 생성. 값은 % 단위, `평균 ± 표준편차`. "
           "\"보정\"은 validation의 학습한 문장 95%가 통과하도록 기준값을 맞춘 판정, \"경계 그대로\"는 원뿔 경계(m ≥ 0)로 바로 판정한 것. "
           "AUROC·FPR95는 기준값과 무관해서 같은 모델의 두 판정에서 값이 같다. 열마다 가장 좋은 값은 굵게.", ""]
    hdr = ["방법", "판정", "AUROC ↑", "FPR95 ↓", "F1 전체 ↑", "F1 known ↑", "F1 unknown ↑"]
    out += ["## 1. 전체 (18개 설정)", "", "표준편차는 18개 설정 사이의 편차라 데이터셋 난이도 차이가 섞여 있다. "
            "방법 간 비교는 3절의 짝지은 비교를 볼 것.", "",
            md_table(hdr, [[lab, how] + [cell(m, c, h) for c, h in cols] for m, lab, how in rows_all], left=2), ""]
    for ds, dname in DATASETS:
        out += [f"## 2. 데이터셋별: {dname} (9개 설정)", "",
                md_table(hdr, [[lab, how] + [cell(m, c, h, ds) for c, h in cols] for m, lab, how in rows_all], left=2), ""]

    out += ["## 3. Hypercone vs 기준 방법: 짝지은 비교 (같은 설정끼리, 18쌍)", "",
            "차이 = Hypercone − 기준 방법 (%p). 우세 = Hypercone이 나은 설정 수. p = Wilcoxon 부호순위 검정(양측). "
            "F1은 둘 다 보정 판정으로 비교.", ""]
    rows = []
    for m, lab in [("msp", "MSP"), ("energy", "Energy"), ("knn", "kNN"), ("adb", "ADB"), ("maha", "Mahalanobis"),
                   ("cq_msp", "cosine MSP (같은 신경망)"), ("cq_maha", "Mahalanobis (같은 신경망)"),
                   ("cq_knn", "kNN (같은 신경망)")]:
        r = [lab]
        for c, h in cols[:3]:
            a, b = values(R, "cone_q", c), values(R, m, c)
            d = a - b
            r += [f"{d.mean():+.2f}", f"{wins(a, b, h)}/18", f"{wilcoxon(a, b).pvalue:.3f}"]
        rows.append(r)
    out += [md_table(["기준 방법", "AUROC 차이", "우세", "p", "FPR95 차이", "우세", "p", "F1 차이", "우세", "p"], rows), ""]

    v = {m: values(R, m, "auroc").mean() for m in ("msp", "cq_msp", "cone_q")}
    out += ["## 4. MSP 대비 AUROC 이득의 분해", "",
            md_table(["단계", "AUROC", "이전 단계 대비"],
                     [["MSP (일반 softmax 학습)", f"{v['msp']:.2f}", "–"],
                      ["cosine MSP (학습만 cosine softmax로)", f"{v['cq_msp']:.2f}", f"{v['cq_msp'] - v['msp']:+.2f}"],
                      ["Hypercone (같은 신경망, 판정만 원뿔로)", f"{v['cone_q']:.2f}", f"{v['cone_q'] - v['cq_msp']:+.2f}"]]), ""]

    out += ["## 5. 가상 이상치 종류별 (경계 그대로 판정)", "",
            md_table(["가상 이상치", "공간", "AUROC", "FPR95", "F1 전체", "F1 vs 없음", "F1 우세"],
                     [[lab, sp, fmt(m, "auroc"), fmt(m, "fpr95"), fmt(m + "_raw", "f1_all"),
                       f"{(values(R, m + '_raw', 'f1_all') - values(R, 'cone_q_raw', 'f1_all')).mean():+.2f}",
                       "–" if m == "cone_q" else
                       f"{wins(values(R, m + '_raw', 'f1_all'), values(R, 'cone_q_raw', 'f1_all'), True)}/18"]
                      for m, lab, sp in [("cone_q", "없음", "–"), ("cone_gauss", "Gaussian (VOS 방식)", "feature"),
                                         ("cone_vmf", "vMF (자르지 않음)", "feature"),
                                         ("cone_tvmf", "truncated vMF", "feature"),
                                         ("in_gauss", "Gaussian (VOS 방식)", "입력"),
                                         ("in_tvmf", "truncated vMF", "입력")]], left=2), ""]
    if margin_stats:
        out += ["## 6. 판정 결과 (known 75%, seed 0, 이상치 없는 Hypercone은 경계 그대로)", "",
                md_table(["데이터셋", "방법", "새어 들어온 관련 없는 문장", "잘못 거부한 학습 문장"],
                         [[d, m, f"{a:.1f}%", f"{b:.1f}%"] for d, m, a, b in margin_stats], left=2), ""]
    with open("its/results/summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--no_margin", action="store_true")
    a = p.parse_args()
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
    os.makedirs("its/figures", exist_ok=True)
    R = load_rows()
    overall_figure(R)
    paired_figure(R)
    gain_figure(R)
    stats = None if a.no_margin else margin_figure()
    summary_md(R, stats)
    print("saved its/figures/summary_*.png, its/results/summary.md")
