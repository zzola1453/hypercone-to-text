"""Result figures from its/results/{main,input}.json.

figures/results_main.png:     AUROC / FPR95 / F1(all) vs known ratio, per dataset (mean ± std over 3 seeds)
figures/results_ablation.png: F1(all) by virtual-outlier type, accepted without threshold calibration

python -m its.plot_results
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # validated categorical slots 1-4 (adjacent)
MARKERS = ["o", "s", "^", "D"]
METHODS = [("in_tvmf_raw", "Hypercone + tvMF 가상 이상치"), ("cone_q_raw", "Hypercone"),
           ("msp", "Softmax 확신도 (MSP)"), ("adb", "ADB (재구현)")]
DATASETS = [("clinc", "CLINC150"), ("banking", "BANKING77")]
METRICS = [("auroc", "AUROC (높을수록 좋음)"), ("fpr95", "FPR95: 새어 들어온 unknown 비율 (낮을수록 좋음)"),
           ("f1_all", "F1 전체 (높을수록 좋음)")]


def rows():
    main = json.load(open("its/results/main.json"))["rows"]
    inp = [dict(r, method=r["method"].replace("cone_", "in_")) for r in json.load(open("its/results/input.json"))["rows"]]
    return main + inp


def stat(R, ds, m, c, ratio):
    v = [r[c] * 100 for r in R if r["dataset"] == ds and r["method"] == m and r["ratio"] == ratio]
    return np.mean(v), np.std(v)


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def main_figure(R):
    ratios = [0.25, 0.5, 0.75]
    fig, axes = plt.subplots(len(DATASETS), len(METRICS), figsize=(14, 7.6), facecolor=SURFACE)
    for i, (ds, dname) in enumerate(DATASETS):
        for j, (c, cname) in enumerate(METRICS):
            ax = axes[i, j]
            style(ax)
            for k, (m, label) in enumerate(METHODS):
                mu, sd = zip(*[stat(R, ds, m, c, q) for q in ratios])
                x = np.arange(3) + (k - 1.5) * 0.06               # small dodge so error bars do not overlap
                ax.errorbar(x, mu, yerr=sd, color=SERIES[k], lw=2, marker=MARKERS[k], markersize=7,
                            markeredgecolor=SURFACE, markeredgewidth=1.5, capsize=0, elinewidth=1.2, label=label)
            ax.set_xticks(range(3), ["25%", "50%", "75%"])
            if i == 0:
                ax.set_title(cname, color=INK, fontsize=11, loc="left")
            if i == len(DATASETS) - 1:
                ax.set_xlabel("학습에 쓴 클래스 비율 (known)", color=INK2, fontsize=9)
            if j == 0:
                ax.set_ylabel(dname, color=INK, fontsize=12)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=10, labelcolor=INK,
               bbox_to_anchor=(0.5, 0.94))
    fig.text(0.5, 0.965, "관련 없는 문장 거르기 성능: 18개 설정 (2 데이터셋 × 3 비율 × seed 3)", ha="center",
             fontsize=14, color=INK)
    fig.text(0.5, 0.01, "점 = seed 3개 평균, 세로 막대 = 표준편차. Hypercone 계열은 거부 기준값 보정 없이 원뿔 경계 그대로 판정, "
             "MSP·ADB는 validation known 95% 통과 기준. 임베딩 all-mpnet-base-v2 고정.", ha="center", fontsize=9, color=INK2)
    fig.subplots_adjust(top=0.84, bottom=0.11, hspace=0.3, wspace=0.22)
    fig.savefig("its/figures/results_main.png", dpi=200, facecolor=SURFACE)


def ablation_figure(R):
    bars = [("cone_gauss_raw", "Gaussian 저확률 (VOS 방식)", "feature"), ("cone_vmf_raw", "vMF (자르지 않음)", "feature"),
            ("cone_tvmf_raw", "truncated vMF (원뿔 밖 띠)", "feature"), ("in_gauss_raw", "Gaussian 저확률 (VOS 방식)", "input"),
            ("in_tvmf_raw", "truncated vMF (원뿔 밖 띠)", "input")]
    base = np.mean([r["f1_all"] * 100 for r in R if r["method"] == "cone_q_raw"])
    vals = [np.mean([r["f1_all"] * 100 for r in R if r["method"] == m]) for m, _, _ in bars]
    fig, ax = plt.subplots(figsize=(9.5, 4.6), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    y = np.array([0, 1, 2, 3.6, 4.6])
    colors = [SERIES[1] if "gauss" in m else SERIES[0] for m, _, _ in bars]
    # dot plot: the axis does not start at zero, which bars would misrepresent
    ax.hlines(y, 70, vals, color=GRID, lw=1)
    ax.scatter(vals, y, s=90, c=colors, edgecolors=SURFACE, linewidths=2, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 0.3, yi, f"{v:.2f}", va="center", color=INK, fontsize=10)
    ax.axvline(base, color=INK2, lw=1.2, ls=(0, (4, 3)))
    ax.text(base + 0.15, -0.75, f"가상 이상치 없음 {base:.2f}", color=INK2, fontsize=9, va="center")
    ax.set_yticks(y, [b[1] for b in bars], color=INK, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(70, 88)
    ax.set_xlabel("F1 전체, 18개 설정 평균 (%)", color=INK2, fontsize=9)
    for yy, txt in ((1, "feature 공간에서 생성"), (4.1, "입력 임베딩 공간에서 생성")):
        ax.text(70.2, yy - 1.55 if yy == 1 else yy - 1.05, txt, color=INK2, fontsize=9, fontweight="bold")
    fig.text(0.02, 0.97, "가상 이상치 종류별 F1 (거부 기준값 보정 없이 원뿔 경계 그대로 판정)", fontsize=13, color=INK, va="top")
    fig.subplots_adjust(left=0.28, right=0.96, top=0.84, bottom=0.13)
    fig.savefig("its/figures/results_ablation.png", dpi=200, facecolor=SURFACE)


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
    os.makedirs("its/figures", exist_ok=True)
    R = rows()
    main_figure(R)
    ablation_figure(R)
    print("saved its/figures/results_main.png, its/figures/results_ablation.png")
