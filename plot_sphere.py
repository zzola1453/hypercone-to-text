"""Explanatory figure: sentence embeddings, class hypercones and tvMF virtual outliers on S^2.

768-d embeddings cannot be drawn without distorting angles, so the cone model itself is trained
with a 3-d output (d=3) on three CLINC150 intents; the plotted points are its actual outputs.
The cones drawn are the HyperSphereDiff hypercones of those outputs (axis = class mean direction,
radius = 95% quantile of the training angles), and the outliers are tvMF draws in the shell just
outside them, with kappa from A_3(kappa) = cos(radius). Illustration only: the 128-d experiments
use learned prototypes and radii.

python plot_sphere.py
"""
import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from data import load
from embed import embed
from methods import OutlierSynth, train_cone
from sphere import Hypercones

HERE = os.path.dirname(os.path.abspath(__file__))

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#a9a8a2"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # validated categorical slots 1-3 (all-pairs)


def cone_circle(mu, theta, n=200):
    a = np.cross(mu, [1.0, 0, 0] if abs(mu[0]) < 0.9 else [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(mu, a)
    t = np.linspace(0, 2 * math.pi, n)
    return math.cos(theta) * mu[None] + math.sin(theta) * (np.cos(t)[:, None] * a + np.sin(t)[:, None] * b)


def args(argv=None, out=f"{HERE}/figures/sphere_embeddings.png"):
    p = argparse.ArgumentParser()
    p.add_argument("--classes", nargs=3, default=["weather", "play_music", "balance"])
    p.add_argument("--n_unknown", type=int, default=250)
    p.add_argument("--delta", type=float, default=0.25)
    p.add_argument("--focus", type=int, default=1, help="cone shown in the close-up panel")
    p.add_argument("--out", default=out)
    return p.parse_args(argv)


def compute(a):
    """Train the d=3 cone model and return everything the figures draw."""
    torch.manual_seed(0)
    rng = np.random.RandomState(0)

    data, names, oos = load("clinc")
    emb = embed("clinc", data)
    ids = [names.index(c) for c in a.classes]
    remap = {c: i for i, c in enumerate(ids)}
    ytr_all = data["train"][1]
    itr = np.flatnonzero(np.isin(ytr_all, ids))
    xtr = emb["train"][itr].cuda()
    ytr = torch.as_tensor([remap[c] for c in ytr_all[itr]]).cuda()

    net, _ = train_cone(xtr, ytr, 3, epochs=60, seed=0, outliers="tvmf", d=3, delta=a.delta)
    with torch.no_grad():
        ztr = net.feats(xtr)
        cones = Hypercones.fit(ztr, ytr, 3, quantile=0.95)
        syn = OutlierSynth("tvmf", a.delta)
        syn.refresh(ztr, ytr, cones.mu, 3, 0.95)
        out = syn.sample(torch.arange(3).repeat_interleave(40).cuda()).cpu().numpy()
        yte_all = data["test"][1]
        ik = np.flatnonzero(np.isin(yte_all, ids))
        iu = rng.choice(np.flatnonzero(~np.isin(yte_all, ids)), a.n_unknown, replace=False)
        inside = lambda z: ((cones.theta - (z @ cones.mu.T).clamp(-1, 1).acos()).max(1).values >= 0).float().mean().item()
        zk_t, zu_t = net.feats(emb["test"][ik].cuda()), net.feats(emb["test"][iu].cuda())
        acc, rej = inside(zk_t), 1 - inside(zu_t)
        zk, zu, P = zk_t.cpu().numpy(), zu_t.cpu().numpy(), cones.mu.cpu().numpy()
    yk = np.array([remap[c] for c in yte_all[ik]])
    th = cones.theta.cpu().numpy()
    print(f"theta (rad) = {np.round(th, 3)}; known test accepted {acc:.1%}, unknown rejected {rej:.1%}")

    return dict(zk=zk, zu=zu, yk=yk, P=P, th=th, out=out, r=syn.r.cpu().numpy(), acc=acc, rej=rej)


def main():
    a = args()
    D = compute(a)
    zk, zu, yk, P, th, out, acc, rej = (D[k] for k in ("zk", "zu", "yk", "P", "th", "out", "acc", "rej"))

    v = P.sum(0)
    v /= np.linalg.norm(v)                         # look from the mean direction of the three cones
    elev, azim = math.degrees(math.asin(v[2])), math.degrees(math.atan2(v[1], v[0]))
    front = lambda z: z @ v > 0

    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
    fig = plt.figure(figsize=(14.5, 7.0), facecolor=SURFACE)
    ax = fig.add_subplot(1, 2, 1, projection="3d", facecolor=SURFACE)
    u, w = np.mgrid[0:2 * np.pi:60j, 0:np.pi:30j]
    ax.plot_surface(np.cos(u) * np.sin(w), np.sin(u) * np.sin(w), np.cos(w), color="#e9e8e4", alpha=0.18,
                    linewidth=0, shade=False)
    for lat in np.radians([-60, -30, 0, 30, 60]):                 # faint graticule
        t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(lat) * np.cos(t), np.cos(lat) * np.sin(t), np.full_like(t, np.sin(lat)),
                color="#d6d5d0", lw=0.5, zorder=1)

    def scatter(z, color, size, marker="o", lw=0.0, label=None):
        f = front(z)
        for mask, alpha in ((~f, 0.18), (f, 0.95)):
            if mask.any():
                ax.scatter(*z[mask].T, s=size, c=color, marker=marker, alpha=alpha, linewidths=lw,
                           edgecolors=SURFACE if marker == "o" else None, depthshade=False,
                           label=label if alpha > 0.5 else None)

    scatter(zu * 1.003, MUTED, 14, label="관련 없는 문장 (unknown)")
    for k in range(3):
        c = cone_circle(P[k], float(th[k])) * 1.005
        f = front(c)
        seg = np.where(f, c.T, np.nan)
        ax.plot(*seg, color=SERIES[k], lw=2.0, zorder=5)
        ax.plot(*np.where(~f, c.T, np.nan), color=SERIES[k], lw=1.0, ls=(0, (3, 3)), alpha=0.35, zorder=2)
        scatter(zk[yk == k] * 1.006, SERIES[k], 26, label=a.classes[k])
        lab = P[k] * 1.28
        ax.text(*lab, a.classes[k], color=INK, fontsize=11, ha="center", va="center",
                alpha=1.0 if P[k] @ v > 0 else 0.4, zorder=10)
    scatter(out * 1.004, INK2, 18, marker="x", lw=1.0, label="가상 이상치 (tvMF, 원뿔 밖 띠)")

    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 1))
    lim = 1.05
    ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim), ax.set_zlim(-lim, lim)
    ax.set_axis_off()
    handles = [Line2D([], [], marker="o", ls="", color=SERIES[k], markersize=7, label=a.classes[k]) for k in range(3)]
    handles += [Line2D([], [], marker="o", ls="", color=MUTED, markersize=6, label="관련 없는 문장 (unknown)"),
                Line2D([], [], marker="x", ls="", color=INK2, markersize=7, label="가상 이상치 (tvMF, 원뿔 밖 띠)"),
                Line2D([], [], color=INK2, lw=2, label="원뿔 경계 (95% 분위수 반지름)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.06), ncol=2, frameon=False,
              fontsize=10, labelcolor=INK)
    ax.set_title("(a) 구 전체: 클래스마다 원뿔 하나", color=INK, fontsize=12, y=1.0)

    # (b) close-up of one cone, azimuthal equidistant around its axis: radius = true angle (deg)
    k = a.focus
    mu, r = P[k], float(D["r"][k])
    e1 = np.cross(mu, [1.0, 0, 0] if abs(mu[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(mu, e1)

    def polar(z):
        ang = np.degrees(np.arccos(np.clip(z @ mu, -1, 1)))
        phi = np.arctan2(z @ e2, z @ e1)
        return ang * np.cos(phi), ang * np.sin(phi), ang

    r0, r1 = math.degrees(r), math.degrees(r + a.delta)
    lim = r1 + 14
    bx = fig.add_subplot(1, 2, 2, facecolor=SURFACE)
    t = np.linspace(0, 2 * np.pi, 300)
    for ring in range(10, int(lim) + 1, 10):
        bx.plot(ring * np.cos(t), ring * np.sin(t), color="#e4e3df", lw=0.6, zorder=0)
        bx.text(ring * math.cos(math.radians(100)), ring * math.sin(math.radians(100)), f"{ring}°",
                color=INK2, fontsize=8, ha="center", va="bottom")
    bx.fill(np.r_[r1 * np.cos(t), r0 * np.cos(t[::-1])], np.r_[r1 * np.sin(t), r0 * np.sin(t[::-1])],
            color="#ecebe7", zorder=1, lw=0)
    bx.plot(r0 * np.cos(t), r0 * np.sin(t), color=SERIES[k], lw=2.0, zorder=4)
    for zz, color, size, marker, lw in ((zu, MUTED, 18, "o", 0), (zk[yk == k], SERIES[k], 34, "o", 0),
                                        (out[k * 40:(k + 1) * 40], INK2, 30, "x", 1.2)):
        x, y, ang = polar(zz)
        m = ang < lim
        bx.scatter(x[m], y[m], s=size, c=color, marker=marker, linewidths=lw,
                   edgecolors=SURFACE if marker == "o" else None, zorder=5 if marker == "x" else 3)
    bx.annotate("원뿔 경계 θ (이 안이면 분류)", xy=(r0 * math.cos(math.radians(-40)), r0 * math.sin(math.radians(-40))),
                xytext=(r1 - 4, -lim + 3), color=INK, fontsize=10,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    bx.annotate("가상 이상치를 뽑는 띠 [θ, θ+δ]\nκ는 A₃(κ) = cos θ 로 계산",
                xy=((r0 + r1) / 2 * math.cos(math.radians(35)), (r0 + r1) / 2 * math.sin(math.radians(35))),
                xytext=(r1 - 6, lim - 6), color=INK, fontsize=10,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    bx.text(0, 0, "+", color=INK, fontsize=12, ha="center", va="center", zorder=6)
    bx.set_xlim(-lim, lim + 14), bx.set_ylim(-lim, lim), bx.set_aspect("equal")
    bx.set_axis_off()
    bx.set_title(f"(b) {a.classes[k]} 원뿔을 위에서 본 모습 (반지름 = 중심에서의 실제 각도)", color=INK, fontsize=12)

    fig.subplots_adjust(top=0.9, bottom=0.12, wspace=0.05)
    fig.text(0.5, 0.97, "구 표면 위의 문장 임베딩과 클래스 원뿔", ha="center", fontsize=15, color=INK)
    fig.text(0.5, 0.02, f"CLINC150 의도 3개로 학습한 3차원 모델의 실제 출력 · 원뿔 = 클래스 평균 방향 + 학습 각도의 95% 분위수 · "
             f"테스트에서 known 문장 {acc:.0%} 통과, 관련 없는 문장 {rej:.0%} 거부\n"
             f"설명용 그림이며 본 실험(128차원, 학습한 원뿔)과 설정이 다릅니다. (a)의 뒷면 점은 흐리게 표시",
             ha="center", fontsize=9.5, color=INK2)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print("saved", a.out)


if __name__ == "__main__":
    main()
