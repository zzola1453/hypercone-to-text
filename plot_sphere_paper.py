"""Paper version of the sphere figure: same data as plot_sphere.py, no titles or annotations,
one shared legend at the bottom.

python plot_sphere_paper.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerPatch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from plot_sphere import HERE, args, compute, cone_circle

BG, INK, RING, MUTED = "#ffffff", "#111111", "#8a8984", "#8f8e8a"
SERIES = ["#1f6fe5", "#f2591f", "#1baf7a"]          # weather, play_music, balance
BAND, REGION = "#ebebea", "#fde3d6"


class HandlerRing(HandlerPatch):
    def create_artists(self, legend, orig, x0, y0, w, h, fontsize, trans):
        c = mpatches.Circle((x0 + w / 2, y0 + h / 2), h * 0.9, fill=False, ls=(0, (3, 2)),
                            ec=RING, lw=1.0, transform=trans)
        return [c]


def main():
    a = args(out=f"{HERE}/figures/sphere_paper.png")
    D = compute(a)
    zk, zu, yk, P, th, out, r = (D[k] for k in ("zk", "zu", "yk", "P", "th", "out", "r"))

    v = P.sum(0)
    v /= np.linalg.norm(v)
    elev, azim = math.degrees(math.asin(v[2])) - 18, math.degrees(math.atan2(v[1], v[0])) + 8
    e, az = math.radians(elev), math.radians(azim)   # tilted so the lower cone (balance) faces the viewer
    v = np.array([math.cos(e) * math.cos(az), math.cos(e) * math.sin(az), math.sin(e)])
    front = lambda z: z @ v > 0

    plt.rcParams.update({"font.family": "Georgia", "axes.unicode_minus": False})
    fig = plt.figure(figsize=(16, 7.2), facecolor=BG)

    # (left) whole sphere
    ax = fig.add_axes([0.08, 0.15, 0.44, 0.85], projection="3d", facecolor=BG)
    u, w = np.mgrid[0:2 * np.pi:60j, 0:np.pi:30j]
    ax.plot_surface(np.cos(u) * np.sin(w), np.sin(u) * np.sin(w), np.cos(w), color="#e6e6e4", alpha=0.22,
                    linewidth=0, shade=False)
    t = np.linspace(0, 2 * np.pi, 200)
    for lat in np.radians([-60, -30, 0, 30, 60]):
        ax.plot(np.cos(lat) * np.cos(t), np.cos(lat) * np.sin(t), np.full_like(t, np.sin(lat)),
                color="#cfcfcc", lw=0.6, zorder=1)

    def scatter(z, color, size, marker="o", lw=0.0):
        f = front(z)
        for mask, alpha in ((~f, 0.15), (f, 0.95)):
            if mask.any():
                ax.scatter(*z[mask].T, s=size, c=color, marker=marker, alpha=alpha, linewidths=lw,
                           edgecolors=BG if marker == "o" else None, depthshade=False)

    scatter(zu * 1.003, MUTED, 16)
    for k in range(3):
        c = cone_circle(P[k], float(th[k])) * 1.005
        if P[k] @ v > 0:
            ax.add_collection3d(Poly3DCollection([c], facecolor=SERIES[k], alpha=0.16, edgecolor="none"))
        f = front(c)
        ax.plot(*np.where(f, c.T, np.nan), color=SERIES[k], lw=1.8, zorder=5)
        ax.plot(*np.where(~f, c.T, np.nan), color=SERIES[k], lw=1.0, ls=(0, (3, 3)), alpha=0.35, zorder=2)
        scatter(zk[yk == k] * 1.006, SERIES[k], 28)
    scatter(out * 1.004, INK, 26, marker="x", lw=1.0)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1.0, 1.0), ax.set_ylim(-1.0, 1.0), ax.set_zlim(-1.0, 1.0)
    ax.set_axis_off()

    # (right) close-up of one cone, azimuthal equidistant around its axis: radius = true angle (deg)
    k = a.focus
    mu = P[k]
    e1 = np.cross(mu, [1.0, 0, 0] if abs(mu[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(mu, e1)

    def polar(z):
        ang = np.degrees(np.arccos(np.clip(z @ mu, -1, 1)))
        phi = np.arctan2(z @ e2, z @ e1)
        return ang * np.cos(phi), ang * np.sin(phi), ang

    r0, r1 = math.degrees(float(r[k])), math.degrees(float(r[k]) + a.delta)
    lim = r1 + 14
    bx = fig.add_axes([0.44, 0.17, 0.46, 0.83], facecolor=BG)
    t = np.linspace(0, 2 * np.pi, 400)
    bx.fill(np.r_[r1 * np.cos(t), r0 * np.cos(t[::-1])], np.r_[r1 * np.sin(t), r0 * np.sin(t[::-1])],
            color=BAND, zorder=0, lw=0)
    bx.fill(r0 * np.cos(t), r0 * np.sin(t), color=REGION, zorder=1, lw=0)
    for ring in range(10, int(lim) + 1, 10):
        bx.plot(ring * np.cos(t), ring * np.sin(t), color=RING, lw=0.8, ls=(0, (4, 3)), zorder=2)
        bx.text(0, ring + 0.6, f"{ring}°", color=INK, fontsize=15, ha="center", va="bottom", zorder=7,
                bbox=dict(boxstyle="square,pad=0.1", fc=BG, ec="none", alpha=0.0))
    bx.plot(r0 * np.cos(t), r0 * np.sin(t), color=SERIES[k], lw=2.4, zorder=4)
    for zz, color, size, marker, lw in ((zu, MUTED, 22, "o", 0), (zk[yk == k], SERIES[k], 36, "o", 0),
                                        (out[k * 40:(k + 1) * 40], INK, 46, "x", 1.1)):
        x, y, ang = polar(zz)
        m = ang < lim
        bx.scatter(x[m], y[m], s=size, c=color, marker=marker, linewidths=lw,
                   edgecolors=BG if marker == "o" else None, zorder=5 if marker == "x" else 3)
    bx.scatter([0], [0], s=160, c=INK, marker="+", linewidths=1.6, zorder=6)
    bx.set_xlim(-lim, lim), bx.set_ylim(-lim, lim), bx.set_aspect("equal")
    bx.set_axis_off()

    # shared legend
    dot = lambda c, lab: Line2D([], [], marker="o", ls="", color=c, markersize=11, label=lab)
    handles = [dot(SERIES[0], "weather"), dot(SERIES[1], "play_music"), dot(SERIES[2], "balance"),
               dot(MUTED, "unknown sentence"),
               Line2D([], [], marker="x", ls="", color=INK, markersize=11, markeredgewidth=1.4,
                      label="virtual outlier\n(tvMF, outside cone)"),
               Line2D([], [], marker="+", ls="", color=INK, markersize=14, markeredgewidth=1.6,
                      label="class center\n(mean direction)"),
               Line2D([], [], color=SERIES[1], lw=2.6, label="cone boundary\n(95% quantile radius)"),
               mpatches.Patch(color=REGION, label="decision region\n(inside boundary)"),
               mpatches.Patch(color=BAND, label="outlier sampling band\n(just outside boundary)"),
               mpatches.Circle((0, 0), 1, label="reference angle\n(10°, 20°, 30°)")]
    leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=5,
                     frameon=True, fancybox=True, fontsize=13, labelcolor=INK, handlelength=1.6,
                     columnspacing=2.2, borderpad=0.9, labelspacing=1.0,
                     handler_map={mpatches.Circle: HandlerRing()})
    leg.get_frame().set_edgecolor("#d9d9d6")
    leg.get_frame().set_linewidth(1.0)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=200, facecolor=BG, bbox_inches="tight")
    print("saved", a.out)


if __name__ == "__main__":
    main()
