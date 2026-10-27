"""Figures for the drone example: rule comparison (mean width error) and a GIF of the crack points kept by each rule."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/results/drone"
COL = {"Manual crack marking": ("#0072B2", "o"), "DeepCrack": ("#D55E00", "s"), "Masonry crack segmentation": ("#009E73", "^")}
ORDER = ["CI 1 (plain reprojection)", "CI 2", "CI 3", "CI 4", "best view (closest)", "best view (most frontal)",
         "fraction >= 1/3", "fraction >= 1/2", "fraction >= 2/3"]
XLAB = ["CI 1\n(plain)", "CI 2", "CI 3", "CI 4", "Best view\n(closest)", "Best view\n(frontal)",
        "≥ 1/3\nof views", "≥ 1/2\nof views", "≥ 2/3\nof views"]


def comparison():
    rows = json.load(open(f"{OUT}/widths.json"))
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in"})
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.set_axisbelow(True); ax.grid(True, color="#c9c9c9", linewidth=0.45, linestyle=(0, (1, 2)))
    ax.axhspan(-0.5, 0.5, color="#ececec", zorder=0); ax.axhline(0, color="#555555", linewidth=0.8)
    ax.axvline(3.5, color="#888888", linewidth=0.8, linestyle="--")
    for k, (seg, (c, m)) in enumerate(COL.items()):
        y = [next((r["mean_error_mm"] for r in rows if r["segmentation"] == seg and r["rule"] == name), None) for name in ORDER]
        y = [np.nan if v is None else v for v in y]
        x = np.arange(len(ORDER)) + (k - 1) * 0.18
        ax.plot(x, y, linestyle="none", marker=m, markersize=5.5, color=c, label=seg, zorder=3)
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(XLAB, fontsize=8)
    ax.set_xlim(-0.6, len(ORDER) - 0.4); ax.set_ylim(-2.5, 1.5); ax.set_yticks(np.arange(-2.5, 1.51, 0.5))
    ax.set_ylabel("Mean width error (mm)")
    ax.text(1.5, -2.38, "Consistency Index", ha="center", fontsize=8); ax.text(6.0, -2.38, "Fixed fusion rules", ha="center", fontsize=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, 0.99), frameon=True, fontsize=8, framealpha=1, edgecolor="black", handletextpad=0.3, borderpad=0.4)
    fig.tight_layout(); fig.savefig(f"{OUT}/rule_comparison.png", dpi=200); plt.close(fig)


def gif(data_dir=f"{ROOT}/data/drone", across=0.010, min_show=3):
    """One frame per rule, in the order of the comparison figure. Rows: the three 2D segmentations; columns: the three
    crack parts, each straightened along its centreline (across-crack scale exaggerated). Red points are the crack
    points the rule keeps for that part, triangles the 10 stations (filled: measured). A part measured at fewer than
    `need` (8) stations is marked * and not used in the mean error; one with fewer than `min_show` stations is a dash."""
    import sys; sys.path.insert(0, os.path.dirname(__file__))
    from ci_pipeline import (load, load_mask, mask_labeler, consistency_index, fusion_rules, part_points, stations,
                             _project_polyline)
    geo = json.load(open(f"{data_dir}/geometry.json")); fr = geo["frame"]; c = geo["width_method"]
    o, A = np.array(fr["origin"]), np.array(fr["axes"])
    lines = {p: np.array(l) for p, l in geo["centrelines_uv"].items()}
    parts = list(lines); ref = [geo["reference_width_mm"][p] for p in parts]; need = c["need"]
    xmax, xstep = {"Part1": 320, "Part2": 270, "Part3": 80}, {"Part1": 80, "Part2": 90, "Part3": 20}

    # background: points of one photo with their colour, straightened along each part
    pts = np.load(f"{data_dir}/visible_points_crop.npz")
    uvw = (pts["xyz_0003"] - o) @ A.T; rgb = 0.55 * pts["rgb_0003"] / 255.0 + 0.45
    on = np.abs(uvw[:, 2]) < c["wtol"]; bg = {}
    for p in parts:
        s, d, _, L = _project_polyline(uvw[on, :2], lines[p])
        m = (np.abs(d) <= across) & (s > 0) & (s < L)
        bg[p] = (s[m] * 1000, d[m] * 1000, rgb[on][m])

    # crack points kept by every rule
    views, P, vis, _ = load(data_dir); normal = A[2]
    keep = {}
    for seg in ("manual", "deepcrack", "masonry"):
        lab = mask_labeler(P, {v: load_mask(data_dir, seg, v) for v in views})
        cand, ci = consistency_index(views, P, vis, lab)
        keep[seg] = (cand, {f"CI {k}": ci >= k for k in (1, 2, 3, 4)} | fusion_rules(views, P, vis, lab, cand, normal))
    rows = json.load(open(f"{OUT}/widths.json"))
    title = {"manual": "Manual crack\nmarking", "deepcrack": "DeepCrack", "masonry": "Masonry crack\nsegmentation"}
    name = {"manual": "Manual crack marking", "deepcrack": "DeepCrack", "masonry": "Masonry crack segmentation"}
    frames_def = [("CI 1", "CI 1 (plain reprojection)", "Consistency Index 1: crack in at least 1 photo (plain reprojection)"),
                  ("CI 2", "CI 2", "Consistency Index 2: crack in at least 2 photos"),
                  ("CI 3", "CI 3", "Consistency Index 3: crack in at least 3 photos"),
                  ("CI 4", "CI 4", "Consistency Index 4: crack in at least 4 photos"),
                  ("best view (closest)", "best view (closest)", "Fixed rule: label of the closest photo only"),
                  ("best view (most frontal)", "best view (most frontal)", "Fixed rule: label of the most frontal photo only"),
                  ("fraction >= 1/3", "fraction >= 1/3", "Fixed rule: crack in at least 1/3 of the photos that see the point"),
                  ("fraction >= 1/2", "fraction >= 1/2", "Fixed rule: crack in at least 1/2 of the photos that see the point"),
                  ("fraction >= 2/3", "fraction >= 2/3", "Fixed rule: crack in at least 2/3 of the photos that see the point")]

    def part_text(r, j):
        n, w = r["stations"][j], r["width_all_stations_mm"][j]
        if w is None or n < min_show:
            return "\u2013"
        t = f"{w:.2f} mm ({w - ref[j]:+.2f})" + (f"   {n}/10 stations*" if n < need else "")
        return t.replace("-", "\u2212")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.linewidth": 0.8,
                         "xtick.direction": "in", "ytick.direction": "in"})
    W, H = 18.0, 6.0; left, right, gapx = 0.12, 0.995, 0.026; aw = (right - left - 2 * gapx) / 3
    ah, top, gapy = 0.215, 0.98, 0.035
    frames = []
    for key, rule_name, caption in frames_def:
        fig = plt.figure(figsize=(W, H), dpi=100)
        for i, seg in enumerate(("manual", "deepcrack", "masonry")):
            cand, sets = keep[seg]; xyz = cand[sets[key]]
            r = next(r for r in rows if r["segmentation"] == name[seg] and r["rule"] == rule_name)
            y0 = top - ah - i * (ah + gapy)
            me = "n/a" if r["mean_error_mm"] is None else f"{r['mean_error_mm']:+.2f} mm".replace("-", "\u2212")
            fig.text(0.006, y0 + ah * 0.62, title[seg], fontsize=13, fontweight="bold", va="center")
            fig.text(0.006, y0 + ah * 0.18, f"mean error\n{me}", fontsize=12, va="center")
            for j, p in enumerate(parts):
                ax = fig.add_axes([left + j * (aw + gapx), y0, aw, ah])
                ax.set_axisbelow(True); ax.grid(True, color="#c9c9c9", linewidth=0.45, linestyle=(0, (1, 2)))
                bs, bd, bc = bg[p]; ax.scatter(bs, bd, s=3, c=bc, linewidths=0, zorder=1)
                s, d, L, _ = part_points(xyz, p, geo)
                m = np.abs(d) <= across
                ax.scatter(s[m] * 1000, d[m] * 1000, s=11, c="#D00000", linewidths=0, zorder=3)
                st, L = stations(xyz, p, geo); done = {k for k, _, _ in st}
                e = np.linspace(c["margin"] * L, (1 - c["margin"]) * L, c["nseg"] + 1); mid = (e[:-1] + e[1:]) / 2 * 1000
                for k, x in enumerate(mid):
                    ax.plot(x, -8.6, marker="^", markersize=6.5, color="#222222",
                            markerfacecolor="#222222" if k in done else "white", markeredgewidth=0.9, zorder=4)
                ax.set_xlim(0, xmax[p]); ax.set_xticks(np.arange(0, xmax[p] + 1, xstep[p]))
                ax.set_ylim(-across * 1000, across * 1000); ax.set_yticks([-10, 0, 10])
                if i < 2: ax.set_xticklabels([])
                else: ax.set_xlabel(f"Part {j + 1} ({ref[j]:.2f} mm): distance along the part (mm)", fontsize=12)
                if j > 0: ax.set_yticklabels([])
                ax.text(0.012, 0.93, f"Part {j + 1}   " + part_text(r, j), transform=ax.transAxes, fontsize=12,
                        fontweight="bold", va="top", ha="left", zorder=5,
                        bbox=dict(boxstyle="square,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85))
        fig.text(0.006, 0.105, caption, fontsize=15, fontweight="bold", va="center")
        fig.text(0.006, 0.055, "Each panel is one part straightened along its centreline; vertical axis: offset across the "
                 "crack (mm, exaggerated).", fontsize=11, va="center", color="#444444")
        fig.text(0.006, 0.02, "Errors against the crack-ruler width in brackets.   * measured at fewer than 8 of the 10 "
                 "stations and not used in the mean error;   \u2013 fewer than 3 stations.", fontsize=11, va="center",
                 color="#444444")
        hd = [Line2D([], [], linestyle="none", marker="o", markersize=5, color="#D00000", label="Crack points kept by the rule"),
              Line2D([], [], linestyle="none", marker="^", markersize=7, color="#222222", label="Station measured"),
              Line2D([], [], linestyle="none", marker="^", markersize=7, color="#222222", markerfacecolor="white",
                     label="Station not measured")]
        lg = fig.legend(handles=hd, loc="center right", bbox_to_anchor=(0.995, 0.105), ncol=3, frameon=True, fontsize=11,
                        edgecolor="black", framealpha=1, handletextpad=0.3, columnspacing=1.2)
        lg.get_frame().set_linewidth(0.8)
        fig.canvas.draw(); frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())); plt.close(fig)
    frames[0].save(f"{OUT}/consistency_index_drone.gif", save_all=True, append_images=frames[1:], duration=1800, loop=0)
    frames[0].save(f"{OUT}/ci1_drone.png"); frames[1].save(f"{OUT}/ci2_drone.png")


if __name__ == "__main__":
    comparison(); gif(); print("figures written to", OUT)
