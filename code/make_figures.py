"""Figures for the drone example: rule comparison (mean width error) and a GIF of the crack kept at CI 1-5."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

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


def gif(data_dir=f"{ROOT}/data/drone", pad=0.03, px=0.0005):
    geo = json.load(open(f"{data_dir}/geometry.json")); fr = geo["frame"]
    o, A = np.array(fr["origin"]), np.array(fr["axes"])
    pts = np.load(f"{data_dir}/visible_points_crop.npz")
    bg_xyz, bg_rgb = pts["xyz_0003"], pts["rgb_0003"]
    lines = np.vstack([np.array(l) for l in geo["centrelines_uv"].values()])
    lo, hi = lines.min(0) - pad, lines.max(0) + pad
    W, H = int((hi[0] - lo[0]) / px) + 1, int((hi[1] - lo[1]) / px) + 1

    def raster(xyz):
        uv = ((xyz - o) @ A.T)[:, :2]; ok = np.all((uv >= lo) & (uv <= hi), 1)
        c = ((uv[ok] - lo) / px).astype(int); return ok, c[:, 0], H - 1 - c[:, 1]
    ok, cx, cy = raster(bg_xyz); base = np.full((H, W, 3), 255, np.uint8)
    for d in range(3):                                  # points are ~1 mm apart, raster at 0.5 mm: paint 3x3 blocks
        for e in range(3):
            base[np.clip(cy + d - 1, 0, H - 1), np.clip(cx + e - 1, 0, W - 1)] = bg_rgb[ok]
    base = (0.55 * base + 0.45 * 255).astype(np.uint8)
    clouds = {seg: np.loadtxt(f"{OUT}/{seg}_ci_points.csv", delimiter=",", skiprows=1) for seg in ("manual", "deepcrack", "masonry")}
    names = {"manual": "Manual marking", "deepcrack": "DeepCrack", "masonry": "Masonry network"}
    try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except OSError: font = ImageFont.load_default()
    frames = []
    for k in range(1, 6):
        panels = []
        for seg, cl in clouds.items():
            img = base.copy(); sel = cl[:, 3] >= k; ok2, x2, y2 = raster(cl[sel, :3])
            for d in range(3):
                for e in range(3):
                    img[np.clip(y2 + d - 1, 0, H - 1), np.clip(x2 + e - 1, 0, W - 1)] = (210, 0, 0)
            im = Image.fromarray(img); dr = ImageDraw.Draw(im); dr.text((8, 6), names[seg], fill=(0, 0, 0), font=font)
            panels.append(im)
        f = Image.new("RGB", (sum(p.width for p in panels) + 20, H + 40), "white")
        x = 0
        for p in panels: f.paste(p, (x, 40)); x += p.width + 10
        ImageDraw.Draw(f).text((8, 8), f"Consistency Index {k}", fill=(0, 0, 0), font=font)
        frames.append(f)
    s = 900 / frames[0].width
    frames = [fr_.resize((900, int(fr_.height * s)), Image.LANCZOS) for fr_ in frames]
    frames[0].save(f"{OUT}/consistency_index_drone.gif", save_all=True, append_images=frames[1:], duration=1200, loop=0)
    frames[0].save(f"{OUT}/ci1_drone.png"); frames[1].save(f"{OUT}/ci2_drone.png")


if __name__ == "__main__":
    comparison(); gif(); print("figures written to", OUT)
