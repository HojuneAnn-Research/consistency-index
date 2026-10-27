"""Figures for the drone example: rule comparison (mean width error) and a GIF of the crack points kept by each rule."""
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


def gif(data_dir=f"{ROOT}/data/drone", pad=0.03, px=0.0005, width_px=1200):
    """One frame per rule, in the order of the comparison figure; one panel per segmentation."""
    import sys; sys.path.insert(0, os.path.dirname(__file__))
    from ci_pipeline import load, load_mask, mask_labeler, consistency_index, fusion_rules
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

    def paint(img, xyz, rgb):
        ok, cx, cy = raster(xyz)                      # points are ~1 mm apart, raster at 0.5 mm: paint 3x3 blocks
        for d in range(3):
            for e in range(3):
                img[np.clip(cy + d - 1, 0, H - 1), np.clip(cx + e - 1, 0, W - 1)] = rgb if np.ndim(rgb) == 1 else rgb[ok]
    base = np.full((H, W, 3), 255, np.uint8); paint(base, bg_xyz, bg_rgb)
    base = (0.55 * base + 0.45 * 255).astype(np.uint8)

    # crack points kept by every rule
    views, P, vis, _ = load(data_dir); normal = A[2]
    keep = {}
    for seg in ("manual", "deepcrack", "masonry"):
        lab = mask_labeler(P, {v: load_mask(data_dir, seg, v) for v in views})
        cand, ci = consistency_index(views, P, vis, lab)
        keep[seg] = (cand, {f"CI {k}": ci >= k for k in (1, 2, 3, 4)} | fusion_rules(views, P, vis, lab, cand, normal))
    rows = json.load(open(f"{OUT}/widths.json"))
    title = {"manual": "Manual crack marking", "deepcrack": "DeepCrack", "masonry": "Masonry crack segmentation"}
    frames_def = [("CI 1", "CI 1 (plain reprojection)", "Consistency Index 1: crack in at least 1 photo (plain reprojection)"),
                  ("CI 2", "CI 2", "Consistency Index 2: crack in at least 2 photos"),
                  ("CI 3", "CI 3", "Consistency Index 3: crack in at least 3 photos"),
                  ("CI 4", "CI 4", "Consistency Index 4: crack in at least 4 photos"),
                  ("best view (closest)", "best view (closest)", "Fixed rule: label of the closest photo only"),
                  ("best view (most frontal)", "best view (most frontal)", "Fixed rule: label of the most frontal photo only"),
                  ("fraction >= 1/3", "fraction >= 1/3", "Fixed rule: crack in at least 1/3 of the photos that see the point"),
                  ("fraction >= 1/2", "fraction >= 1/2", "Fixed rule: crack in at least 1/2 of the photos that see the point"),
                  ("fraction >= 2/3", "fraction >= 2/3", "Fixed rule: crack in at least 2/3 of the photos that see the point")]

    def font(size):
        for f in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "arialbd.ttf"):
            try: return ImageFont.truetype(f, size)
            except OSError: pass
        return ImageFont.load_default()
    f_head, f_panel, f_err = font(24), font(19), font(19)
    gap, pw = 12, (width_px - 2 * 12) // 3; ph = int(H * pw / W); top, head, foot = 52, 34, 36
    frames = []
    for key, rule_name, caption in frames_def:
        fr_img = Image.new("RGB", (width_px, top + head + ph + foot), "white"); dr = ImageDraw.Draw(fr_img)
        dr.text((10, 12), caption, fill=(0, 0, 0), font=f_head)
        for i, seg in enumerate(("manual", "deepcrack", "masonry")):
            x0 = i * (pw + gap); cand, sets = keep[seg]
            dr.text((x0 + 4, top + 6), title[seg], fill=(0, 0, 0), font=f_panel)
            if key in sets:
                img = base.copy(); paint(img, cand[sets[key]], np.array([210, 0, 0], np.uint8))
                fr_img.paste(Image.fromarray(img).resize((pw, ph), Image.LANCZOS), (x0, top + head))
                r = next(r for r in rows if r["segmentation"] == title[seg] and r["rule"] == rule_name)
                txt = "mean width error: " + ("n/a" if r["mean_error_mm"] is None else f"{r['mean_error_mm']:+.2f} mm".replace("-", "\u2212"))
                dr.text((x0 + 4, top + head + ph + 8), txt, fill=(0, 0, 0), font=f_err)
            else:
                dr.rectangle([x0, top + head, x0 + pw, top + head + ph], fill=(238, 238, 238))
                dr.text((x0 + 20, top + head + ph // 2 - 12), "not run (masks not kept)", fill=(90, 90, 90), font=f_panel)
        frames.append(fr_img)
    frames[0].save(f"{OUT}/consistency_index_drone.gif", save_all=True, append_images=frames[1:], duration=1600, loop=0)
    frames[0].save(f"{OUT}/ci1_drone.png"); frames[1].save(f"{OUT}/ci2_drone.png")


if __name__ == "__main__":
    comparison(); gif(); print("figures written to", OUT)
