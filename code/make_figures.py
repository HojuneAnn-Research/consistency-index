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


def gif(data_dir=f"{ROOT}/data/drone", pad=0.03, px=0.0005, width_px=1500, min_show=3):
    """One frame per rule, in the order of the comparison figure; one panel per segmentation.
    Under each panel: width and error of every part and the mean width error. A part measured at fewer than
    `need` (8) of the 10 stations is shown with its number of stations and is not used in the mean error;
    a part with fewer than `min_show` stations is shown as a dash."""
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
    f_head, f_panel, f_txt, f_note, f_lab = font(26), font(21), font(19), font(17), font(17)
    ref = [geo["reference_width_mm"][p] for p in ("Part1", "Part2", "Part3")]; need = geo["width_method"]["need"]
    gap, pw = 12, (width_px - 2 * 12) // 3; ph = int(H * pw / W)
    top, head, line_h = 56, 36, 27; foot = 5 * line_h + 12; note_h = 56
    # part labels: next to the middle of each centreline, offset across it
    labels = []
    for k, (name, l) in enumerate(geo["centrelines_uv"].items()):
        l = np.array(l); m = len(l) // 2; a, b = l[max(m - 2, 0)], l[min(m + 2, len(l) - 1)]
        t = (b - a) / np.linalg.norm(b - a); n = np.array([-t[1], t[0]])
        labels.append((f"Part {k + 1}", l[m], n))

    def to_panel(uv):
        c = (uv - lo) / px; return np.array([c[0] * pw / W, (H - 1 - c[1]) * ph / H])

    def part_line(r, i):
        n, w = r["stations"][i], r["width_all_stations_mm"][i]
        if w is None or n < min_show:
            return f"Part {i + 1}   \u2013"
        s = f"Part {i + 1}   {w:.2f} mm   ({w - ref[i]:+.2f})"
        if n < need:
            s += f"   {n}/10 st.*"
        return s.replace("-", "\u2212")

    note = ["Crack-ruler widths: Part 1 3.00 mm, Part 2 2.48 mm, Part 3 1.25 mm; errors in brackets.",
            "* measured at fewer than 8 of the 10 stations and not used in the mean width error;   \u2013 fewer than 3 stations."]
    frames = []
    for key, rule_name, caption in frames_def:
        fr_img = Image.new("RGB", (width_px, top + head + ph + foot + note_h), "white"); dr = ImageDraw.Draw(fr_img)
        dr.text((10, 13), caption, fill=(0, 0, 0), font=f_head)
        for i, seg in enumerate(("manual", "deepcrack", "masonry")):
            x0 = i * (pw + gap); cand, sets = keep[seg]
            r = next(r for r in rows if r["segmentation"] == title[seg] and r["rule"] == rule_name)
            dr.text((x0 + 4, top + 6), title[seg], fill=(0, 0, 0), font=f_panel)
            img = base.copy(); paint(img, cand[sets[key]], np.array([210, 0, 0], np.uint8))
            panel = Image.fromarray(img).resize((pw, ph), Image.LANCZOS); pd = ImageDraw.Draw(panel)
            for txt, mid, nrm in labels:
                c = to_panel(mid); d = np.array([nrm[0], -nrm[1]]) * 30
                c = c + d if 0 <= (c + d)[1] <= ph - 20 else c - d
                bw = pd.textlength(txt, font=f_lab)
                pd.rectangle([c[0] - bw / 2 - 3, c[1] - 11, c[0] + bw / 2 + 3, c[1] + 11], fill=(255, 255, 255))
                pd.text((c[0] - bw / 2, c[1] - 10), txt, fill=(60, 60, 60), font=f_lab)
            fr_img.paste(panel, (x0, top + head))
            y = top + head + ph + 8
            for j in range(3):
                dr.text((x0 + 4, y), part_line(r, j), fill=(0, 0, 0), font=f_txt); y += line_h
            me = "n/a" if r["mean_error_mm"] is None else f"{r['mean_error_mm']:+.2f} mm".replace("-", "\u2212")
            dr.text((x0 + 4, y), f"Mean width error   {me}", fill=(0, 0, 0), font=f_txt); y += line_h
            dr.text((x0 + 4, y), f"Crack points   {r['points']}", fill=(0, 0, 0), font=f_txt)
        for j, t in enumerate(note):
            dr.text((10, top + head + ph + foot + 4 + 24 * j), t, fill=(70, 70, 70), font=f_note)
        frames.append(fr_img)
    frames[0].save(f"{OUT}/consistency_index_drone.gif", save_all=True, append_images=frames[1:], duration=1600, loop=0)
    frames[0].save(f"{OUT}/ci1_drone.png"); frames[1].save(f"{OUT}/ci2_drone.png")


if __name__ == "__main__":
    comparison(); gif(); print("figures written to", OUT)
