"""Figures for the drone example: rule comparison (width of each part and mean width error) and a GIF of the crack points kept by each rule."""
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
        "Fraction\n≥ 1/3", "Fraction\n≥ 1/2", "Fraction\n≥ 2/3"]


def comparison():
    """Width of each part (a-c) and mean width error (d) for every rule and segmentation; the black line is the
    crack-ruler width (error 0 in d). Open markers: parts measured at only 3 to 7 of the 10 stations."""
    rows = json.load(open(f"{OUT}/widths.json")); geo = json.load(open(f"{ROOT}/data/drone/geometry.json"))
    ref = [geo["reference_width_mm"][p] for p in ("Part1", "Part2", "Part3")]; need = geo["width_method"]["need"]
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "font.size": 9,
                         "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in"})
    fig = plt.figure(figsize=(7.4, 9.0))
    left, right, ah, gap = 0.105, 0.985, 0.16, 0.055
    tops = [0.925 - k * (ah + gap) for k in range(4)]
    xf = lambda xd: left + (right - left) * (xd + 0.6) / (len(ORDER) + 0.2)
    panels = [("(a) Part 1, actual width 3.00 mm", "Width (mm)", (0, 5, 1), ref[0]),
              ("(b) Part 2, actual width 2.48 mm", "Width (mm)", (0, 4, 1), ref[1]),
              ("(c) Part 3, actual width 1.25 mm", "Width (mm)", (0, 3, 1), ref[2]),
              ("(d) Mean width error of the parts measured at 8 or more stations", "Mean error (mm)", (-3, 2, 1), 0.0)]
    x = np.arange(len(ORDER))
    for k, (sub, ylab, (y0, y1, dy), gt) in enumerate(panels):
        ax = fig.add_axes([left, tops[k] - ah, right - left, ah])
        ax.set_axisbelow(True); ax.grid(True, color="#c9c9c9", linewidth=0.45, linestyle=(0, (1, 2)))
        ax.axhspan(gt - 0.5, gt + 0.5, color="#e9e9e9", zorder=0, linewidth=0)
        ax.axhline(gt, color="black", linewidth=1.8, zorder=2)
        ax.axvline(3.5, color="#888888", linewidth=0.8, linestyle="--")
        for j, (seg, (c, m)) in enumerate(COL.items()):
            for i, name in enumerate(ORDER):
                r = next(r for r in rows if r["segmentation"] == seg and r["rule"] == name)
                if k < 3:
                    n, v = r["stations"][k], r["width_all_stations_mm"][k]
                    if v is None or n < 3: continue
                    open_ = n < need
                else:
                    v, open_ = r["mean_error_mm"], False
                    if v is None: continue
                ax.plot(x[i] + (j - 1) * 0.2, v, linestyle="none", marker=m, markersize=5.5, color=c,
                        markerfacecolor="white" if open_ else c, markeredgewidth=1.0, zorder=3)
        ax.set_xlim(-0.6, len(ORDER) - 0.4); ax.set_xticks(x)
        ax.set_ylim(y0, y1); ax.set_yticks(np.arange(y0, y1 + 0.01, dy)); ax.set_ylabel(ylab)
        if k < 3:
            ax.set_xticklabels([])
            fig.text(left, tops[k] - ah - 0.012, sub, va="top", fontsize=9)
        else:
            ax.set_xticklabels(XLAB, fontsize=8)
            fig.text(xf(1.5), tops[k] - ah - 0.052, "Consistency Index", ha="center", va="top", fontsize=8.5)
            fig.text(xf(6.0), tops[k] - ah - 0.052, "Fixed fusion rules (best view, fraction of views)", ha="center",
                     va="top", fontsize=8.5)
            fig.text(left, tops[k] - ah - 0.074, sub, va="top", fontsize=9)
    hd = [Line2D([], [], linestyle="none", marker=m, markersize=5.5, color=c, label=seg) for seg, (c, m) in COL.items()]
    hd += [Line2D([], [], color="black", linewidth=1.8, label="Actual width (crack ruler)"),
           plt.Rectangle((0, 0), 1, 1, color="#e9e9e9", label="Actual width \u00b1 0.5 mm"),
           Line2D([], [], linestyle="none", marker="o", markersize=5.5, color="#555555", markerfacecolor="white",
                  label="Open: 3 to 7 of the 10 stations")]
    lg = fig.legend(handles=hd, loc="upper center", bbox_to_anchor=(0.545, 0.995), ncol=3, frameon=True, fontsize=8.5,
                    edgecolor="black", framealpha=1, handletextpad=0.4, columnspacing=1.2)
    lg.get_frame().set_linewidth(0.7)
    fig.savefig(f"{OUT}/rule_comparison.png", dpi=200); plt.close(fig)


def gif(data_dir=f"{ROOT}/data/drone", min_show=3, cell=(500, 326), pad=0.014):
    """One frame per rule, in the order of the comparison figure. Rows: the three 2D segmentations; columns: the three
    crack parts, each shown as a zoomed view of the point cloud around the part (front view of the wall, true scale).
    Dark red: crack points of that part kept by the rule; light red: crack points of the other parts. Under each view:
    width and error of the part; a part measured at fewer than `need` (8) stations is marked * and not used in the
    mean error, one with fewer than `min_show` stations is a dash."""
    import sys; sys.path.insert(0, os.path.dirname(__file__))
    from PIL import ImageDraw, ImageFont
    from ci_pipeline import load, load_mask, mask_labeler, consistency_index, fusion_rules, part_points
    geo = json.load(open(f"{data_dir}/geometry.json")); fr = geo["frame"]; c = geo["width_method"]
    o, A = np.array(fr["origin"]), np.array(fr["axes"])
    lines = {p: np.array(l) for p, l in geo["centrelines_uv"].items()}
    parts = list(lines); ref = [geo["reference_width_mm"][p] for p in parts]; need = c["need"]
    cw, ch = cell

    # crop box of each part: centreline extent plus a margin, widened to the aspect of the view
    boxes = {}
    for p, l in lines.items():
        lo, hi = l.min(0) - pad, l.max(0) + pad; ctr = (lo + hi) / 2; w, h = hi - lo
        if w / h < cw / ch: w = h * cw / ch
        else: h = w * ch / cw
        boxes[p] = (ctr[0] - w / 2, ctr[1] - h / 2, w / cw)          # u0, v0, metres per pixel

    def view(p, layers):
        u0, v0, mpp = boxes[p]; img = np.full((ch, cw, 3), 255, np.uint8)
        for uv, col, rad_mm in layers:
            x = ((uv[:, 0] - u0) / mpp).astype(int); y = ch - 1 - ((uv[:, 1] - v0) / mpp).astype(int)
            m = (x >= 0) & (x < cw) & (y >= 0) & (y < ch); x, y = x[m], y[m]
            col = col if np.ndim(col) == 1 else col[m]; r = max(1, int(round(rad_mm / 1000 / mpp)))
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        img[np.clip(y + dy, 0, ch - 1), np.clip(x + dx, 0, cw - 1)] = col
        return Image.fromarray(img)

    pts = np.load(f"{data_dir}/visible_points_crop.npz")                 # background: visible points of all photos
    xyz_all = np.vstack([pts[k] for k in pts.files if k.startswith("xyz_")])
    rgb_all = np.vstack([pts[k] for k in pts.files if k.startswith("rgb_")])
    _, first = np.unique(np.round(xyz_all, 4), axis=0, return_index=True)
    bg_uv = ((xyz_all[first] - o) @ A.T)[:, :2]
    bg_rgb = (0.55 * rgb_all[first] + 0.45 * 255).astype(np.uint8)

    views, P, vis, _ = load(data_dir); normal = A[2]
    keep = {}
    for seg in ("manual", "deepcrack", "masonry"):
        lab = mask_labeler(P, {v: load_mask(data_dir, seg, v) for v in views})
        cand, ci = consistency_index(views, P, vis, lab)
        keep[seg] = (cand, {f"CI {k}": ci >= k for k in (1, 2, 3, 4)} | fusion_rules(views, P, vis, lab, cand, normal))
    rows = json.load(open(f"{OUT}/widths.json"))
    title = {"manual": ["Manual crack", "marking"], "deepcrack": ["DeepCrack"], "masonry": ["Masonry crack", "segmentation"]}
    name = {"manual": "Manual crack marking", "deepcrack": "DeepCrack", "masonry": "Masonry crack segmentation"}
    frames_def = [("CI 1", "CI 1 (plain reprojection)", "Consistency Index 1: crack in at least 1 photo (plain reprojection)"),
                  ("CI 2", "CI 2", "Consistency Index 2: crack in at least 2 photos"),
                  ("CI 3", "CI 3", "Consistency Index 3: crack in at least 3 photos"),
                  ("CI 4", "CI 4", "Consistency Index 4: crack in at least 4 photos"),
                  ("best view (closest)", "best view (closest)", "Best view (closest): fixed rule, the label of the closest photo only"),
                  ("best view (most frontal)", "best view (most frontal)", "Best view (most frontal): fixed rule, the label of the most frontal photo only"),
                  ("fraction >= 1/3", "fraction >= 1/3", "Fraction \u2265 1/3: fixed rule, crack in at least 1/3 of the photos that see the point"),
                  ("fraction >= 1/2", "fraction >= 1/2", "Fraction \u2265 1/2: fixed rule, crack in at least 1/2 of the photos that see the point"),
                  ("fraction >= 2/3", "fraction >= 2/3", "Fraction \u2265 2/3: fixed rule, crack in at least 2/3 of the photos that see the point")]

    def font(size, bold=False):
        for f in (("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "arialbd.ttf") if bold else
                  ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "arial.ttf")):
            try: return ImageFont.truetype(f, size)
            except OSError: pass
        return ImageFont.load_default()
    f_cap, f_row, f_txt, f_note = font(24, True), font(20, True), font(19, True), font(16)

    def part_text(r, j):
        n, w = r["stations"][j], r["width_all_stations_mm"][j]
        if w is None or n < min_show:
            return f"Part {j + 1}   \u2013"
        t = f"Part {j + 1}   {w:.2f} mm ({w - ref[j]:+.2f})" + (f"   {n}/10 st.*" if n < need else "")
        return t.replace("-", "\u2212")

    left, gap, top, txt_h, row_gap = 196, 16, 10, 34, 12
    W = left + 3 * cw + 2 * gap + 8; row_h = ch + txt_h + row_gap; H = top + 3 * row_h + 96
    dark, light = np.array([200, 0, 0], np.uint8), np.array([245, 160, 160], np.uint8)
    frames = []
    for key, rule_name, caption in frames_def:
        im = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(im)
        for i, seg in enumerate(("manual", "deepcrack", "masonry")):
            cand, sets = keep[seg]; xyz = cand[sets[key]]
            r = next(r for r in rows if r["segmentation"] == name[seg] and r["rule"] == rule_name)
            uv = ((xyz - o) @ A.T)[:, :2]; y0 = top + i * row_h
            for k, t in enumerate(title[seg]):
                dr.text((8, y0 + ch / 2 - 46 + 26 * k), t, fill=(0, 0, 0), font=f_row)
            me = "n/a" if r["mean_error_mm"] is None else f"{r['mean_error_mm']:+.2f} mm".replace("-", "\u2212")
            dr.text((8, y0 + ch / 2 + 16), "mean error", fill=(0, 0, 0), font=f_txt)
            dr.text((8, y0 + ch / 2 + 40), me, fill=(0, 0, 0), font=f_txt)
            for j, p in enumerate(parts):
                own = np.zeros(len(xyz), bool); own[part_points(xyz, p, geo)[3]] = True
                img = view(p, [(bg_uv, bg_rgb, 0.55), (uv[~own], light, 0.8), (uv[own], dark, 0.8)])
                x0 = left + j * (cw + gap); im.paste(img, (x0, y0))
                dr.rectangle([x0, y0, x0 + cw - 1, y0 + ch - 1], outline=(120, 120, 120), width=1)
                dr.text((x0 + 2, y0 + ch + 6), part_text(r, j), fill=(0, 0, 0), font=f_txt)
        yb = top + 3 * row_h
        dr.text((8, yb + 2), caption, fill=(0, 0, 0), font=f_cap)
        dr.text((960, yb + 40), "Front view of the wall around each part, true scale.", fill=(70, 70, 70), font=f_note)
        dr.text((8, yb + 64), "Width: mean over the 10 stations of the part, error against the crack-ruler width in brackets "
                "(3.00, 2.48 and 1.25 mm).   * fewer than 8 of the 10 stations, not used in the mean error;   \u2013 fewer than 3 stations.",
                fill=(70, 70, 70), font=f_note)
        for k, (col, lab) in enumerate(((dark, "crack points of the part kept by the rule"), (light, "crack points of the other parts"))):
            xs = 8 + k * 470; dr.ellipse([xs, yb + 44, xs + 12, yb + 56], fill=tuple(int(v) for v in col))
            dr.text((xs + 18, yb + 40), lab, fill=(70, 70, 70), font=f_note)
        frames.append(im)
    pal = frames[0].quantize(colors=160, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)   # one palette for all frames
    lut = np.array(pal.getpalette()[:160 * 3]).reshape(-1, 3)
    for exact in ((255, 255, 255), (0, 0, 0)):                       # keep the page white and the text black
        lut[np.argmin(np.abs(lut - np.array(exact)).sum(1))] = exact
    pal.putpalette(lut.astype(int).flatten().tolist())
    gif_frames = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    gif_frames[0].save(f"{OUT}/consistency_index_drone.gif", save_all=True, append_images=gif_frames[1:], duration=1800, loop=0)
    frames[0].save(f"{OUT}/ci1_drone.png"); frames[1].save(f"{OUT}/ci2_drone.png")


if __name__ == "__main__":
    comparison(); gif(); print("figures written to", OUT)
