"""Figures for the drone example: rule comparison (width of each part and mean width error), a GIF of the crack
points kept by each rule, and the crack points after each pre-filtering stage."""
import json, os
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/results/drone"


COL = {"Manual crack marking": ("#0072B2", "o"), "DeepCrack": ("#D55E00", "s"), "Masonry crack segmentation": ("#009E73", "^")}
ORDER = ["CI 1 (plain reprojection)", "CI 2", "best view (closest)", "best view (most frontal)",
         "fraction >= 1/3", "fraction >= 1/2", "fraction >= 2/3"]
XLAB = ["CI 1\n(plain)", "CI 2", "Best view\n(closest)", "Best view\n(frontal)", "Fraction\n\u2265 1/3", "Fraction\n\u2265 1/2",
        "Fraction\n\u2265 2/3"]


def comparison():
    """Width of each part (a-c) and mean width error (d) for CI 1, CI 2 and the fixed rules, for every segmentation; the
    black line is the crack-ruler width (error 0 in d). Open markers: parts measured at only 3 to 7 of the 10 stations."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    rows = json.load(open(f"{OUT}/widths.json")); geo = json.load(open(f"{ROOT}/data/drone/geometry.json"))
    ref = [geo["reference_width_mm"][p] for p in ("Part1", "Part2", "Part3")]; need = geo["width_method"]["need"]
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "font.size": 9,
                         "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in"})
    fig = plt.figure(figsize=(7.4, 10.4))
    left, right, ah, gap = 0.105, 0.985, 0.14, 0.085
    tops = [0.93 - k * (ah + gap) for k in range(4)]
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
        ax.axvline(1.5, color="#888888", linewidth=0.8, linestyle="--")
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
        ax.set_xticklabels(XLAB, fontsize=8)
        if k < 3:
            fig.text(left, tops[k] - ah - 0.042, sub, va="top", fontsize=9)
        else:
            fig.text(xf(0.5), tops[k] - ah - 0.045, "Consistency Index", ha="center", va="top", fontsize=8.5)
            fig.text(xf(4.0), tops[k] - ah - 0.045, "Fixed fusion rules (best view, fraction of views)", ha="center",
                     va="top", fontsize=8.5)
            fig.text(left, tops[k] - ah - 0.066, sub, va="top", fontsize=9)
    hd = [Line2D([], [], linestyle="none", marker=m, markersize=5.5, color=c, label=seg) for seg, (c, m) in COL.items()]
    hd += [Line2D([], [], color="black", linewidth=1.8, label="Actual width (crack ruler)"),
           plt.Rectangle((0, 0), 1, 1, color="#e9e9e9", label="Actual width \u00b1 0.5 mm"),
           Line2D([], [], linestyle="none", marker="o", markersize=5.5, color="#555555", markerfacecolor="white",
                  label="Open: 3 to 7 of the 10 stations")]
    lg = fig.legend(handles=hd, loc="upper center", bbox_to_anchor=(0.545, 0.997), ncol=3, frameon=True, fontsize=8.5,
                    edgecolor="black", framealpha=1, handletextpad=0.4, columnspacing=1.2)
    lg.get_frame().set_linewidth(0.7)
    fig.savefig(f"{OUT}/rule_comparison.png", dpi=200); plt.close(fig)


def gif(data_dir=f"{ROOT}/data/drone", panel_w=490, pad=0.02):
    """One frame per rule, in the order of the README table without CI 3 and CI 4: the whole crack region for the three
    2D segmentations side by side (front view of the wall, true scale), with the crack points kept by the rule in red and
    the mean width error of each segmentation under its view."""
    import sys; sys.path.insert(0, os.path.dirname(__file__))
    from PIL import ImageDraw, ImageFont
    from ci_pipeline import load, load_mask, mask_labeler, consistency_index, fusion_rules
    geo = json.load(open(f"{data_dir}/geometry.json")); fr = geo["frame"]
    o, A = np.array(fr["origin"]), np.array(fr["axes"])
    lines = {p: np.array(l) for p, l in geo["centrelines_uv"].items()}

    lo = np.min([l.min(0) for l in lines.values()], 0) - pad; hi = np.max([l.max(0) for l in lines.values()], 0) + pad
    mpp = (hi[0] - lo[0]) / panel_w; cw, ch = panel_w, int(round((hi[1] - lo[1]) / mpp))

    def view(layers):
        img = np.full((ch, cw, 3), 255, np.uint8)
        for uv, col, rad_mm in layers:
            x = ((uv[:, 0] - lo[0]) / mpp).astype(int); y = ch - 1 - ((uv[:, 1] - lo[1]) / mpp).astype(int)
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
    f_cap, f_row, f_txt, f_note = font(30, True), font(26, True), font(24, True), font(20)

    left, gap, top, lab_h = 8, 16, 10, 80
    W = left + 3 * cw + 2 * gap + 8; H = top + ch + lab_h + 96
    red = np.array([200, 0, 0], np.uint8)
    frames = []
    for key, rule_name, caption in frames_def:
        im = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(im)
        for i, seg in enumerate(("manual", "deepcrack", "masonry")):
            cand, sets = keep[seg]; xyz = cand[sets[key]]
            r = next(r for r in rows if r["segmentation"] == name[seg] and r["rule"] == rule_name)
            uv = ((xyz - o) @ A.T)[:, :2]
            x0 = left + i * (cw + gap); im.paste(view([(bg_uv, bg_rgb, 0.55), (uv, red, 0.8)]), (x0, top))
            dr.rectangle([x0, top, x0 + cw - 1, top + ch - 1], outline=(120, 120, 120), width=1)
            me = "n/a" if r["mean_error_mm"] is None else f"{r['mean_error_mm']:+.2f} mm".replace("-", "\u2212")
            dr.text((x0 + 2, top + ch + 6), name[seg], fill=(0, 0, 0), font=f_row)
            dr.text((x0 + 2, top + ch + 40), f"mean width error {me}", fill=(0, 0, 0), font=f_txt)
        yb = top + ch + lab_h + 6
        dr.text((8, yb + 2), caption, fill=(0, 0, 0), font=f_cap)
        dr.text((8, yb + 48), "Red: crack points kept by the rule. Front view of the wall, true scale. Mean width error against "
                "the crack-ruler width, as in the README table.", fill=(0, 0, 0), font=f_note)
        frames.append(im)
    pal = frames[0].quantize(colors=160, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)   # one palette for all frames
    lut = np.array(pal.getpalette()[:160 * 3]).reshape(-1, 3)
    for exact in ((255, 255, 255), (0, 0, 0)):                       # keep the page white and the text black
        lut[np.argmin(np.abs(lut - np.array(exact)).sum(1))] = exact
    pal.putpalette(lut.astype(int).flatten().tolist())
    gif_frames = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    gif_frames[0].save(f"{OUT}/consistency_index_drone.gif", save_all=True, append_images=gif_frames[1:], duration=1800, loop=0)
    frames[0].save(f"{OUT}/ci1_drone.png"); frames[1].save(f"{OUT}/ci2_drone.png")


def prefilter_figure(data_dir=f"{ROOT}/data/drone"):
    """Front view of the crack points of four photos (manual crack marking) after depth filtering (a) and after depth
    filtering followed by hidden point removal (b), coloured by their distance from the tile surface (mm). The depth
    thresholds themselves are 3.6 to 10.0 m from the camera (prefilter.json)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    geo = json.load(open(f"{data_dir}/geometry.json")); fr = geo["frame"]
    o, A = np.array(fr["origin"]), np.array(fr["axes"])
    st = np.load(f"{data_dir}/prefilter_stages.npz"); views = sorted({k.split("_")[1] for k in st.files})
    pts = np.load(f"{data_dir}/visible_points_crop.npz")
    bg = np.vstack([pts[k] for k in pts.files if k.startswith("xyz_")])
    bg = (np.unique(np.round(bg, 4), axis=0) - o) @ A.T * 1000
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "font.size": 11,
                         "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    cats = [("< 10 mm (on the tile surface)", lambda d: d < 10, "#C80000"),
            ("10 to 50 mm", lambda d: (d >= 10) & (d < 50), "#E69F00"),
            ("> 50 mm (occluded geometry)", lambda d: d >= 50, "#0072B2")]
    for ax, tag, sub in zip(axes, ("depth", "hpr"), ("(a) After depth filtering",
                                                    "(b) After depth filtering and hidden point removal")):
        uvw = np.vstack([(st[f"{tag}_{v}"] - o) @ A.T for v in views]) * 1000; d = np.abs(uvw[:, 2])
        ax.scatter(bg[:, 0], bg[:, 1], s=0.3, c="#d9d9d9", linewidths=0, rasterized=True)
        for lab, sel, col in cats:
            m = sel(d); ax.scatter(uvw[m, 0], uvw[m, 1], s=7 if col != "#C80000" else 3, c=col, linewidths=0, label=lab)
        ax.set_xlim(-200, 200); ax.set_ylim(-150, 250); ax.set_aspect("equal")
        ax.set_xticks(range(-200, 201, 100)); ax.set_yticks(range(-150, 251, 50))
        ax.set_xlabel("Along the wall (mm)"); ax.set_ylabel("Up the wall (mm)")
        ax.grid(True, color="#bbbbbb", linestyle=":", linewidth=0.6)
        ax.tick_params(axis="x", pad=7)
        for sp in ax.spines.values(): sp.set_visible(True)
        ax.text(0.5, -0.15, sub, transform=ax.transAxes, ha="center", va="top", fontsize=12)
    leg = axes[0].legend(loc="lower right", frameon=True, fontsize=9.5, markerscale=2.5, framealpha=1, edgecolor="black",
                         title="Distance of a crack point from the tile surface", title_fontsize=9.5)
    leg.get_frame().set_linewidth(0.7)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.98, bottom=0.17, wspace=0.22)
    fig.savefig(f"{OUT}/prefilter_stages.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    comparison(); gif(); prefilter_figure(); print("figures written to", OUT)
