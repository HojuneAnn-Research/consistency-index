"""Figure for the drone example: a GIF of the crack points kept by each rule."""
import json, os
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/results/drone"


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


if __name__ == "__main__":
    gif(); print("GIF written to", OUT)
