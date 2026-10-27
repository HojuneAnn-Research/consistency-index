"""Drone example: CI point clouds for three 2D segmentations, automatic widths at CI 1-4,
and the same widths for fixed fusion rules on the same point cloud and photos.
Usage: python code/run_drone_example.py   (writes results/drone)"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from ci_pipeline import load, load_mask, mask_labeler, run_labeler, consistency_index, fusion_rules, width

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, OUT = f"{ROOT}/data/drone", f"{ROOT}/results/drone"
SEGS = {"manual": "Manual crack marking", "deepcrack": "DeepCrack", "masonry": "Masonry crack segmentation"}
PARTS = ["Part1", "Part2", "Part3"]


def main():
    os.makedirs(OUT, exist_ok=True)
    views, P, vis, geo = load(DATA)
    ref = geo["reference_width_mm"]; normal = np.array(geo["frame"]["axes"])[2]
    rows = []
    for seg, title in SEGS.items():
        if seg == "masonry":   # masks of the original masonry run were not kept: its CI clouds are used as produced
            run = np.load(f"{DATA}/masonry_run_ci.npz"); pts, ci, lab = run["xyz"], run["ci"], None
        else:
            lab = mask_labeler(P, {v: load_mask(DATA, seg, v) for v in views})
            pts, ci = consistency_index(views, P, vis, lab)
        np.savetxt(f"{OUT}/{seg}_ci_points.csv", np.c_[pts, ci], delimiter=",", fmt=["%.5f"] * 3 + ["%d"],
                   header="x,y,z,ci", comments="")
        sets = {f"CI {k}" + (" (plain reprojection)" if k == 1 else ""): ci >= k for k in (1, 2, 3, 4)}
        if lab is not None:
            sets.update(fusion_rules(views, P, vis, lab, pts, normal))
        for name, sel in sets.items():
            w = [width(pts[sel], p, geo)[0] for p in PARTS]
            e = [x - ref[p] for x, p in zip(w, PARTS) if x is not None]
            rows.append(dict(segmentation=title, rule=name, points=int(sel.sum()),
                             width_mm=[None if x is None else round(x, 2) for x in w],
                             mean_error_mm=None if not e else round(float(np.mean(e)), 2), parts_measured=len(e)))
            print(rows[-1], flush=True)
    json.dump(rows, open(f"{OUT}/widths.json", "w"), indent=1)


if __name__ == "__main__":
    main()
