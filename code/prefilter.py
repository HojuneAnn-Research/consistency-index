"""Pre-filtering of the point cloud for each photo (manuscript, Section 2.2) and the comparison of its two stages
(Section 3.4). Depth filtering keeps the points that project inside the image and are no deeper than a threshold;
hidden point removal (Open3D) then keeps the points visible from the camera centre. The full point cloud is
available on request; the crack points stored after each stage for four photos reproduce the comparison.
Usage: python code/prefilter.py   (prints the comparison of the stages)"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from ci_pipeline import load, project, camera_centre, _project_polyline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = f"{ROOT}/data/drone"


def depth_filter(xyz, P, max_depth, size=(5472, 3648)):
    """Points that project inside the image with a projective depth of at most max_depth (m), as in the original
    implementation (only an upper bound on the depth)."""
    u, v, w = project(xyz, P)
    return (u >= 0) & (u < size[0]) & (v >= 0) & (v < size[1]) & (w <= max_depth)


def hidden_point_removal(xyz, camera, radius):
    """Indices of the points visible from `camera` (Open3D hidden point removal)."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz, float))
    _, idx = pcd.hidden_point_removal(np.asarray(camera, float), float(radius))
    return np.asarray(idx, int)


def prefilter(xyz, P, max_depth, radius):
    """Indices of the points kept for one photo: depth filtering, then hidden point removal from the camera centre."""
    keep = np.flatnonzero(depth_filter(xyz, P, max_depth))
    return keep[hidden_point_removal(xyz[keep], camera_centre(P), radius)]


def classify(xyz, geo, on_tol=0.010, off_tol=0.050, line_tol=0.008):
    """Crack points on the tile surface (|offset| < 10 mm), 10 to 50 mm off the surface along the crack line
    (within 8 mm of a part centreline), and more than 50 mm off the surface."""
    fr = geo["frame"]; uvw = (xyz - np.array(fr["origin"])) @ np.array(fr["axes"]).T
    off = np.abs(uvw[:, 2])
    dmin = np.min(np.stack([_project_polyline(uvw[:, :2], np.array(l))[2] for l in geo["centrelines_uv"].values()], 1), 1)
    return dict(on_surface=int((off < on_tol).sum()),
                off_10_50_mm=int(((off >= on_tol) & (off < off_tol) & (dmin < line_tol)).sum()),
                off_over_50_mm=int((off >= off_tol).sum()))


def main():
    geo = json.load(open(f"{DATA}/geometry.json")); st = np.load(f"{DATA}/prefilter_stages.npz")
    views = sorted({k.split("_")[1] for k in st.files}); total = {}
    print("photo  stage                   on surface  10-50 mm off  >50 mm off")
    for v in views:
        for tag, name in (("depth", "depth filtering"), ("hpr", "+ hidden point removal")):
            c = classify(st[f"{tag}_{v}"], geo)
            print(f"{v}   {name:24s}{c['on_surface']:10d}{c['off_10_50_mm']:14d}{c['off_over_50_mm']:12d}")
            for k, n in c.items(): total[(tag, k)] = total.get((tag, k), 0) + n
    for tag, name in (("depth", "depth filtering"), ("hpr", "+ hidden point removal")):
        print(f"all    {name:24s}{total[(tag, 'on_surface')]:10d}{total[(tag, 'off_10_50_mm')]:14d}{total[(tag, 'off_over_50_mm')]:12d}")


if __name__ == "__main__":
    main()
