"""Reprojection-based 3D crack identification with the Consistency Index (CI), plus the
fixed fusion rules it is compared with. Works on the cropped drone dataset in data/drone."""
import json
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

Image.MAX_IMAGE_PIXELS = None


# ---------------------------------------------------------------- data
def load(data_dir):
    cams = json.load(open(f"{data_dir}/cameras.json"))
    pts = np.load(f"{data_dir}/visible_points_crop.npz")
    geo = json.load(open(f"{data_dir}/geometry.json"))
    views = cams["views"]
    P = {v: np.array(cams["P"][v]) for v in views}
    vis = {v: (pts[f"xyz_{v}"], pts[f"rgb_{v}"]) for v in views}
    return views, P, vis, geo


def load_mask(data_dir, seg, v):
    return np.asarray(Image.open(f"{data_dir}/masks/{seg}/DJI_{v}.png")) > 0


# ---------------------------------------------------------------- reprojection
def project(xyz, P):
    h = np.c_[xyz, np.ones(len(xyz))] @ P.T
    return h[:, 0] / h[:, 2], h[:, 1] / h[:, 2], h[:, 2]


def label_view(xyz, P, mask):
    """Crack label of every visible point in one photo: pixel by truncation, exact mask hit, no dilation."""
    u, v, _ = project(xyz, P)
    u, v = u.astype(int), v.astype(int)
    ok = (u >= 0) & (u < mask.shape[1]) & (v >= 0) & (v < mask.shape[0])
    hit = np.zeros(len(xyz), bool)
    hit[ok] = mask[v[ok], u[ok]]
    return hit


def mask_labeler(P, masks):
    """Per-photo labels from the 2D masks."""
    return lambda v, xyz: label_view(xyz, P[v], masks[v])


def consistency_index(views, P, vis, labeler):
    """Number of photos in which each point reprojects onto a crack pixel.
    As in the original implementation, every row of the visible-point export is counted,
    so a point exported twice for one photo counts twice."""
    keys = {}
    for v in views:
        xyz = vis[v][0]
        for k in map(tuple, np.round(xyz[labeler(v, xyz)], 5)):
            keys[k] = keys.get(k, 0) + 1
    return np.array(list(keys)), np.array(list(keys.values()))


# ---------------------------------------------------------------- fixed fusion rules on the same data
def camera_centre(P):
    return -np.linalg.solve(P[:, :3], P[:, 3])


def fusion_rules(views, P, vis, labeler, cand, normal):
    """Labels of the candidate points (those labelled crack in at least one photo) under fixed rules."""
    n = len(cand)
    V = np.zeros((n, len(views)), bool); C = np.zeros_like(V)
    dep = np.full((n, len(views)), np.inf); ang = np.full((n, len(views)), np.inf)
    for j, v in enumerate(views):
        xyz = vis[v][0]
        d, _ = cKDTree(xyz).query(cand); V[:, j] = d < 1e-4
        lab = xyz[labeler(v, xyz)]
        if len(lab):
            d2, _ = cKDTree(lab).query(cand); C[:, j] = (d2 < 1e-4) & V[:, j]
        _, _, w = project(cand, P[v]); dep[:, j] = np.where(V[:, j], w, np.inf)
        ray = camera_centre(P[v]) - cand; ray /= np.linalg.norm(ray, axis=1)[:, None]
        a = np.degrees(np.arccos(np.clip(np.abs(ray @ normal), 0, 1))); ang[:, j] = np.where(V[:, j], a, np.inf)
    nv, nc, i = V.sum(1), C.sum(1), np.arange(n)
    rules = {"best view (closest)": C[i, np.argmin(dep, 1)] & np.isfinite(dep.min(1)),
             "best view (most frontal)": C[i, np.argmin(ang, 1)] & np.isfinite(ang.min(1))}
    for f, name in ((1 / 3, "fraction >= 1/3"), (1 / 2, "fraction >= 1/2"), (2 / 3, "fraction >= 2/3")):
        rules[name] = (nv > 0) & (nc >= f * nv - 1e-9)
    return rules


# ---------------------------------------------------------------- automatic width at 10 stations per part
def _project_polyline(q, line):
    seg = line[1:] - line[:-1]; L = np.linalg.norm(seg, axis=1); cum = np.r_[0, np.cumsum(L)]
    best = np.full(len(q), np.inf); s = np.zeros(len(q)); d = np.zeros(len(q))
    for i in range(len(seg)):
        t = seg[i] / L[i]; nrm = np.array([-t[1], t[0]]); r = q - line[i]
        tt = np.clip(r @ t, 0, L[i]); dist = np.linalg.norm(q - (line[i] + np.outer(tt, t)), axis=1)
        up = dist < best; best[up] = dist[up]; s[up] = cum[i] + tt[up]; d[up] = (r @ nrm)[up]
    return s, d, best, cum[-1]


def width(xyz, part, geo):
    """Across-crack extent (mm) of the contiguous band at 10 equally spaced stations.
    Returns (width, n, width_all): n is the number of stations that could be measured, width_all the mean
    over those n stations, and width the same mean but None when n is below `need` (8), as in the manuscript."""
    c = geo["width_method"]; fr = geo["frame"]
    uvw = (xyz - np.array(fr["origin"])) @ np.array(fr["axes"]).T
    q2 = uvw[np.abs(uvw[:, 2]) < c["wtol"], :2]
    lines = {p: np.array(l) for p, l in geo["centrelines_uv"].items()}
    names = list(lines)
    dd = np.stack([_project_polyline(q2, lines[p])[2] for p in names], 1)
    q = q2[(dd.argmin(1) == names.index(part)) & (dd.min(1) < c["R"])]
    if len(q) < 2:
        return None, 0, None
    s, d, _, L = _project_polyline(q, lines[part])
    edges = np.linspace(c["margin"] * L, (1 - c["margin"]) * L, c["nseg"] + 1); ws = []
    for a, b in zip(edges[:-1], edges[1:]):
        cand = np.arange(a, b, 0.0005)
        for s0 in cand[np.argsort(np.abs(cand - (a + b) / 2))]:
            sel = np.abs(s - s0) <= c["h"]
            if sel.sum() >= c["minpts"]:
                g = np.sort(d[sel]); groups = np.split(g, np.where(np.diff(g) > c["gap"])[0] + 1)
                g = min(groups, key=lambda x: 0 if x.min() <= 0 <= x.max() else min(abs(x.min()), abs(x.max())))
                if len(g) >= c["minpts"] and g.max() > g.min():
                    ws.append(g.max() - g.min()); break
    w_all = float(np.mean(ws)) * 1000 if ws else None
    return (w_all if len(ws) >= c["need"] else None), len(ws), w_all
