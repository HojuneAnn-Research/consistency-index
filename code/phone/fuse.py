"""Planar multi-view crack fusion with the Consistency Index (CI).
The wall is planar, so every wall point (a pixel of the reference photo) is reprojected into each photo
through the plane-induced homography. CI(point) = number of photos whose crack mask covers the point."""
import cv2, numpy as np, json, argparse, os, glob, itertools
from PIL import Image, ImageOps

def load(path, gray=False):
    im = ImageOps.exif_transpose(Image.open(path)).convert('L' if gray else 'RGB')
    return np.asarray(im)

REG_LONG = 3000   # registration runs on images resized to this long side; homographies are rescaled to full resolution

def features(photos):
    sift = cv2.SIFT_create(nfeatures=20000, contrastThreshold=0.01)
    cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    out = []
    for p in photos:
        g = load(p, gray=True); s = min(1.0, REG_LONG / max(g.shape))
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g
        kp, des = sift.detectAndCompute(cl.apply(g), None)
        out.append((kp, des, s))
    return out

def pair_h(f1, f2):
    (k1, d1, s1), (k2, d2, s2) = f1, f2
    m = [a for a, b in cv2.BFMatcher().knnMatch(d1, d2, k=2) if a.distance < 0.8 * b.distance]
    if len(m) < 8: return None, 0, np.inf
    p1 = np.float32([k1[a.queryIdx].pt for a in m]); p2 = np.float32([k2[a.trainIdx].pt for a in m])
    H, inl = cv2.findHomography(p1, p2, cv2.USAC_MAGSAC, 4.0, maxIters=20000, confidence=0.999)
    if H is None: return None, 0, np.inf
    inl = inl.ravel().astype(bool)
    q = cv2.perspectiveTransform(p1[inl][None], H)[0]; err = float(np.median(np.linalg.norm(q - p2[inl], axis=1)) / s2)
    H = np.diag([1 / s2, 1 / s2, 1]) @ H @ np.diag([s1, s1, 1])   # full-resolution homography, error in full-resolution px
    return H, int(inl.sum()), err

def to_reference(F, ref, min_inl):
    """Homography from each photo to the reference: direct, or chained through the strongest link."""
    n = len(F); Hs = {}; INL = np.zeros((n, n), int); ERR = np.full((n, n), np.inf)
    for i, j in itertools.permutations(range(n), 2):
        H, c, e = pair_h(F[i], F[j]); Hs[(i, j)] = H; INL[i, j] = c; ERR[i, j] = e
    out = {ref: (np.eye(3), 'reference', 0, 0.0)}
    for i in range(n):
        if i == ref: continue
        if INL[i, ref] >= min_inl: out[i] = (Hs[(i, ref)], 'direct', int(INL[i, ref]), float(ERR[i, ref])); continue
        best = max((min(INL[i, k], INL[k, ref]), k) for k in range(n) if k not in (i, ref))
        if best[0] >= min_inl:
            k = best[1]; out[i] = (Hs[(k, ref)] @ Hs[(i, k)], f'via {k+1}', int(best[0]), float(max(ERR[i, k], ERR[k, ref])))
    return out, INL

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--photos', required=True); ap.add_argument('--masks', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--ref', type=int, default=4, help='1-based reference photo')
    ap.add_argument('--min-inliers', type=int, default=300)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    photos = sorted(glob.glob(os.path.join(a.photos, '*.jpg'))); names = [os.path.splitext(os.path.basename(p))[0] for p in photos]
    F = features(photos); ref = a.ref - 1
    T, INL = to_reference(F, ref, a.min_inliers)
    R = load(photos[ref]); Hh, Ww = R.shape[:2]
    count = np.zeros((Hh, Ww), np.uint8); vis = np.zeros((Hh, Ww), np.uint8); used = []
    for i, p in enumerate(photos):
        if i not in T: continue
        m = (load(os.path.join(a.masks, names[i] + '.png'), gray=True) > 127).astype(np.uint8)
        H = T[i][0]
        count += cv2.warpPerspective(m, H, (Ww, Hh), flags=cv2.INTER_NEAREST)
        vis += cv2.warpPerspective(np.ones_like(m), H, (Ww, Hh), flags=cv2.INTER_NEAREST)
        used.append(dict(photo=names[i], link=T[i][1], inliers=T[i][2], median_err_px=round(T[i][3], 2)))
    np.save(os.path.join(a.out, 'ci_count.npy'), count); np.save(os.path.join(a.out, 'ci_visible.npy'), vis)
    kmax = int(count.max())
    summ = dict(reference=names[ref], photos_used=used, photos_dropped=[names[i] for i in range(len(photos)) if i not in T],
                crack_pixels_per_CI={k: int((count >= k).sum()) for k in range(1, kmax + 1)})
    json.dump(summ, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
    print(json.dumps(summ, indent=1))

if __name__ == '__main__':
    main()
