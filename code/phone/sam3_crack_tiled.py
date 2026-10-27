# -*- coding: utf-8 -*-
"""[GPU PC] SAM3 crack segmentation on full-resolution photos, run tile by tile.

  python 01f_sam3_crack_tiled.py --images photos --out masks --ckpt sam3.pt

A hairline crack is only a few pixels wide, so each photo is cut into overlapping
tiles at full resolution (SAM3 works at about 1008 px) and the tile masks are merged.
Output: masks/<name>.png (255 = crack), masks/<name>_overlay.jpg, masks/manifest.json
"""
import os, json, argparse, glob
import numpy as np
from PIL import Image, ImageOps


def tiles(W, H, t, ov):
    step = t - ov
    xs = list(range(0, max(W - t, 0) + 1, step)) or [0]
    ys = list(range(0, max(H - t, 0) + 1, step)) or [0]
    if xs[-1] + t < W: xs.append(W - t)
    if ys[-1] + t < H: ys.append(H - t)
    for y in ys:
        for x in xs:
            yield x, y, min(x + t, W), min(y + t, H)


def _edt_stub():
    """SAM3's video tracker imports a triton kernel (sam3.model.edt) at import time.
    Image segmentation never calls it and triton is not installed here, so only that one
    module is replaced. A global triton placeholder would mislead torch into using triton."""
    import sys, types, importlib.util
    if importlib.util.find_spec("triton") is not None:
        return
    m = types.ModuleType("sam3.model.edt")
    def edt_triton(data):
        raise RuntimeError("edt_triton needs triton (video tracker only)")
    m.edt_triton = edt_triton
    sys.modules["sam3.model.edt"] = m
    print("[참고] triton 없음: 영상 추적용 커널(sam3.model.edt)만 대체")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompts", default="crack,hairline crack")
    ap.add_argument("--tile", type=int, default=1008)
    ap.add_argument("--overlap", type=int, default=160)
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    a = ap.parse_args()

    import torch
    _edt_stub()
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("[경고] GPU가 안 잡혔습니다. CPU로는 매우 느립니다.")
    if not os.path.exists(a.ckpt):
        raise SystemExit("체크포인트가 없습니다: " + a.ckpt)
    model = build_sam3_image_model(device=dev, checkpoint_path=a.ckpt, load_from_HF=False)
    proc = Sam3Processor(model)
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(a.amp)

    def autocast():
        if amp is None or dev != "cuda":
            return torch.autocast("cpu", enabled=False)
        return torch.autocast("cuda", dtype=amp)

    prompts = [p.strip() for p in a.prompts.split(",") if p.strip()]
    files = sorted(glob.glob(os.path.join(a.images, "*.jpg")) + glob.glob(os.path.join(a.images, "*.jpeg"))
                   + glob.glob(os.path.join(a.images, "*.png")))
    if not files:
        raise SystemExit("이미지 없음: " + a.images)
    os.makedirs(a.out, exist_ok=True)
    man = {"ckpt": a.ckpt, "prompts": prompts, "tile": a.tile, "overlap": a.overlap,
           "score_thr": a.score_thr, "images": {}}

    for i, f in enumerate(files):
        img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        W, H = img.size
        mask = np.zeros((H, W), bool)
        recs = []
        for (x0, y0, x1, y1) in tiles(W, H, a.tile, a.overlap):
            crop = img.crop((x0, y0, x1, y1))
            with torch.inference_mode(), autocast():
                st = proc.set_image(crop)
            for p in prompts:
                with torch.inference_mode(), autocast():
                    out = proc.set_text_prompt(state=st, prompt=p)
                ms = out["masks"]
                ms = ms.float().cpu().numpy() if hasattr(ms, "cpu") else np.asarray(ms)
                sc = out["scores"]
                sc = (sc.float().cpu().numpy() if hasattr(sc, "cpu") else np.asarray(sc)).reshape(-1)
                for k in np.where(sc >= a.score_thr)[0]:
                    m = ms[k]
                    m = m[0] if m.ndim == 3 else m
                    m = np.asarray(m) > 0.5
                    mask[y0:y1, x0:x1] |= m[: y1 - y0, : x1 - x0]
                    recs.append(dict(tile=[x0, y0, x1, y1], prompt=p, score=float(sc[k]), area=int(m.sum())))
        base = os.path.splitext(os.path.basename(f))[0]
        Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(a.out, base + ".png"))
        ov = np.asarray(img).copy()
        ov[mask] = (0.3 * ov[mask] + 0.7 * np.array([255, 0, 0])).astype(np.uint8)
        Image.fromarray(ov).save(os.path.join(a.out, base + "_overlay.jpg"), quality=90)
        man["images"][base] = dict(size=[W, H], crack_pixels=int(mask.sum()), detections=recs)
        print(f"  [{i+1}/{len(files)}] {base}: crack px {int(mask.sum())}, detections {len(recs)}", flush=True)

    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[완료]", a.out)


if __name__ == "__main__":
    main()
