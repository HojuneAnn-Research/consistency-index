"""Frames and GIF of the crack kept at each Consistency Index, drawn on the reference photo."""
import numpy as np, json, argparse, os, cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fused', required=True); ap.add_argument('--ref-photo', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--width', type=int, default=640)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    count = np.load(os.path.join(a.fused, 'ci_count.npy')); kmax = int(count.max())
    img = np.asarray(ImageOps.exif_transpose(Image.open(a.ref_photo)).convert('RGB'))
    ys, xs = np.where(count >= 1); pad = 80
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, img.shape[0]); x0, x1 = max(xs.min() - 6 * pad, 0), min(xs.max() + 6 * pad, img.shape[1])
    base = (0.55 * img[y0:y1, x0:x1] + 0.45 * 255).astype(np.uint8)   # faded photo so the red crack points stand out
    s = a.width / base.shape[1]
    try: font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 28)
    except OSError: font = ImageFont.load_default()
    frames = []
    for k in range(1, kmax + 1):
        m = count[y0:y1, x0:x1] >= k
        m = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0   # keep 1-px lines visible after resizing
        f = base.copy(); f[m] = (200, 0, 0)
        im = Image.fromarray(f).resize((a.width, int(base.shape[0] * s)), Image.LANCZOS)
        d = ImageDraw.Draw(im); txt = f'Consistency Index {k}'
        d.rectangle((10, 10, 20 + d.textlength(txt, font=font), 52), fill=(255, 255, 255)); d.text((15, 14), txt, fill=(0, 0, 0), font=font)
        im.save(os.path.join(a.out, f'ci_{k}.png')); frames.append(im.convert('P', palette=Image.ADAPTIVE))
    frames[0].save(os.path.join(a.out, 'consistency_index.gif'), save_all=True, append_images=frames[1:], duration=1000, loop=0)
    print('frames', kmax, 'roi', (x0, y0, x1, y1), '->', os.path.join(a.out, 'consistency_index.gif'))

if __name__ == '__main__':
    main()
