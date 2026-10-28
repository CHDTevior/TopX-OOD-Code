"""Skeleton GIF rendering used by scripts/generate.py."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

PANEL_W, PANEL_H, GAP, FOOT = 300, 360, 12, 66


COLS = {"demo": (13, 110, 100), "gen_ric": (176, 61, 8),
        "gen_fk": (91, 44, 184), "gt": (185, 28, 28)}


def draw_panel(img, xy, parents, col, x0):
    d = ImageDraw.Draw(img)
    for j, p in enumerate(parents):
        if p < 0:
            continue
        d.line([x0 + xy[j, 0], xy[j, 1], x0 + xy[p, 0], xy[p, 1]], fill=col, width=3)
    for j in range(len(parents)):
        r = 4 if j == 0 else 2
        d.ellipse([x0 + xy[j, 0] - r, xy[j, 1] - r, x0 + xy[j, 0] + r, xy[j, 1] + r], fill=col)


def project(seqs, yaw_deg=28.0):
    """Shared orthographic projection: x' = x cosA + z sinA, y' = y (up). One bounding box over all sequences so
    the panels share a scale; returns (list of [T,J,2] pixel coordinates, ground_row_px). The box always includes world
    y = 0, so the caller can draw the ground line.
    """
    a = np.radians(yaw_deg)
    flat = [np.stack([s[..., 0] * np.cos(a) + s[..., 2] * np.sin(a), s[..., 1]], axis=-1)
            for s in seqs]
    allp = np.concatenate([f.reshape(-1, 2) for f in flat], axis=0)
    lo, hi = allp.min(0), allp.max(0)
    lo[1] = min(lo[1], 0.0)            # ground always in frame
    hi[1] = max(hi[1], 0.0)
    span = float(max(hi[0] - lo[0], hi[1] - lo[1], 1e-6))
    sc = (min(PANEL_W, PANEL_H) - 46) / span
    out = []
    for f in flat:
        px = (f[..., 0] - (lo[0] + hi[0]) / 2) * sc + PANEL_W / 2
        py = PANEL_H - ((f[..., 1] - (lo[1] + hi[1]) / 2) * sc + PANEL_H / 2)
        out.append(np.stack([px, py], axis=-1))
    ground_px = PANEL_H - ((0.0 - (lo[1] + hi[1]) / 2) * sc + PANEL_H / 2)
    return out, float(ground_px)


def render_gif(out_path, panels, parents, caption, rig, fps=20):
    """panels: list of (color_key, tag, seq [T,J,3]); shared scale across all of them."""
    names = [c for c, _, _ in panels]
    tags = [t for _, t, _ in panels]
    seqs, ground_px = project([w for _, _, w in panels])
    T = max(s.shape[0] for s in seqs)
    W = PANEL_W * len(panels) + GAP * (len(panels) - 1)
    frames = []
    for t in range(T):
        img = Image.new("RGB", (W, PANEL_H + FOOT), (246, 248, 247))
        d = ImageDraw.Draw(img)
        for k, (name, seq) in enumerate(zip(names, seqs)):
            x0 = k * (PANEL_W + GAP)
            # ground: shaded underground band + line at world y=0, drawn UNDER the skeleton
            gy = min(max(ground_px, 0), PANEL_H - 1)
            d.rectangle([x0, gy, x0 + PANEL_W - 1, PANEL_H - 1], fill=(236, 240, 238))
            d.line([x0, gy, x0 + PANEL_W - 1, gy], fill=(170, 186, 178), width=2)
            for tx in range(x0 + 10, x0 + PANEL_W - 4, 28):   # sparse ticks: reads as a floor
                d.line([tx, gy, tx - 6, gy + 5], fill=(198, 210, 202), width=1)
            d.rectangle([x0, 0, x0 + PANEL_W - 1, PANEL_H - 1], outline=(216, 223, 225))
            tt = min(t, seq.shape[0] - 1)                    # shorter panels freeze on last frame
            draw_panel(img, seq[tt], parents, COLS[name], x0)
            d.text((x0 + 8, 6), tags[k], fill=COLS[name])
            d.text((x0 + 8, PANEL_H - 18), f"f{tt+1}/{seq.shape[0]}", fill=(116, 133, 150))
        d.text((8, PANEL_H + 8), rig, fill=(15, 23, 32))
        for li, line in enumerate([caption[i:i + 96] for i in range(0, min(len(caption), 192), 96)]):
            d.text((8, PANEL_H + 26 + 16 * li), line, fill=(64, 80, 94))
        frames.append(img)
    cum = [round(i * 100.0 / fps) * 10 for i in range(len(frames) + 1)]
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=[b - a for a, b in zip(cum, cum[1:])], loop=0)
