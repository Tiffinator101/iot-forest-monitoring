# Simplest version of the tree measurement (steps 5 and 6 in one script, extras removed).
#
#   1. Canopy: yoloe-26s-seg with the prompt "tree". The largest mask is kept, and its
#      top row, bottom row, left and right columns give the tree top, the canopy base
#      and the canopy width.
#   2. Trunk: in a strip below the canopy, Gaussian blur, Lab a* channel, Otsu threshold,
#      opening and closing, then the tallest thin object. It is traced down row by row
#      against the grass in the same row to find the trunk base.
#   3. Pixels to metres: the stick's marks (100 mm apart, 2.5 m from the camera) give one
#      focal length for the camera, and size = pixels x distance / focal length.
#   4. Trunk width: 0.5 m above the trunk base, from the a* profile
#      across the trunk averaged over a block of rows, at half maximum.
#
# Left out compared with steps 5 and 6: convex hull, tape band search, Canny cross-check,
# low-leaf texture tracing, stage images, per-photo focal length and the trunk-scale method.

import csv
import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLOE

# Settings
TEST_DIR = Path(r"C:\Users\tiffa\OneDrive\Documents\University\Engineering\Year 4"
                r"\Semester 2\Project 448\Project\Testing\Tree Photo Test\Analysis Images")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "simple"
STICK_DISTANCE_M = 2.5          # camera to stick
MARK_SPACING_M = 0.100          # stick marks, top to top
TAPE_HEIGHT_M, TAPE_CANOPY_M, CALLIPER_TRUNK_MM = 4.0, 3.5, 121.0   # reference values
WIDTH_HEIGHT_M = 0.50           # trunk width is measured this far above the trunk base (just below the test tape band)

# The model files, including YOLOE's text encoder (mobileclip*.ts), are in ../models, and
# Ultralytics looks for them in the working folder
os.chdir(Path(__file__).resolve().parent.parent / "models")
model = YOLOE("yoloe-26s-seg.pt")
model.set_classes(["tree"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_canopy(img):
    """Largest YOLOE tree mask and the rows and columns it covers."""
    r = model.predict(source=img, conf=0.25, retina_masks=True, verbose=False)[0]
    masks = r.masks.data.cpu().numpy() > 0.5
    mask = masks[masks.sum(axis=(1, 2)).argmax()]
    ys, xs = np.nonzero(mask)
    return {"top": ys.min(), "base": ys.max(), "left": xs.min(), "right": xs.max(),
            "width": xs.max() - xs.min() + 1, "height": ys.max() - ys.min() + 1}


def find_trunk(img, canopy):
    """Trunk mask in a strip below the canopy: a*, Otsu, opening and closing, tallest thin object."""
    cx = (canopy["left"] + canopy["right"]) / 2
    x1, x2 = int(cx - 0.3 * canopy["width"]), int(cx + 0.3 * canopy["width"])
    y1 = int(canopy["base"] - 0.1 * canopy["height"])
    y2 = int(min(img.shape[0], canopy["base"] + 0.6 * canopy["height"]))
    a_star = cv2.cvtColor(cv2.GaussianBlur(img[y1:y2, x1:x2], (5, 5), 0), cv2.COLOR_BGR2LAB)[:, :, 1]
    _, mask = cv2.threshold(a_star, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    best = None
    for k in range(1, n):
        obj = labels == k
        rows = obj.any(axis=1)
        if rows.sum() < 0.3 * mask.shape[0]:                     # too short to be the trunk
            continue
        if np.median(obj.sum(axis=1)[rows]) < 0.15 * mask.shape[1]:   # thin for most of its height
            if best is None or rows.sum() > best.any(axis=1).sum():
                best = obj
    return best, a_star, (x1, y1)


def trace_down(a_star, trunk, width):
    """Follow the trunk down row by row while it is redder than the grass in the same row."""
    ys, xs = np.nonzero(trunk)
    line = np.polyfit(ys, xs, 1)
    half, base, misses = max(2, int(width / 2)), ys.max(), 0
    for y in range(ys.max() + 1, a_star.shape[0]):
        x = int(round(np.polyval(line, y)))
        if x - 6 * half < 0 or x + 6 * half >= a_star.shape[1]:
            break
        row = a_star[y].astype(float)
        grass = np.concatenate((row[x - 6 * half:x - 2 * half], row[x + 2 * half + 1:x + 6 * half + 1]))
        if row[x - half:x + half + 1].mean() - np.median(grass) >= 3:
            base, misses = y, 0
        else:
            misses += 1
            if misses > 3:
                break
    return base, line


def half_max_width(a_star, line, y0, y1, width):
    """Width where the a* profile across the trunk, averaged over rows y0..y1, crosses
    halfway between grass and bark."""
    step = 0.25
    offsets = np.arange(-3 * width, 3 * width + step, step)
    cols = np.arange(a_star.shape[1])
    profile = np.mean([np.interp(np.polyval(line, y) + offsets, cols, a_star[y].astype(float))
                       for y in range(y0, y1)], axis=0)
    grass = np.median(profile[np.abs(offsets) > 1.8 * width])
    bark = np.median(profile[np.abs(offsets) < 0.25 * width])
    half = (grass + bark) / 2
    left = right = int(np.argmin(np.abs(offsets)))              # walk out from the centre to the crossings
    while profile[left] > half:
        left -= 1
    while profile[right] > half:
        right += 1
    x_left = offsets[left] + (half - profile[left]) / (profile[left + 1] - profile[left]) * step
    x_right = offsets[right - 1] + (half - profile[right - 1]) / (profile[right] - profile[right - 1]) * step
    return float(x_right - x_left)


def stick_mark_spacing(img):
    """Mark spacing of the stick in pixels, from the autocorrelation of its brightness."""
    h = img.shape[0]
    bright = (cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2] > 190).astype(np.uint8)
    bright[:int(0.6 * h)] = 0
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bright)
    stick = max((i for i in range(1, n) if stats[i, cv2.CC_STAT_WIDTH] > 12 and stats[i, cv2.CC_STAT_HEIGHT] > 50),
                key=lambda i: stats[i, cv2.CC_STAT_AREA])
    ys, xs = np.nonzero(labels == stick)
    slope, intercept = np.polyfit(ys, xs, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    rows = np.arange(int(0.55 * h), h - 2)
    profile = np.array([gray[y, int(round(slope * y + intercept)) - 3:int(round(slope * y + intercept)) + 4].mean()
                        for y in rows])
    profile -= np.convolve(profile, np.ones(41) / 41, mode="same")
    ac = np.correlate(profile, profile, "full")[len(profile) - 1:]
    k = 40 + int(np.argmax(ac[40:200]))
    k_fine = k + (ac[k - 1] - ac[k + 1]) / (2 * (ac[k - 1] - 2 * ac[k] + ac[k + 1]))   # parabola through the peak
    return k_fine * np.sqrt(1 + slope ** 2)


# One focal length for the camera, from the stick in every photo
paths = sorted(TEST_DIR.glob("*/*.jpg"))
F = float(np.mean([stick_mark_spacing(cv2.imread(str(p))) for p in paths])) * STICK_DISTANCE_M / MARK_SPACING_M

# Measure every photo in pixels
rows = []
for path in paths:
    img = cv2.imread(str(path))
    distance = float(path.parent.name.split(" - ")[1].rstrip("m"))
    canopy = find_canopy(img)
    trunk, a_star, (sx, sy) = find_trunk(img, canopy)
    width = float(np.median(trunk.sum(axis=1)[trunk.any(axis=1)]))     # rough width, for window sizes
    base, line = trace_down(a_star, trunk, width)
    yw = base - int(round(WIDTH_HEIGHT_M * F / distance))      # row WIDTH_HEIGHT_M above the trunk base
    y0 = yw - int(0.75 * width)                                 # block of rows centred on that row
    trunk_px = half_max_width(a_star, line, y0, y0 + int(1.5 * width), width)
    rows.append({"position": path.parent.name, "image": path.name, "distance_m": distance,
                 "height_px": base + sy - canopy["top"], "canopy_px": int(canopy["width"]),
                 "trunk_px": round(trunk_px, 1), "crown_base_px": base + sy - canopy["base"]})

    # Result photo: canopy box, trunk box and width bar
    out = img.copy()
    cv2.rectangle(out, (int(canopy["left"]), int(canopy["top"])), (int(canopy["right"]), int(canopy["base"])), (0, 165, 255), 3)
    xw = np.polyval(line, yw) + sx
    cv2.rectangle(out, (int(xw - trunk_px / 2), int(canopy["base"])), (int(xw + trunk_px / 2), int(base + sy)), (0, 0, 255), 3)
    cv2.line(out, (int(xw - trunk_px / 2), int(yw + sy)), (int(xw + trunk_px / 2), int(yw + sy)), (0, 255, 0), 4)
    cv2.imwrite(str(OUTPUT_DIR / f"{path.parent.name} - {path.name}"), out)

# Pixels to metres
for r in rows:
    scale = r["distance_m"] / F
    r.update(height_m=round(r["height_px"] * scale, 3), canopy_m=round(r["canopy_px"] * scale, 3),
             trunk_mm=round(r["trunk_px"] * scale * 1000, 1), crown_base_m=round(r["crown_base_px"] * scale, 3))

with open(OUTPUT_DIR / "measurements.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

positions = sorted({r["position"] for r in rows})
mean = lambda key, p: np.mean([r[key] for r in rows if r["position"] == p])
spread = lambda key, p: np.ptp([r[key] for r in rows if r["position"] == p])
print(f"Focal length {F:.0f} px\n")
print(f"{'position':20s}{'height m':>10s}{'canopy m':>10s}{'trunk mm':>10s}{'(spread)':>10s}{'crown base m':>14s}")
for p in positions:
    print(f"{p:20s}{mean('height_m', p):10.2f}{mean('canopy_m', p):10.2f}{mean('trunk_mm', p):10.1f}"
          f"{spread('trunk_mm', p):10.1f}{mean('crown_base_m', p):14.2f}")
for key, ref in (("height_m", TAPE_HEIGHT_M), ("canopy_m", TAPE_CANOPY_M), ("trunk_mm", CALLIPER_TRUNK_MM)):
    m = np.mean([mean(key, p) for p in positions])
    print(f"{key:10s} mean {m:.3f}, error {100 * (m / ref - 1):+.1f}%")
d = [mean("distance_m", p) for p in positions]
print("\nGrowth of height from 20 m: " + ", ".join(f"{mean('height_px', p) / mean('height_px', positions[0]):.3f}" for p in positions)
      + "  (expected " + ", ".join(f"{d[0] / x:.3f}" for x in d) + ")")
