# Step 5: find the canopy with YOLOE and the trunk with classical image processing.
#
# Canopy (YOLOE + convex hull):
#   1. yoloe-26s-seg with the prompt "tree" gives a canopy-only mask (step 4c).
#   2. Keep the largest mask, so a distant bush can never be picked instead.
#   3. Wrap it in a convex hull (workshop Part 3) for a clean outline:
#      tree top, canopy width, canopy bottom and canopy area.
#
# Trunk (workshop Part 3/4 pipeline in a strip below the canopy):
#   1. Crop a strip under the middle of the canopy (region of interest).
#   2. Gaussian blur, then convert to Lab colour space and take the a* channel
#      (green to red axis): grass is green (low a*), bark is reddish (high a*).
#   3. Otsu threshold on a* gives a binary mask, cleaned with opening and closing.
#   4. Find contours and keep the tallest one that is thin for most of its height: the trunk.
#   5. Trunk base: traced down row by row from the trunk contour (shade can hide the
#      lower trunk from Otsu).
#   6. Trunk width: measured just below the red tape band (where the circumference was
#      taped), from the averaged a* profile across the trunk at half maximum. Canny
#      edges (workshop Part 4) are measured at the same rows as a cross-check.
#
# Tree height (px) = trunk base - tree top. All values are in pixels; converting
# to metres is the next step.

import csv
import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLOE

# Settings
TEST_DIR = Path(r"C:\Users\tiffa\OneDrive\Documents\University\Engineering\Year 4"
                r"\Semester 2\Project 448\Project\Testing\Tree Photo Test\Analysis Images")
CONF = 0.25
STRIP_HALF_WIDTH = 0.3   # strip spans canopy centre ± 0.3 x canopy width
STRIP_ABOVE = 0.1        # strip starts 0.1 x canopy height above the canopy bottom
STRIP_BELOW = 0.6        # and ends 0.6 x canopy height below it
TRACE_LOW_LEAVES = False  # True: extend the canopy base into low leaves against grass (texture
                          # tracing). False: canopy base = bottom of the YOLOE canopy outline.
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "step5"

# The model files, including YOLOE's text encoder (mobileclip*.ts), are in ../models, and
# Ultralytics looks for them in the working folder
os.chdir(Path(__file__).resolve().parent.parent / "models")
model = YOLOE("yoloe-26s-seg.pt")
model.set_classes(["tree"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "stages").mkdir(exist_ok=True)


def find_canopy(img):
    """Largest YOLOE tree mask, wrapped in a convex hull."""
    r = model.predict(source=img, conf=CONF, retina_masks=True, verbose=False)[0]
    if r.masks is None:
        return None
    masks = r.masks.data.cpu().numpy() > 0.5
    mask = masks[masks.sum(axis=(1, 2)).argmax()].astype(np.uint8) * 255   # largest mask

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    x, y, w, h = cv2.boundingRect(hull)
    return {"hull": hull, "mask": mask, "top": y, "bottom": y + h - 1, "left": x, "right": x + w - 1,
            "width": w, "height": h, "hull_area": int(cv2.contourArea(hull)),
            "mask_area": int((mask > 0).sum())}


def trace_canopy_base(img, canopy, trunk):
    """Extend the canopy mask down into low leaves that hang in front of the grass.
    YOLOE separates leaves from sky well, but stops where grass is behind the leaves
    (green on green). Colour can't tell the two apart here, but texture can: leaves are
    a busy pattern of small light and dark specks, grass at this distance looks smooth.
      1. Speckle: the photo minus its median-filtered copy (workshop Part 3). A median
         filter removes small specks but keeps edges, so the hill line and the edges of
         the tree's shadow score low and leaf texture scores high.
      2. Threshold halfway between the speckle of grass (below the trunk base) and of the
         lower canopy, so it adapts to each photo's light and sharpness.
      3. Only look in a window under the canopy, and skip the trunk columns (bark is busy too).
      4. Keep compact clumps that touch the canopy mask; long thin streaks are edges, not leaves.
    Returns (canopy base row, mask of the added leaf pixels)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    speckle = cv2.blur(np.abs(gray.astype(np.float32) - cv2.medianBlur(gray, 5).astype(np.float32)), (9, 9))
    mask = canopy["mask"] > 0
    bottom, height = canopy["bottom"], canopy["height"]
    cols = slice(canopy["left"], canopy["right"] + 1)

    grass = np.median(speckle[trunk["base"] + 20:trunk["base"] + 120, cols])
    low_rows = slice(bottom - int(0.15 * height), bottom)
    leaf = np.median(speckle[low_rows][mask[low_rows]])
    busy = (speckle > (grass + leaf) / 2).astype(np.uint8)

    window = np.zeros_like(busy)
    window[bottom - 3:bottom + int(0.3 * height), cols] = 1
    half = int(1.5 * trunk["width"])
    window[:, int(trunk["centre_x"]) - half:int(trunk["centre_x"]) + half + 1] = 0
    busy = cv2.morphologyEx(busy * window, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    touching = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(busy)
    leaves = np.zeros(busy.shape, bool)
    for k in range(1, n):
        w, h, area = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT], stats[k, cv2.CC_STAT_AREA]
        component = labels == k
        if not touching[component].any():
            continue
        if area / (w * h) < 0.35 or w > 4 * h:                  # long thin streak: an edge
            continue
        leaves |= component

    # Canopy base: lowest row with at least 1% of the canopy width covered, so a stray
    # pixel can't set it
    covered = (mask | leaves).sum(axis=1)
    base = int(np.where(covered >= max(3, int(0.01 * canopy["width"])))[0].max())
    return base, leaves


def trace_trunk_down(a_star, trunk_mask, width, min_contrast=3, misses_allowed=3):
    """Follow the trunk below where Otsu lost it (e.g. the shaded lower trunk).
    A straight line is fitted through the trunk pixels found so far. Going down one
    row at a time, the a* value on that line is compared with the grass on both sides
    in the same row. The trunk continues while it is clearly redder than its own row's
    grass, so a darker row does not stop it the way one global threshold does."""
    ys, xs = np.nonzero(trunk_mask)
    slope, intercept = np.polyfit(ys, xs, 1)                    # trunk centre x as a function of y
    half = max(2, int(width / 2))
    base, misses = int(ys.max()), 0
    for y in range(int(ys.max()) + 1, a_star.shape[0]):
        x = int(round(slope * y + intercept))
        if x - 6 * half < 0 or x + 6 * half >= a_star.shape[1]:
            break
        row = a_star[y].astype(float)
        trunk_value = row[x - half:x + half + 1].mean()
        grass = np.concatenate((row[x - 6 * half:x - 2 * half], row[x + 2 * half + 1:x + 6 * half + 1]))
        if trunk_value - np.median(grass) >= min_contrast:
            base, misses = y, 0
        else:
            misses += 1
            if misses > misses_allowed:
                break
    return base


def find_band(a_star, line, top, base, a_width):
    """Find the red tape band on the trunk: the rows where a* on the trunk line peaks
    above the trunk's own local level. Returns (first_row, last_row)."""
    slope, intercept = line
    ys = np.arange(top, base)
    centre = np.array([a_star[y, int(round(slope * y + intercept)) - 1:
                              int(round(slope * y + intercept)) + 2].mean() for y in ys])
    k = int(2 * a_width)
    local = np.array([np.median(centre[max(0, i - k):i + k + 1]) for i in range(len(centre))])
    excess = centre - local
    peak = int(np.argmax(excess))
    if excess[peak] < 8:                                        # nothing clearly redder than the bark
        return None
    # The band is the rows around the peak that are above half the peak excess,
    # the same half-maximum idea as the width measurement, so a weaker tape still gives
    # the same band edges
    first = last = peak
    while first > 0 and excess[first - 1] > excess[peak] / 2:
        first -= 1
    while last < len(excess) - 1 and excess[last + 1] > excess[peak] / 2:
        last += 1
    return int(ys[first]), int(ys[last])


def width_profile(a_star, line, y0, y1, a_width):
    """Trunk width from the average a* profile across the trunk (full width at half maximum).
    Each row between y0 and y1 is sampled across the trunk, centred on the fitted trunk line,
    and the rows are averaged so bark texture cancels out. The width is where the averaged
    profile crosses halfway between the grass level and the trunk level, found to a
    fraction of a pixel by linear interpolation."""
    slope, intercept = line
    step = 0.25
    offsets = np.arange(-3 * a_width, 3 * a_width + step, step)
    columns = np.arange(a_star.shape[1])
    profile = np.mean([np.interp(slope * y + intercept + offsets, columns, a_star[y].astype(float))
                       for y in range(y0, y1)], axis=0)
    grass = np.median(np.concatenate((profile[offsets < -1.8 * a_width], profile[offsets > 1.8 * a_width])))
    trunk = np.median(profile[np.abs(offsets) < 0.25 * a_width])
    half = (grass + trunk) / 2
    left = right = int(np.argmin(np.abs(offsets)))
    while left > 0 and profile[left] > half:
        left -= 1
    while right < len(profile) - 1 and profile[right] > half:
        right += 1
    x_left = offsets[left] + (half - profile[left]) / (profile[left + 1] - profile[left]) * step
    x_right = offsets[right - 1] + (half - profile[right - 1]) / (profile[right] - profile[right - 1]) * step
    return float(x_right - x_left)


def width_canny(a_star, trunk_mask, y0, y1, a_width):
    """Cross-check with Canny edges (workshop Part 4). a* is stretched to 0-255 and the
    Canny thresholds come from Otsu. In each row the Canny edge closest to each side of
    the a* trunk outline is taken, and the median row width is returned."""
    stretched = cv2.normalize(a_star, None, 0, 255, cv2.NORM_MINMAX)
    high, _ = cv2.threshold(stretched, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(stretched, 0.5 * high, high)
    widths = []
    for y in range(y0, y1):
        xs = np.where(trunk_mask[y])[0]
        if len(xs) == 0:
            continue
        e = np.where(edges[y])[0]
        left = e[np.abs(e - xs.min()) <= 0.4 * a_width]
        right = e[np.abs(e - xs.max()) <= 0.4 * a_width]
        if len(left) and len(right):
            widths.append(right[np.argmin(np.abs(right - xs.max()))] - left[np.argmin(np.abs(left - xs.min()))])
    return (float(np.median(widths)) if widths else None), edges


def find_trunk(img, canopy, stages_prefix=None):
    """Classical trunk segmentation in a strip below the canopy."""
    cx = (canopy["left"] + canopy["right"]) / 2
    sx1 = int(max(0, cx - STRIP_HALF_WIDTH * canopy["width"]))
    sx2 = int(min(img.shape[1], cx + STRIP_HALF_WIDTH * canopy["width"]))
    sy1 = int(max(0, canopy["bottom"] - STRIP_ABOVE * canopy["height"]))
    sy2 = int(min(img.shape[0], canopy["bottom"] + STRIP_BELOW * canopy["height"]))
    strip = img[sy1:sy2, sx1:sx2]

    blur = cv2.GaussianBlur(strip, (5, 5), 0)
    a_star = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)[:, :, 1]
    _, thresh = cv2.threshold(a_star, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)    # erode then dilate: removes specks
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)    # dilate then erode: fills small gaps

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h < 0.3 * strip.shape[0]:                             # too short to be the trunk
            continue
        blob = np.zeros_like(clean)
        cv2.drawContours(blob, [c], -1, 255, cv2.FILLED)
        widths = (blob[y:y + h] > 0).sum(axis=1)
        if np.median(widths) < 0.15 * strip.shape[1]:            # thin for most of its height
            candidates.append((h, c, blob))
    trunk, edges = None, None
    if candidates:
        _, contour, trunk_mask = max(candidates, key=lambda t: t[0])
        rows = np.where(trunk_mask.any(axis=1))[0]
        a_width = float(np.median((trunk_mask[rows] > 0).sum(axis=1)))   # rough width, used for window sizes
        base = trace_trunk_down(a_star, trunk_mask, a_width)

        # Width is measured just below the tape band, where the circumference was taped
        ys, xs = np.nonzero(trunk_mask)
        line = np.polyfit(ys, xs, 1)
        band = find_band(a_star, line, int(rows.min()), int(rows.max()), a_width)
        y0 = band[1] + 3 if band else int((rows.min() + rows.max()) / 2)
        y1 = min(int(rows.max()) - 3, y0 + int(1.5 * a_width))
        width = width_profile(a_star, line, y0, y1, a_width)
        canny, edges = width_canny(a_star, trunk_mask, y0, y1, a_width)

        trunk = {"contour": contour + [sx1, sy1],                # shift back to full-photo coordinates
                 "otsu_bottom": int(rows.max()) + sy1,
                 "base": base + sy1,
                 "band": (band[0] + sy1, band[1] + sy1) if band else None,
                 "width_rows": (y0 + sy1, y1 + sy1),
                 "centre_x": float(np.polyval(line, (y0 + y1) / 2)) + sx1,
                 "width": width, "width_canny": canny, "width_a_star": a_width}

    if stages_prefix:
        cv2.imwrite(f"{stages_prefix} 1 strip.png", strip)
        cv2.imwrite(f"{stages_prefix} 2 a-star.png", cv2.normalize(a_star, None, 0, 255, cv2.NORM_MINMAX))
        cv2.imwrite(f"{stages_prefix} 3 otsu.png", thresh)
        cv2.imwrite(f"{stages_prefix} 4 cleaned.png", clean)
        overlay = strip.copy()
        if trunk:
            cv2.drawContours(overlay, [trunk["contour"] - [sx1, sy1]], -1, COLOURS["trunk"][0], 2)
            cv2.line(overlay, (0, trunk["base"] - sy1), (overlay.shape[1], trunk["base"] - sy1), COLOURS["trunk_box"][0], 1)
            draw_width(overlay, trunk, -sx1, -sy1, thickness=2)
        cv2.imwrite(f"{stages_prefix} 5 trunk.png", overlay)
        if edges is not None:
            cv2.imwrite(f"{stages_prefix} 6 canny.png", edges)
    return trunk, (sx1, sy1, sx2, sy2)


# Drawing colours (OpenCV uses blue, green, red order) and their legend labels
COLOURS = {
    "canopy":     ((255, 0, 0),   "Canopy outline (YOLOE mask, convex hull)"),
    "leaves":     ((255, 255, 0), "Low leaves added by texture tracing"),
    "canopy_box": ((0, 165, 255), "Canopy box (tree top, canopy sides, canopy base)"),
    "trunk_box":  ((0, 0, 255),   "Trunk box (canopy base to traced trunk base)"),
    "strip":      ((0, 255, 255), "Trunk search strip"),
    "trunk":      ((255, 0, 255), "Trunk found by Otsu on a*"),
    "width":      ((0, 255, 0),   "Trunk width (below tape band)"),
}


def draw_width(img, trunk, dx=0, dy=0, thickness=3):
    """Bar across the trunk showing the measured width, at the rows it was measured."""
    y = int((trunk["width_rows"][0] + trunk["width_rows"][1]) / 2) + dy
    x = trunk["centre_x"] + dx
    cv2.line(img, (int(round(x - trunk["width"] / 2)), y), (int(round(x + trunk["width"] / 2)), y),
             COLOURS["width"][0], thickness)


def draw_legend(img, keys):
    """Legend box in the top-right corner: a short line in each colour with its label."""
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    pad, line_len, row_h = 15, 50, 38
    text_w = max(cv2.getTextSize(COLOURS[k][1], font, scale, thick)[0][0] for k in keys)
    box_w, box_h = pad * 3 + line_len + text_w, pad * 2 + row_h * len(keys) - 10
    x0, y0 = img.shape[1] - box_w - 20, 20
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (40, 40, 40), cv2.FILLED)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, dst=img)          # semi-transparent dark box
    for i, k in enumerate(keys):
        y = y0 + pad + i * row_h + 14
        cv2.line(img, (x0 + pad, y), (x0 + pad + line_len, y), COLOURS[k][0], 4)
        cv2.putText(img, COLOURS[k][1], (x0 + 2 * pad + line_len, y + 9), font, scale, (255, 255, 255), thick)


def draw_result(img, canopy, trunk, strip_box, out_path):
    out = img.copy()
    cv2.rectangle(out, strip_box[:2], strip_box[2:], COLOURS["strip"][0], 2)
    cv2.drawContours(out, [canopy["hull"]], -1, COLOURS["canopy"][0], 3)
    # Canopy box: top = tree top, sides = widest canopy points, bottom = canopy base
    keys = ["canopy"]
    if canopy.get("low_leaves") is not None and canopy["low_leaves"].any():
        out[canopy["low_leaves"]] = COLOURS["leaves"][0]
        keys.append("leaves")
    cv2.rectangle(out, (canopy["left"], canopy["top"]), (canopy["right"], canopy["base"]),
                  COLOURS["canopy_box"][0], 3)
    keys.append("canopy_box")
    if trunk:
        cv2.drawContours(out, [trunk["contour"]], -1, COLOURS["trunk"][0], 3)
        # Trunk box: from the canopy base down to the traced trunk base, as wide as the measured trunk
        half = trunk["width"] / 2
        cv2.rectangle(out, (int(round(trunk["centre_x"] - half)), canopy["base"]),
                      (int(round(trunk["centre_x"] + half)), trunk["base"]), COLOURS["trunk_box"][0], 3)
        draw_width(out, trunk, thickness=4)
        keys += ["trunk_box", "trunk", "width"]
    keys.append("strip")
    draw_legend(out, keys)
    cv2.imwrite(str(out_path), out)


rows = []
first_of_position = set()
for image_path in sorted(TEST_DIR.glob("*/*.jpg")):
    position = image_path.parent.name
    img = cv2.imread(str(image_path))
    row = {"position": position, "image": image_path.name}

    canopy = find_canopy(img)
    if canopy is None:
        print(f"{position} / {image_path.name}: no canopy found")
        rows.append(row)
        continue

    # Save every pipeline stage for the first photo of each position
    stages = None
    if position not in first_of_position:
        first_of_position.add(position)
        stages = str(OUTPUT_DIR / "stages" / position)
    trunk, strip_box = find_trunk(img, canopy, stages)
    canopy["base"], canopy["low_leaves"] = (trace_canopy_base(img, canopy, trunk) if trunk and TRACE_LOW_LEAVES
                                            else (canopy["bottom"], None))

    row.update(tree_top=canopy["top"], canopy_bottom=canopy["bottom"], canopy_base=canopy["base"],
               canopy_left=canopy["left"], canopy_right=canopy["right"], canopy_width=canopy["width"],
               crown_length=canopy["base"] - canopy["top"],                   # tree top to canopy base
               canopy_hull_area=canopy["hull_area"], canopy_mask_area=canopy["mask_area"])
    if trunk:
        row.update(trunk_base=trunk["base"], otsu_bottom=trunk["otsu_bottom"], trunk_width=round(trunk["width"], 1),
                   trunk_width_canny=trunk["width_canny"] if trunk["width_canny"] is not None else "",
                   trunk_width_a_star=trunk["width_a_star"],
                   band_top=trunk["band"][0] if trunk["band"] else "", band_bottom=trunk["band"][1] if trunk["band"] else "",
                   tree_height=trunk["base"] - canopy["top"],
                   crown_base_height=trunk["base"] - canopy["base"])           # canopy base to the ground
    rows.append(row)
    draw_result(img, canopy, trunk, strip_box, OUTPUT_DIR / f"{position} - {image_path.name}")
    print(f"{position:20s} {image_path.stem[-8:]}  top={canopy['top']:4d}  "
          f"trunk_base={row.get('trunk_base', '-'):>5}  height={row.get('tree_height', '-'):>5}  "
          f"canopy_w={canopy['width']:4d}  trunk_w={row.get('trunk_width', '-'):>5}  "
          f"(canny {row.get('trunk_width_canny', '-')}, a* {row.get('trunk_width_a_star', '-')})  "
          f"band={row.get('band_top', '-')}-{row.get('band_bottom', '-')}  hull_area={canopy['hull_area']}")

fields = ["position", "image", "tree_top", "canopy_bottom", "canopy_base", "trunk_base", "otsu_bottom",
          "tree_height", "crown_length", "crown_base_height",
          "canopy_left", "canopy_right", "canopy_width", "trunk_width", "trunk_width_canny", "trunk_width_a_star",
          "band_top", "band_bottom", "canopy_hull_area", "canopy_mask_area"]
with open(OUTPUT_DIR / "canopy_trunk.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

KEYS = ("tree_top", "canopy_base", "trunk_base", "tree_height", "crown_length", "crown_base_height",
        "canopy_width", "trunk_width", "trunk_width_canny")
print("\nMean (spread = max - min) across the 4 photos at each position, in px")
print(f"{'position':20s}" + "".join(f"{k:>18s}" for k in KEYS))
means = {}
for position in sorted({r["position"] for r in rows}):
    line = f"{position:20s}"
    for key in KEYS:
        v = [r[key] for r in rows if r["position"] == position and r.get(key, "") != ""]
        decimals = 1 if key.startswith("trunk_width") else 0
        line += f"{f'{np.mean(v):.{decimals}f} ({max(v) - min(v):.{decimals}f})' if len(v) == 4 else f'{len(v)}/4':>18}"
        means.setdefault(key, []).append(np.mean(v) if v else np.nan)
    print(line)

# Sanity check: apparent size goes as 1/distance, so each size should grow by d_first / d
distances = [float(p.split(" - ")[1].rstrip("m")) for p in sorted({r["position"] for r in rows})]
print("\nGrowth factor vs the first position (expected from distance: "
      + ", ".join(f"{distances[0] / d:.2f}" for d in distances) + ")")
for key in ("tree_height", "crown_length", "crown_base_height", "canopy_width", "trunk_width", "trunk_width_canny"):
    print(f"  {key:18s} " + ", ".join(f"{m / means[key][0]:.2f}" for m in means[key]))
