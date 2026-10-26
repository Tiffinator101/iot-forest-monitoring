# Step 7: close-up trunk test (2026-10-17).
# Purpose: check whether the trunk width error in the 10 to 20 m photos comes from low
# resolution. The camera was put 140 cm and 90 cm from the trunk, so the trunk is about
# ten times more pixels wide, with the reference stick standing beside the trunk.
#
# Trunk width in pixels: the same idea as step 5. In the lower part of the photo the
# trunk stands against grass, and the Lab a* channel separates them (bark is redder).
# Otsu on a* finds the trunk, a line is fitted through its middle, and the a* profile
# across the trunk is averaged over a block of rows; the width is taken at half maximum.
# The trunk bends, so this is done in several blocks of rows, each with its own line.
#
# Three scales turn pixels into millimetres:
#   1. stick marks: 100 mm apart, stick beside the trunk, so mm/px = 100 / mark spacing
#   2. stick width: 30 mm wide, measured with the same half-maximum method
#   3. pinhole model: mm/px = distance / focal length, with f = 2403 px from step 6 (development/step6_measure_metres.py)
# The result is compared with Tiffany's calliper measurement of the trunk.

import csv
from pathlib import Path
import cv2
import numpy as np

# Settings
IMAGES_DIR = Path(r"C:\Users\tiffa\OneDrive\Documents\University\Engineering\Year 4"
                  r"\Semester 2\Project 448\Project\Code\BaseStation\images")
POSITIONS = {   # camera to trunk distance (m): first and last photo time of the still rig
    1.40: ("2026-10-17_15-47-10", "2026-10-17_15-48-26"),
    0.90: ("2026-10-17_15-48-51", "2026-10-17_15-49-54"),
}
F_CALIBRATED = 2403            # focal length in px, from step 6 (development/step6_measure_metres.py)
MARK_SPACING_MM = 100.0
STICK_WIDTH_MM = 30.0
CALLIPER_MM = 121.0            # Tiffany's calliper reading in the camera's direction
ROW_BLOCKS = 6                 # trunk width is measured in this many blocks of rows
OUTPUT_DIR = Path(__file__).parent / "output" / "step7"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def half_max_width(channel, slope, intercept, y0, y1, half_range, brighter=True):
    """Width of a vertical object from the profile across it, averaged over rows y0..y1
    along the line x = slope * y + intercept, at half maximum (as in step 5)."""
    step = 0.25
    offsets = np.arange(-half_range, half_range + step, step)
    columns = np.arange(channel.shape[1])
    profile = np.mean([np.interp(slope * y + intercept + offsets, columns, channel[y].astype(float))
                       for y in range(y0, y1)], axis=0)
    if not brighter:
        profile = -profile
    outside = np.median(np.concatenate((profile[offsets < -0.8 * half_range], profile[offsets > 0.8 * half_range])))
    inside = np.median(profile[np.abs(offsets) < 0.15 * half_range])
    half = (outside + inside) / 2
    left = right = int(np.argmin(np.abs(offsets)))
    while left > 0 and profile[left] > half:
        left -= 1
    while right < len(profile) - 1 and profile[right] > half:
        right += 1
    if left == 0 or right == len(profile) - 1:
        return None
    x_left = offsets[left] + (half - profile[left]) / (profile[left + 1] - profile[left]) * step
    x_right = offsets[right - 1] + (half - profile[right - 1]) / (profile[right] - profile[right - 1]) * step
    return float((x_right - x_left) * np.cos(np.arctan(slope)))  # width across a leaning object


def find_stick(img):
    """Stick: bright, low-saturation, wide component in the lower part of the photo.
    Returns (slope, intercept) of its centre line."""
    h = img.shape[0]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bright = ((hsv[:, :, 2] > 200) & (hsv[:, :, 1] < 90)).astype(np.uint8) * 255
    bright[:int(0.6 * h)] = 0
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bright)
    stick = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    ys, xs = np.nonzero(labels == stick)
    return np.polyfit(ys, xs, 1)


def stick_mark_spacing(img, slope, intercept):
    """Mark spacing along the stick from the autocorrelation of its brightness (as in step 6)."""
    h = img.shape[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    rows = np.arange(int(0.3 * h), h - 2)
    profile = np.array([gray[y, int(round(slope * y + intercept)) - 3:int(round(slope * y + intercept)) + 4].mean()
                        for y in rows])
    detrended = profile - np.convolve(profile, np.ones(61) / 61, mode="same")
    ac = np.correlate(detrended, detrended, "full")[len(detrended) - 1:]
    ac /= ac[0]
    lo, hi = 60, 450
    k = lo + int(np.argmax(ac[lo:hi]))
    k_fine = k + (ac[k - 1] - ac[k + 1]) / (2 * (ac[k - 1] - 2 * ac[k] + ac[k + 1]))
    return k_fine * np.sqrt(1 + slope ** 2)


def stick_width(img, slope, intercept):
    """Stick width in px from its brightness profile, in rows on the white segments only."""
    h = img.shape[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows = [y for y in range(int(0.62 * h), h - 5) if gray[y, int(round(slope * y + intercept))] > 180]
    blocks = [rows[i:i + 1] for i in range(len(rows))]
    step = 0.25
    offsets = np.arange(-90, 90 + step, step)
    columns = np.arange(gray.shape[1])
    profile = np.mean([np.interp(slope * y + intercept + offsets, columns, gray[y].astype(float)) for y in rows], axis=0)
    outside = np.median(np.concatenate((profile[offsets < -70], profile[offsets > 70])))
    inside = np.median(profile[np.abs(offsets) < 8])
    half = (outside + inside) / 2
    left = right = int(np.argmin(np.abs(offsets)))
    while profile[left] > half:
        left -= 1
    while profile[right] > half:
        right += 1
    x_left = offsets[left] + (half - profile[left]) / (profile[left + 1] - profile[left]) * step
    x_right = offsets[right - 1] + (half - profile[right - 1]) / (profile[right] - profile[right - 1]) * step
    return float((x_right - x_left) * np.cos(np.arctan(slope)))


def trunk_widths(img, stick_line):
    """Trunk width in px, in several blocks of rows in the lower part of the photo."""
    h, w = img.shape[:2]
    a_star = cv2.cvtColor(cv2.GaussianBlur(img, (7, 7), 0), cv2.COLOR_BGR2LAB)[:, :, 1]
    y_top, y_bottom = int(0.62 * h), h - 10                 # trunk against grass, below the pink-tinted strip of grass under the hill line
    region = a_star[y_top:y_bottom].copy()
    # Blank out the stick first: its pink cast makes it redder than the bark in a*,
    # and Otsu would otherwise split the stick from everything else
    grass_level = int(np.median(region))
    for y in range(region.shape[0]):
        x = int(round(np.polyval(stick_line, y + y_top)))
        region[y, max(0, x - 60):x + 60] = grass_level
    _, mask = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    trunk = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_HEIGHT] * stats[i, cv2.CC_STAT_AREA])
    trunk_mask = labels == trunk

    rough = float(np.median(trunk_mask.sum(axis=1)[trunk_mask.any(axis=1)]))
    edges = np.linspace(0, mask.shape[0], ROW_BLOCKS + 1).astype(int)
    widths, bars = [], []
    for b0, b1 in zip(edges[:-1], edges[1:]):
        ys, xs = np.nonzero(trunk_mask[b0:b1])
        if len(ys) < 100:
            continue
        slope, intercept = np.polyfit(ys + b0 + y_top, xs, 1)
        wpx = half_max_width(a_star, slope, intercept, b0 + y_top, b1 + y_top, 1.5 * rough)
        if wpx:
            widths.append(wpx)
            y_mid = (b0 + b1) // 2 + y_top
            bars.append((y_mid, float(slope * y_mid + intercept), wpx))   # for drawing: row, centre x, width
    return widths, trunk_mask, y_top, bars


def draw_result(img, trunk_mask, y_top, bars, stick_line, r, out_path):
    """Annotated copy: trunk mask, width bars, stick line, and a legend with the results."""
    out = img.copy()
    region = out[y_top:y_top + trunk_mask.shape[0]]
    region[trunk_mask] = (0.6 * region[trunk_mask] + 0.4 * np.array([255, 0, 255])).astype(np.uint8)
    cv2.rectangle(out, (0, y_top), (out.shape[1] - 1, y_top + trunk_mask.shape[0]), (0, 255, 255), 2)
    h = out.shape[0]
    cv2.line(out, (int(np.polyval(stick_line, 0)), 0), (int(np.polyval(stick_line, h)), h), (0, 165, 255), 3)
    for y, x, wpx in bars:
        cv2.line(out, (int(round(x - wpx / 2)), y), (int(round(x + wpx / 2)), y), (0, 255, 0), 5)
        cv2.line(out, (int(round(x - wpx / 2)), y - 15), (int(round(x - wpx / 2)), y + 15), (0, 255, 0), 3)
        cv2.line(out, (int(round(x + wpx / 2)), y - 15), (int(round(x + wpx / 2)), y + 15), (0, 255, 0), 3)

    entries = [((255, 0, 255), "Trunk found by Otsu on a*"),
               ((0, 255, 0), f"Trunk width per block of rows, median {r['trunk_px']:.0f} px"),
               ((0, 165, 255), f"Stick centre line, marks {r['mark_spacing_px']:.1f} px apart"),
               ((0, 255, 255), "Region measured (trunk against grass)"),
               (None, f"Trunk {r['trunk_mm_marks']:.1f} mm (stick marks), calliper {CALLIPER_MM:.1f} mm"),
               (None, f"Stick width {r['stick_mm_from_marks']:.1f} mm (true 30 mm)")]
    font, scale, thick, pad, line_len, row_h = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2, 15, 50, 38
    text_w = max(cv2.getTextSize(t, font, scale, thick)[0][0] for _, t in entries)
    box_w, box_h = pad * 3 + line_len + text_w, pad * 2 + row_h * len(entries) - 10
    x0, y0 = out.shape[1] - box_w - 20, 20
    shade = out.copy()
    cv2.rectangle(shade, (x0, y0), (x0 + box_w, y0 + box_h), (40, 40, 40), cv2.FILLED)
    cv2.addWeighted(shade, 0.75, out, 0.25, 0, dst=out)
    for i, (colour, text) in enumerate(entries):
        y = y0 + pad + i * row_h + 14
        if colour:
            cv2.line(out, (x0 + pad, y), (x0 + pad + line_len, y), colour, 4)
        cv2.putText(out, text, (x0 + 2 * pad + line_len, y + 9), font, scale, (255, 255, 255), thick)
    cv2.imwrite(str(out_path), out)


rows = []
for distance, (first, last) in POSITIONS.items():
    for path in sorted(IMAGES_DIR.glob("TreeCam1_2026-10-17_*.jpg")):
        stamp = path.stem.replace("TreeCam1_", "")
        if not first <= stamp <= last:
            continue
        img = cv2.imread(str(path))
        stick_line = find_stick(img)
        spacing = stick_mark_spacing(img, *stick_line)
        s_width = stick_width(img, *stick_line)
        widths, trunk_mask, y_top, bars = trunk_widths(img, stick_line)
        trunk_px = float(np.median(widths))
        rows.append({
            "distance_m": distance, "image": path.name,
            "mark_spacing_px": round(spacing, 1), "stick_width_px": round(s_width, 1),
            "trunk_px": round(trunk_px, 1), "trunk_px_blocks": " ".join(f"{w:.0f}" for w in widths),
            "trunk_mm_marks": round(trunk_px * MARK_SPACING_MM / spacing, 1),
            "trunk_mm_stick_width": round(trunk_px * STICK_WIDTH_MM / s_width, 1),
            "trunk_mm_pinhole": round(trunk_px * distance / F_CALIBRATED * 1000, 1),
            "stick_mm_from_marks": round(s_width * MARK_SPACING_MM / spacing, 1),
        })
        r = rows[-1]
        print(f"{distance:.2f} m {stamp}  marks {spacing:6.1f} px  stick {s_width:5.1f} px  "
              f"trunk {trunk_px:6.1f} px (blocks {r['trunk_px_blocks']})  ->  "
              f"{r['trunk_mm_marks']} mm (marks), {r['trunk_mm_stick_width']} mm (stick width), "
              f"{r['trunk_mm_pinhole']} mm (pinhole)")

        draw_result(img, trunk_mask, y_top, bars, stick_line, r,
                    OUTPUT_DIR / f"{distance:.2f} m - {path.name}")

with open(OUTPUT_DIR / "closeup_trunk.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"\nCalliper: {CALLIPER_MM} mm (121 to 122 mm)")
for distance in POSITIONS:
    sel = [r for r in rows if r["distance_m"] == distance]
    print(f"\n{distance:.2f} m, {len(sel)} photos (mean, range)")
    for key, label in (("trunk_mm_marks", "trunk, stick marks scale"),
                       ("trunk_mm_stick_width", "trunk, stick width scale"),
                       ("trunk_mm_pinhole", "trunk, pinhole f = 2403 px"),
                       ("stick_mm_from_marks", "stick width check (true 30 mm)")):
        v = [r[key] for r in sel]
        ref = 30.0 if key.startswith("stick") else CALLIPER_MM
        print(f"  {label:32s} {np.mean(v):6.1f} mm  ({min(v):.1f} to {max(v):.1f})  "
              f"error {100 * (np.mean(v) / ref - 1):+.1f}%")
