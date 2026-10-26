# Step 6: convert the pixel measurements from step 5 into metres.
#
# Pinhole camera model (workshop Part 3): an object of real size S at distance Z
# appears s = f * S / Z pixels big, where f is the focal length in pixels.
#
#   1. Calibrate f in every photo from the reference stick: its black marks are
#      100 mm apart and the stick stood 2.5 m from the camera, so
#      f = mark_spacing_px * 2.5 / 0.100.
#   2. Size at the tree:  S = s * d_tree / f,  with d_tree the camera-to-trunk
#      distance of that position (20, 17.5, 15, 12.5 or 10 m).
#   3. Compare with the tape-measured values of the tree.
#
# The trunk-scale method from the test plan (cm per px from the taped trunk diameter)
# is also worked out, for comparison.
#
# The mark spacing is found automatically: the stick is the brightest wide object
# against the grass in the lower part of the photo. A line is fitted along it, the
# brightness is sampled along that line, and the repeat distance of the dark marks
# comes from the autocorrelation of that brightness profile.

import csv
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Settings
TEST_DIR = Path(r"C:\Users\tiffa\OneDrive\Documents\University\Engineering\Year 4"
                r"\Semester 2\Project 448\Project\Testing\Tree Photo Test\Analysis Images")
STEP5_CSV = Path(__file__).resolve().parent.parent / "output" / "step5" / "crown_trunk.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "step6"

STICK_DISTANCE_M = 2.5          # camera to stick (Tiffany, 2026-10-04)
MARK_SPACING_M = 0.100          # stick marks, top to top
TAPE_HEIGHT_M = 4.0             # tape-measured reference values of the tree
TAPE_CROWN_M = 3.5
TAPE_CIRCUMFERENCE_M = 0.420
TAPE_DIAMETER_M = TAPE_CIRCUMFERENCE_M / np.pi

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def stick_mark_spacing(img, plot_path=None):
    """Mark spacing of the reference stick in pixels, measured along the stick."""
    h = img.shape[0]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Stick pixels: bright, in the lower 40% of the photo where the background is grass
    bright = (hsv[:, :, 2] > 190).astype(np.uint8) * 255
    bright[:int(0.6 * h)] = 0
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bright)
    # Wider than the measuring tape on the ground, and reasonably tall; keep the biggest
    candidates = [i for i in range(1, n) if stats[i, cv2.CC_STAT_WIDTH] > 12 and stats[i, cv2.CC_STAT_HEIGHT] > 50]
    if not candidates:
        return None
    stick = max(candidates, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    ys, xs = np.nonzero(labels == stick)
    slope, intercept = np.polyfit(ys, xs, 1)                    # stick centre x as a function of y

    # Brightness along the stick (7 px wide), with the slow trend removed
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    rows = np.arange(int(0.55 * h), h - 2)
    profile = np.array([gray[y, int(round(slope * y + intercept)) - 3:int(round(slope * y + intercept)) + 4].mean()
                        for y in rows])
    detrended = profile - np.convolve(profile, np.ones(41) / 41, mode="same")

    # Autocorrelation: the profile matches a shifted copy of itself best when shifted by one mark spacing
    ac = np.correlate(detrended, detrended, "full")[len(detrended) - 1:]
    ac /= ac[0]
    lo, hi = 40, 200
    k = lo + int(np.argmax(ac[lo:hi]))
    k_fine = k + (ac[k - 1] - ac[k + 1]) / (2 * (ac[k - 1] - 2 * ac[k] + ac[k + 1]))   # parabola through the peak
    spacing = k_fine * np.sqrt(1 + slope ** 2)                   # rows to distance along a leaning stick

    if plot_path:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
        ax1.plot(rows, profile, lw=1)
        ax1.set_xlabel("image row (px)")
        ax1.set_ylabel("brightness along stick")
        ax1.set_title("Brightness along the stick (dark dips = marks)")
        ax2.plot(np.arange(len(ac))[:hi + 50], ac[:hi + 50], lw=1)
        ax2.axvline(k_fine, color="r", ls="--", label=f"{k_fine:.1f} rows")
        ax2.set_xlabel("shift (rows)")
        ax2.set_ylabel("autocorrelation")
        ax2.set_title("Autocorrelation: first peak = mark spacing")
        ax2.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
    return spacing


with open(STEP5_CSV) as f:
    step5 = list(csv.DictReader(f))

rows = []
plotted = set()
for r in step5:
    position = r["position"]
    d_tree = float(position.split(" - ")[1].rstrip("m"))
    img = cv2.imread(str(TEST_DIR / position / r["image"]))
    plot = None
    if position not in plotted:
        plotted.add(position)
        plot = OUTPUT_DIR / f"stick profile - {position}.png"
    spacing = stick_mark_spacing(img, plot)
    f_px = spacing * STICK_DISTANCE_M / MARK_SPACING_M          # focal length from the stick

    height_px, crown_px, trunk_px = float(r["tree_height"]), float(r["crown_width"]), float(r["trunk_width"])
    stick_scale = d_tree / f_px                                 # metres per pixel at the tree
    trunk_scale = TAPE_DIAMETER_M / trunk_px                    # metres per pixel from the taped trunk
    rows.append({
        "position": position, "image": r["image"], "distance_m": d_tree,
        "mark_spacing_px": round(spacing, 2), "focal_px": round(f_px, 1),
        "height_px": height_px, "crown_px": crown_px, "trunk_px": trunk_px,
        "crown_length_px": float(r["crown_length"]), "crown_base_px": float(r["crown_base_height"]),
        "height_m": round(height_px * stick_scale, 3),
        "crown_m": round(crown_px * stick_scale, 3),
        "trunk_diameter_mm": round(trunk_px * stick_scale * 1000, 1),
        "height_m_trunk_scale": round(height_px * trunk_scale, 3),
        "crown_m_trunk_scale": round(crown_px * trunk_scale, 3),
    })

# The focal length belongs to the camera, not to the photo, so the main result uses one
# value calibrated from all photos. The photo-to-photo differences in the stick come from
# placing the stick by hand, not from the camera changing.
F_CALIBRATED = float(np.mean([r["focal_px"] for r in rows]))
for r in rows:
    scale = r["distance_m"] / F_CALIBRATED
    r["height_m_calibrated"] = round(r["height_px"] * scale, 3)
    r["crown_m_calibrated"] = round(r["crown_px"] * scale, 3)
    r["trunk_diameter_mm_calibrated"] = round(r["trunk_px"] * scale * 1000, 1)
    r["crown_length_m_calibrated"] = round(r["crown_length_px"] * scale, 3)        # tree top to crown base
    r["crown_base_height_m_calibrated"] = round(r["crown_base_px"] * scale, 3)     # crown base to the ground

with open(OUTPUT_DIR / "measurements_m.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

positions = sorted({r["position"] for r in rows}, key=lambda p: -float(p.split(" - ")[1].rstrip("m")))


def per_position(key):
    return [np.mean([r[key] for r in rows if r["position"] == p]) for p in positions]


def spread(key):
    return [np.ptp([r[key] for r in rows if r["position"] == p]) for p in positions]


print("Stick calibration (mean of 4 photos, spread in brackets)")
for p, s, ds, fp in zip(positions, per_position("mark_spacing_px"), spread("mark_spacing_px"), per_position("focal_px")):
    print(f"  {p:20s} mark spacing {s:6.2f} px ({ds:.2f})   focal length {fp:6.0f} px")

for title, suffix in ((f"one calibrated focal length, f = {F_CALIBRATED:.0f} px (main result)", "_calibrated"),
                      ("focal length from the stick in each photo", "")):
    print(f"\nSizes in metres, {title}")
    print(f"  tape: height {TAPE_HEIGHT_M} m, crown {TAPE_CROWN_M} m, trunk diameter {TAPE_DIAMETER_M * 1000:.1f} mm")
    print(f"  {'position':20s}{'height m':>12s}{'crown m':>12s}{'trunk mm':>12s}")
    for p, hm, cm, tm in zip(positions, per_position("height_m" + suffix), per_position("crown_m" + suffix),
                             per_position("trunk_diameter_mm" + suffix)):
        print(f"  {p:20s}{hm:12.2f}{cm:12.2f}{tm:12.1f}")
    for name, key, ref in (("height", "height_m", TAPE_HEIGHT_M), ("crown", "crown_m", TAPE_CROWN_M),
                           ("trunk", "trunk_diameter_mm", TAPE_DIAMETER_M * 1000)):
        v = per_position(key + suffix)
        print(f"  {name:8s} mean {np.mean(v):.3f}, error vs tape {100 * (np.mean(v) / ref - 1):+.1f}%, "
              f"range over positions {100 * np.ptp(v) / np.mean(v):.1f}%")

print("\nCrown split (calibrated focal length; no tape reference for these)")
print(f"  {'position':20s}{'crown length m':>16s}{'crown base m':>14s}")
for p, cl, cb in zip(positions, per_position("crown_length_m_calibrated"), per_position("crown_base_height_m_calibrated")):
    print(f"  {p:20s}{cl:16.2f}{cb:14.2f}")

print("\nSizes in metres, trunk scale (taped trunk diameter as the ruler, from the test plan)")
for p, hm, cm in zip(positions, per_position("height_m_trunk_scale"), per_position("crown_m_trunk_scale")):
    print(f"  {p:20s}{hm:12.2f}{cm:12.2f}")
for name, key, ref in (("height", "height_m_trunk_scale", TAPE_HEIGHT_M), ("crown", "crown_m_trunk_scale", TAPE_CROWN_M)):
    v = per_position(key)
    print(f"  {name:8s} mean {np.mean(v):.3f}, error vs tape {100 * (np.mean(v) / ref - 1):+.1f}%")

d = [float(p.split(" - ")[1].rstrip("m")) for p in positions]
print("\nApparent growth vs the 20 m baseline (expected from distance: "
      + ", ".join(f"{d[0] / x:.3f}" for x in d) + ")")
for key in ("height_px", "crown_px", "trunk_px"):
    v = per_position(key)
    raw = [x / v[0] for x in v]
    spacing = per_position("mark_spacing_px")
    with_stick = [(x / s) / (v[0] / spacing[0]) for x, s in zip(v, spacing)]
    print(f"  {key:10s} pixels only     " + ", ".join(f"{x:.3f}" for x in raw))
    print(f"  {'':10s} divided by stick " + ", ".join(f"{x:.3f}" for x in with_stick))
