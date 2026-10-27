# Measure one tree photo: the method of development/simple_measure.py as a function, for app.py.
#
#   measure(img, distance_m) -> (results, annotated photo, warnings)
#
# Differences from development/simple_measure.py, so that any single photo can be measured:
#   - The focal length is fixed at 2403 px (calibrated from the stick in the 20 test photos),
#     so the photo does not need the stick in it. That value only holds for the camera node's
#     2048 x 1536 photos, so other sizes give a warning.
#   - A photo with no tree or no trunk raises MeasureError with a short message instead of
#     stopping the program.
#   - The result photo gets a legend and the outline of the YOLOE crown mask.

import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLOE

F_PX = 2403                     # focal length in px, from the stick (development/step6_measure_metres.py)
CAMERA_SIZE = (2048, 1536)      # camera node photo size the focal length belongs to
WIDTH_HEIGHT_M = 0.5            # default height above the trunk base for the trunk width

MODELS_DIR = Path(__file__).resolve().parent / "models"
_model = None


class MeasureError(Exception):
    """The photo could not be measured (no tree, no trunk, ...)."""


def load_model():
    """Load YOLOE once. app.py calls this at start-up; otherwise it happens on the first photo."""
    global _model
    if _model is None:
        # The model files, including YOLOE's text encoder (mobileclip*.ts), are in models/, and Ultralytics
        # looks for them in the working folder, so switch there while the model loads
        cwd = os.getcwd()
        os.chdir(MODELS_DIR)
        try:
            _model = YOLOE("yoloe-26s-seg.pt")
            _model.set_classes(["tree"])
        finally:
            os.chdir(cwd)
    return _model


def find_crown(img):
    """Largest YOLOE tree mask, the rows and columns it covers, and its outline for the result photo."""
    r = load_model().predict(source=img, conf=0.25, retina_masks=True, verbose=False)[0]
    if r.masks is None:
        raise MeasureError("No tree found in the photo.")
    masks = r.masks.data.cpu().numpy() > 0.5
    mask = masks[masks.sum(axis=(1, 2)).argmax()]
    ys, xs = np.nonzero(mask)
    outline, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return {"outline": outline, "top": ys.min(), "base": ys.max(), "left": xs.min(), "right": xs.max(),
            "width": xs.max() - xs.min() + 1, "height": ys.max() - ys.min() + 1}


def find_trunk(img, crown):
    """Trunk mask in a strip below the crown: a*, Otsu, opening and closing, tallest thin object."""
    cx = (crown["left"] + crown["right"]) / 2
    x1 = int(max(0, cx - 0.3 * crown["width"]))
    x2 = int(min(img.shape[1], cx + 0.3 * crown["width"]))
    y1 = int(max(0, crown["base"] - 0.1 * crown["height"]))
    y2 = int(min(img.shape[0], crown["base"] + 0.6 * crown["height"]))
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
    if best is None:
        raise MeasureError("Tree found, but no trunk below the crown.")
    return best, a_star, (x1, y1, x2, y2)


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
    while left > 0 and profile[left] > half:
        left -= 1
    while right < len(profile) - 1 and profile[right] > half:
        right += 1
    if left == 0 or right == len(profile) - 1:
        raise MeasureError("Trunk edges not found at the width height.")
    x_left = offsets[left] + (half - profile[left]) / (profile[left + 1] - profile[left]) * step
    x_right = offsets[right - 1] + (half - profile[right - 1]) / (profile[right] - profile[right - 1]) * step
    return float(x_right - x_left)


# Drawing colours (OpenCV order: blue, green, red) and legend labels. Chosen to stand out against the
# camera's pink and blue sky, the green leaves and grass, and the brown bark.
COLOURS = [((0, 0, 255), "Crown box"),         # red
           ((0, 255, 255), "Crown outline"),   # yellow
           ((0, 0, 0), "Trunk box"),           # black
           ((255, 255, 0), "Trunk width")]     # cyan


def draw_legend(img):
    """Light legend in the bottom-right corner, which is grass in the camera node's photos. Each colour
    line has a black edge so the yellow and cyan lines show on the light background. Sized so the text
    can be read when the photo is printed at half the report's page width, while the legend still stays
    below the trunk base in the 10 m test photos."""
    font, scale, thick, pad, line_len, row_h = cv2.FONT_HERSHEY_SIMPLEX, 1.8, 4, 33, 112, 84
    text_w = max(cv2.getTextSize(t, font, scale, thick)[0][0] for _, t in COLOURS)
    box_w, box_h = pad * 3 + line_len + text_w, pad * 2 + row_h * len(COLOURS) - 21
    x0, y0 = img.shape[1] - box_w - 20, img.shape[0] - box_h - 20
    shade = img.copy()
    cv2.rectangle(shade, (x0, y0), (x0 + box_w, y0 + box_h), (235, 235, 235), cv2.FILLED)
    cv2.addWeighted(shade, 0.85, img, 0.15, 0, dst=img)
    for i, (colour, text) in enumerate(COLOURS):
        y = y0 + pad + i * row_h + 30
        cv2.line(img, (x0 + pad, y), (x0 + pad + line_len, y), (0, 0, 0), 16)
        cv2.line(img, (x0 + pad, y), (x0 + pad + line_len, y), colour, 9)
        cv2.putText(img, text, (x0 + 2 * pad + line_len, y + 20), font, scale, (0, 0, 0), thick)


def measure(img, distance_m, width_height_m=WIDTH_HEIGHT_M, f_px=F_PX):
    """Measure one photo (OpenCV BGR image) of a tree distance_m from the camera.
    Returns (results dict, annotated BGR photo, list of warnings)."""
    if distance_m <= 0:
        raise MeasureError("The distance to the tree must be more than 0 m.")
    warnings = []
    if (img.shape[1], img.shape[0]) != CAMERA_SIZE:
        warnings.append(f"The photo is {img.shape[1]} x {img.shape[0]} px, not the camera node's "
                        f"{CAMERA_SIZE[0]} x {CAMERA_SIZE[1]} px, so the {f_px} px focal length "
                        "does not apply and the metre values are wrong.")

    crown = find_crown(img)
    trunk, a_star, (sx, sy, _, _) = find_trunk(img, crown)
    width = float(np.median(trunk.sum(axis=1)[trunk.any(axis=1)]))     # rough width, for window sizes
    base, line = trace_down(a_star, trunk, width)

    # Trunk width in a block of rows centred width_height_m above the trunk base
    yw = base - int(round(width_height_m * f_px / distance_m))
    y0, y1 = yw - int(0.75 * width), yw - int(0.75 * width) + int(1.5 * width)
    if y0 < 0 or yw + sy <= crown["base"]:
        raise MeasureError(f"{width_height_m} m above the trunk base is inside the crown. "
                           "Choose a lower width height.")
    trunk_px = half_max_width(a_star, line, y0, y1, width)

    scale = distance_m / f_px                                   # metres per pixel at the tree
    height_px = int(base + sy - crown["top"])
    crown_px = int(crown["base"] - crown["top"])              # tree top to crown base
    crown_base_px = int(base + sy - crown["base"])           # trunk base (ground) to crown base
    results = {
        "Tree height (m)": round(height_px * scale, 2),
        "Crown width (m)": round(int(crown["width"]) * scale, 2),
        "Trunk width (mm)": round(trunk_px * scale * 1000, 1),
        "Crown length (m)": round(crown_px * scale, 2),
        "Height to crown base (m)": round(crown_base_px * scale, 2),
        "Tree height (px)": height_px,
        "Crown width (px)": int(crown["width"]),
        "Trunk width (px)": round(trunk_px, 1),
        "Crown length (px)": crown_px,
        "Height to crown base (px)": crown_base_px,
        "Pixel size at the tree (mm)": round(scale * 1000, 1),
    }

    out = img.copy()
    cv2.drawContours(out, crown["outline"], -1, COLOURS[1][0], 2)    # under the boxes, so they stay on top
    cv2.rectangle(out, (int(crown["left"]), int(crown["top"])), (int(crown["right"]), int(crown["base"])),
                  COLOURS[0][0], 3)
    xw = np.polyval(line, yw) + sx
    cv2.rectangle(out, (int(xw - trunk_px / 2), int(crown["base"])), (int(xw + trunk_px / 2), int(base + sy)),
                  COLOURS[2][0], 3)
    cv2.line(out, (int(xw - trunk_px / 2), int(yw + sy)), (int(xw + trunk_px / 2), int(yw + sy)), COLOURS[3][0], 4)
    draw_legend(out)
    return results, out, warnings
